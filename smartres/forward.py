"""SmartRes vision forward: low-res encode, route, selective re-encode, assemble."""

from dataclasses import dataclass
from typing import List, Optional

import torch

from .assembly import order_preserving_assembly
from .grid import high_res_owner, native_to_raster_maps, window_to_native
from .windowing import (
    full_window_context,
    run_blocks,
    selective_window_context,
    snap_to_windows,
)


@dataclass
class VisionOutput:
    """Merged visual tokens plus the numbers a caller legitimately needs."""

    tokens: torch.Tensor              # [sum_b L_b, D_llm] merger output for the batch
    lengths: List[int]                # per sample, assembled length BEFORE the merger
    activated_ratio: float            # fraction of low-res patches routed to high res
    high_res_encoded: int             # high-res patches actually pushed through the tower
    high_res_total: int               # high-res patches in the full frame
    loss_route: Optional[torch.Tensor] = None
    loss_hinge: Optional[torch.Tensor] = None

    @property
    def encode_fraction(self) -> float:
        return self.high_res_encoded / max(self.high_res_total, 1)


def smartres_vision_forward(
    vision_tower,
    low_res_pixels: torch.Tensor,
    low_res_grid: torch.Tensor,
    high_res_pixels: torch.Tensor,
    high_res_grid: torch.Tensor,
    router,
    router_layer: int,
    encode_snap: str = "window",
    route_target: Optional[torch.Tensor] = None,
) -> VisionOutput:
    """Encode one batch of images with dynamic resolution routing."""
    if encode_snap not in ("window", "unit"):
        raise ValueError(f"encode_snap must be 'window' or 'unit', got {encode_snap!r}")

    device = low_res_pixels.device
    merge = vision_tower.spatial_merge_size
    merge_unit = merge * merge

    # --- 1. low-resolution encoding, up to the routing layer ----------------------
    low_res = vision_tower.patch_embed(low_res_pixels)
    low_res_ctx = full_window_context(vision_tower, low_res, low_res_grid)
    intermediate = run_blocks(vision_tower, low_res_ctx["hidden_states"], low_res_ctx, 0, router_layer)

    # --- 2. routing ---------------------------------------------------------------
    # The router reads patch-raster order, which is what the supervision is defined in;
    native_to_raster, raster_to_native = native_to_raster_maps(low_res_grid, merge, device)
    in_raster = window_to_native(intermediate, low_res_ctx["window_index"], merge_unit)[raster_to_native]
    routing = router(in_raster, route_target)
    route_mask = routing.mask[native_to_raster] > 0                 # back to native order

    # --- 3. selective re-encoding --------------------------------------------------
    batch = low_res_grid.shape[0]
    low_dims = [tuple(int(v) for v in low_res_grid[b].tolist()) for b in range(batch)]
    high_dims = [tuple(int(v) for v in high_res_grid[b].tolist()) for b in range(batch)]
    high_res_total = sum(T * H * W for (T, H, W) in high_dims)

    owners, keep_parts = [], []
    low_offset = high_offset = 0
    for b in range(batch):
        t, h, w = low_dims[b]
        T, H, W = high_dims[b]
        n_low, n_high = t * h * w, T * H * W
        owner = high_res_owner(
            route_mask[low_offset:low_offset + n_low], t, h, w, T, H, W, merge, merge_unit, device
        )
        owners.append(owner)
        # Round the cells the assembly will read up to whole merge units.
        needed = torch.nonzero(owner >= 0, as_tuple=False).flatten()
        units = torch.unique(needed // merge_unit)
        keep_parts.append(
            (units[:, None] * merge_unit + torch.arange(merge_unit, device=device)[None, :]).reshape(-1)
            + high_offset
        )
        low_offset += n_low
        high_offset += n_high

    keep_patches = torch.sort(torch.cat(keep_parts))[0] if keep_parts else torch.empty(0, dtype=torch.long, device=device)
    keep_units = torch.zeros(high_res_total // merge_unit, dtype=torch.bool, device=device)
    keep_units[keep_patches // merge_unit] = True

    if encode_snap == "window":
        # Only ever adds units, so `owners` -- and hence the assembled sequence -- is
        # untouched; the extra patches exist to restore each window's attention context.
        keep_units, keep_patches = snap_to_windows(
            vision_tower, keep_units, high_res_grid, merge_unit, device
        )
    
    high_res_features = high_res_pixels.new_zeros(
        (high_res_total, vision_tower.config.hidden_size),
        dtype=vision_tower.merger.mlp[0].weight.dtype,
    )
    n_encoded = int(keep_patches.numel())
    if n_encoded:
        kept = vision_tower.patch_embed(high_res_pixels[keep_patches]) # encode high-res patches
        ctx = selective_window_context(vision_tower, kept, high_res_grid, keep_units, merge_unit)
        feats = run_blocks(vision_tower, ctx["hidden_states"], ctx, 0, len(vision_tower.blocks))
        # Undo the windowed permutation over the survivors only.
        order = torch.argsort(ctx["window_index"][ctx["kept_slots"]])
        feats = feats.reshape(-1, merge_unit, feats.shape[-1])[order].reshape(-1, feats.shape[-1])
        high_res_features[keep_patches] = feats.to(high_res_features.dtype)

    # --- 4. low-resolution tail, then assembly -------------------------------------
    low_res_features = run_blocks(
        vision_tower, intermediate, low_res_ctx, router_layer, len(vision_tower.blocks)
    )
    low_res_features = window_to_native(low_res_features, low_res_ctx["window_index"], merge_unit)

    chunks, lengths = [], []
    low_offset = high_offset = 0
    for b in range(batch):
        t, h, w = low_dims[b]
        T, H, W = high_dims[b]
        n_low, n_high = t * h * w, T * H * W
        assembled = order_preserving_assembly(
            low_res_features[low_offset:low_offset + n_low],
            high_res_features[high_offset:high_offset + n_high],
            route_mask[low_offset:low_offset + n_low],
            owners[b],
            merge_unit,
        )
        chunks.append(assembled.tokens)
        lengths.append(assembled.tokens.size(0))
        low_offset += n_low
        high_offset += n_high

    sequence = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    target_dtype = vision_tower.merger.mlp[0].weight.dtype
    if sequence.dtype != target_dtype:
        sequence = sequence.to(target_dtype)

    return VisionOutput(
        tokens=vision_tower.merger(sequence),
        lengths=lengths,
        activated_ratio=routing.activated_ratio,
        high_res_encoded=n_encoded,
        high_res_total=high_res_total,
        loss_route=routing.loss_route,
        loss_hinge=routing.loss_hinge,
    )
