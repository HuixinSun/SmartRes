"""Attention-window and rotary context for a fully or partially encoded frame."""

from typing import Dict, Tuple

import torch


def unit_window_id(vision_tower, grid_thw: torch.Tensor, merge_unit: int, device) -> Tuple[torch.Tensor, int]:
    """[n_units] attention-window id per merge unit, in NATIVE order, plus the count."""
    window_index, cu_window_seqlens = vision_tower.get_window_index(grid_thw)
    window_index = window_index.to(device)
    bounds = torch.unique_consecutive(
        torch.tensor(cu_window_seqlens, device=device, dtype=torch.long)
    ) // merge_unit
    n_units = int(window_index.numel())
    slots = torch.arange(n_units, device=device)
    id_by_slot = torch.searchsorted(bounds[1:].contiguous(), slots, right=True)
    window_id = torch.empty(n_units, dtype=torch.long, device=device)
    window_id[window_index] = id_by_slot
    return window_id, int(bounds.numel() - 1)


def snap_to_windows(
    vision_tower, keep_units: torch.Tensor, grid_thw: torch.Tensor, merge_unit: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Grow a kept-unit mask to whole attention windows."""
    window_id, n_windows = unit_window_id(vision_tower, grid_thw, merge_unit, device)
    touched = torch.zeros(n_windows, dtype=torch.bool, device=device)
    touched[window_id[keep_units]] = True
    keep_units = keep_units | touched[window_id]
    return keep_units, expand_units_to_patches(keep_units, merge_unit, device)


def expand_units_to_patches(keep_units: torch.Tensor, merge_unit: int, device) -> torch.Tensor:
    """Kept merge units -> the patch indices they contain, ascending."""
    units = torch.nonzero(keep_units, as_tuple=False).flatten()
    return (
        units[:, None] * merge_unit + torch.arange(merge_unit, device=device)[None, :]
    ).reshape(-1)


def full_window_context(
    vision_tower, patch_embeddings: torch.Tensor, grid_thw: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """The vision tower's own windowing context, for a frame encoded in full."""
    device = patch_embeddings.device
    merge_unit = vision_tower.spatial_merge_unit
    rotary = vision_tower.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = vision_tower.get_window_index(grid_thw)
    window_index = window_index.to(device)
    cu_window_seqlens = torch.unique_consecutive(
        torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
    )

    seq_len = patch_embeddings.size(0)
    n_units = seq_len // merge_unit
    hidden = patch_embeddings.reshape(n_units, merge_unit, -1)[window_index].reshape(seq_len, -1)

    rot = rotary.reshape(n_units, merge_unit, -1)[window_index].reshape(seq_len, -1)
    emb = torch.cat((rot, rot), dim=-1)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    return {
        "hidden_states": hidden,
        "position_embeddings": (emb.cos(), emb.sin()),
        "cu_window_seqlens": cu_window_seqlens,
        "cu_seqlens": cu_seqlens,
        "window_index": window_index,
    }


def run_blocks(
    vision_tower, hidden_states: torch.Tensor, context: Dict[str, torch.Tensor],
    start: int, end: int,
) -> torch.Tensor:
    """Run vision blocks ``[start, end)``, choosing the right attention extent per block."""
    # Honour the tower's own checkpointing flag: without this the config's
    # gradient_checkpointing reaches the language model but not these 32 blocks.
    checkpoint = getattr(vision_tower, "gradient_checkpointing", False) and vision_tower.training
    for layer in range(start, min(end, len(vision_tower.blocks))):
        block = vision_tower.blocks[layer]
        cu = (
            context["cu_seqlens"]
            if layer in vision_tower.fullatt_block_indexes
            else context["cu_window_seqlens"]
        )
        if checkpoint:
            hidden_states = vision_tower._gradient_checkpointing_func(
                block.__call__, hidden_states, cu, None, context["position_embeddings"]
            )
        else:
            hidden_states = block(
                hidden_states, cu_seqlens=cu, position_embeddings=context["position_embeddings"]
            )
    return hidden_states


def selective_window_context(
    vision_tower,
    patch_embeddings: torch.Tensor,
    grid_thw: torch.Tensor,
    keep_units: torch.Tensor,
    merge_unit: int,
) -> Dict[str, torch.Tensor]:
    """Windowing and rotary context restricted to the kept merge units."""
    device = patch_embeddings.device
    rotary = vision_tower.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = vision_tower.get_window_index(grid_thw)
    window_index = window_index.to(device)
    cu_window_seqlens = torch.unique_consecutive(
        torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
    )
    n_units = int(window_index.numel())

    kept_windowed = keep_units[window_index]                       # survivor per windowed slot
    kept_slots = torch.nonzero(kept_windowed, as_tuple=False).flatten()

    # Rotary: windowed order, then subset. Each survivor keeps its own coordinates.
    rot = rotary.reshape(n_units, merge_unit, -1)[window_index][kept_slots]
    rot = rot.reshape(-1, rot.shape[-1])
    emb = torch.cat((rot, rot), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # Windowed blocks: count survivors inside each original window span.
    unit_bounds = (cu_window_seqlens // merge_unit).to(torch.long)
    per_window = [
        int(kept_windowed[int(unit_bounds[i]):int(unit_bounds[i + 1])].sum())
        for i in range(unit_bounds.numel() - 1)
    ]
    new_cu_window = torch.tensor([0] + per_window, device=device, dtype=torch.int32)
    new_cu_window = torch.unique_consecutive((new_cu_window.cumsum(0) * merge_unit).to(torch.int32))

    # Full-attention blocks: survivors per frame. Units are laid out frame-major.
    units_per_frame = []
    for row in grid_thw.tolist():
        frames, height, width = int(row[0]), int(row[1]), int(row[2])
        units_per_frame += [height * width // merge_unit] * frames
    frame_bounds = torch.tensor([0] + units_per_frame, device=device).cumsum(0)
    per_frame = [
        int(keep_units[int(frame_bounds[i]):int(frame_bounds[i + 1])].sum()) * merge_unit
        for i in range(frame_bounds.numel() - 1)
    ]
    cu_seqlens = torch.tensor([0] + per_frame, device=device, dtype=torch.int32).cumsum(0)
    cu_seqlens = torch.unique_consecutive(cu_seqlens.to(torch.int32))

    # Hidden states: the caller supplied them in native order; the blocks want windowed.
    kept_native_units = torch.nonzero(keep_units, as_tuple=False).flatten()
    rank = torch.full((n_units,), -1, dtype=torch.long, device=device)
    rank[kept_native_units] = torch.arange(kept_native_units.numel(), device=device)
    gather = rank[window_index[kept_slots]]
    hidden = patch_embeddings.reshape(-1, merge_unit, patch_embeddings.shape[-1])[gather]

    return {
        "hidden_states": hidden.reshape(-1, patch_embeddings.shape[-1]),
        "position_embeddings": position_embeddings,
        "cu_window_seqlens": new_cu_window,
        "cu_seqlens": cu_seqlens,
        "kept_slots": kept_slots,
        "window_index": window_index,
    }
