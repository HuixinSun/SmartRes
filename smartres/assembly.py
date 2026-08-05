"""Order-preserving visual token assembly: interleave the two resolutions in image order."""

from dataclasses import dataclass

import torch


@dataclass
class AssembledSequence:
    """One sample's visual sequence, before the patch merger."""

    tokens: torch.Tensor        # [L, D] interleaved low- and high-resolution features
    is_high_res: torch.Tensor   # [L] bool
    source_high_res: torch.Tensor  # [L] native high-res cell index, -1 for low-res tokens
    source_low_res: torch.Tensor   # [L] native low-res patch index, -1 for high-res tokens


def order_preserving_assembly(
    low_res_features: torch.Tensor,
    high_res_features: torch.Tensor,
    route_mask: torch.Tensor,
    owner: torch.Tensor,
    merge_unit: int,
) -> AssembledSequence:
    """Interleave the two resolutions into one sequence."""
    device = low_res_features.device
    n_low, dim = low_res_features.shape

    owned = owner >= 0
    owners_of_cell = owner[owned]                                   # [K]
    cells = torch.nonzero(owned, as_tuple=False).flatten()          # [K], ascending

    # How many tokens each low-res patch contributes: its owned cells if activated,
    # otherwise exactly one.
    high_res_counts = torch.bincount(owners_of_cell, minlength=n_low)
    counts = torch.where(route_mask, high_res_counts, torch.ones_like(high_res_counts))
    offsets = torch.cumsum(counts, 0) - counts                      # exclusive scan
    total = int(counts.sum())

    tokens = low_res_features.new_zeros((total, dim))
    is_high_res = torch.zeros(total, dtype=torch.bool, device=device)
    source_high_res = torch.full((total,), -1, dtype=torch.long, device=device)
    source_low_res = torch.full((total,), -1, dtype=torch.long, device=device)

    # Non-activated patches: one token each, at their own offset.
    low_res_slots = offsets[~route_mask]
    tokens[low_res_slots] = low_res_features[~route_mask]
    source_low_res[low_res_slots] = torch.nonzero(~route_mask, as_tuple=False).flatten()

    if cells.numel():
        # A stable sort by owner keeps the cell indices ascending inside each group, which
        # is what makes the expansion follow the image order rather than an arbitrary one.
        order = torch.argsort(owners_of_cell, stable=True)
        cells_sorted, owners_sorted = cells[order], owners_of_cell[order]
        group_start = torch.cumsum(high_res_counts, 0) - high_res_counts
        rank = torch.arange(cells_sorted.numel(), device=device) - group_start[owners_sorted]
        slots = offsets[owners_sorted] + rank
        tokens[slots] = high_res_features[cells_sorted]
        is_high_res[slots] = True
        source_high_res[slots] = cells_sorted

    remainder = total % merge_unit
    if remainder:
        pad = merge_unit - remainder
        tokens = torch.cat([tokens, tokens.new_zeros((pad, dim))], dim=0)
        is_high_res = torch.cat([is_high_res, is_high_res.new_zeros(pad)], dim=0)
        source_high_res = torch.cat([source_high_res, source_high_res.new_full((pad,), -1)], dim=0)
        source_low_res = torch.cat([source_low_res, source_low_res.new_full((pad,), -1)], dim=0)

    return AssembledSequence(
        tokens=tokens,
        is_high_res=is_high_res,
        source_high_res=source_high_res,
        source_low_res=source_low_res,
    )
