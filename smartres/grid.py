"""Index maps between the low- and high-resolution patch grids, plus cell ownership."""

from typing import Tuple

import torch


def native_to_raster_maps(grid_thw: torch.Tensor, merge: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutations between native and patch-raster order, for a whole batch."""
    n2r, r2n, offset = [], [], 0
    for row in grid_thw.tolist():
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        n = t * h * w
        # Entry at [t, unit_y, unit_x, sub_y, sub_x] is the native index of patch (t,y,x).
        flat = (
            torch.arange(n, device=device)
            .view(t, h // merge, w // merge, merge, merge)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1)
        )
        n2r.append(torch.argsort(flat) + offset)
        r2n.append(flat + offset)
        offset += n
    return torch.cat(n2r), torch.cat(r2n)


def native_cells(t: int, h: int, w: int, merge: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """native flat index -> (frame, y, x)."""
    idx = torch.arange(t * h * w, device=device, dtype=torch.long)
    frame = idx // (h * w)
    rem = idx % (h * w)
    unit, sub = rem // (merge * merge), rem % (merge * merge)
    unit_y, unit_x = unit // (w // merge), unit % (w // merge)
    sub_y, sub_x = sub // merge, sub % merge
    return frame, unit_y * merge + sub_y, unit_x * merge + sub_x


def native_cell_map(t: int, h: int, w: int, merge: int, device) -> torch.Tensor:
    """[t, h, w] holding the native flat index of each patch cell (the inverse map)."""
    return (
        torch.arange(t * h * w, device=device, dtype=torch.long)
        .view(t, h // merge, w // merge, merge, merge)
        .permute(0, 1, 3, 2, 4)
        .reshape(t, h, w)
    )


def window_to_native(feats: torch.Tensor, window_index: torch.Tensor, merge_unit: int) -> torch.Tensor:
    """Undo the vision tower's windowed permutation, returning features in native order."""
    inverse = torch.argsort(window_index)
    return feats.reshape(-1, merge_unit, feats.shape[-1])[inverse].reshape(-1, feats.shape[-1])


def _axis_coverage(n_lo: int, n_hi: int, device) -> torch.Tensor:
    """[n_lo, n_hi] bool: which high-res indices each low-res index covers, on one axis."""
    i = torch.arange(n_lo, device=device)
    lo = (i * n_hi) // n_lo
    hi = ((i + 1) * n_hi) // n_lo + ((((i + 1) * n_hi) % n_lo) > 0).long()
    hi = torch.clamp(hi, max=n_hi)
    j = torch.arange(n_hi, device=device)
    return (j[None, :] >= lo[:, None]) & (j[None, :] < hi[:, None])


def _covering_table(n_lo: int, n_hi: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`_axis_coverage`: for each high-res index, its covering low-res ones."""
    cover = _axis_coverage(n_lo, n_hi, device).t()                 # [n_hi, n_lo]
    k = int(cover.sum(1).max().clamp(min=1))
    idx = torch.arange(n_lo, device=device).expand(n_hi, n_lo)
    padded = torch.where(cover, idx, torch.full_like(idx, n_lo))
    cov = padded.sort(dim=1).values[:, :k]
    return cov, cov < n_lo


def high_res_owner(
    route_mask: torch.Tensor,
    t: int, h: int, w: int,
    T: int, H: int, W: int,
    merge: int, merge_unit: int, device,
) -> torch.Tensor:
    """[T*H*W] the low-res patch that owns each high-res cell, ``-1`` if none."""
    n_lo, n_hi = t * h * w, T * H * W
    cov_y, ok_y = _covering_table(h, H, device)
    cov_x, ok_x = _covering_table(w, W, device)

    def native_index(frame: int, y, x):
        return (
            frame * (h * w)
            + ((y // merge) * (w // merge) + (x // merge)) * merge_unit
            + (y % merge) * merge + (x % merge)
        )

    owner = torch.full((T, H, W), n_lo, dtype=torch.long, device=device)
    for frame in range(t):
        hr_frame = (frame * T) // t if t else frame
        if hr_frame >= T:
            continue
        best = torch.full((H, W), n_lo, dtype=torch.long, device=device)
        for a in range(cov_y.shape[1]):
            y = cov_y[:, a].view(H, 1).expand(H, W)
            y_ok = ok_y[:, a].view(H, 1).expand(H, W)
            for b in range(cov_x.shape[1]):
                x = cov_x[:, b].view(1, W).expand(H, W)
                x_ok = ok_x[:, b].view(1, W).expand(H, W)
                candidate = native_index(frame, y.clamp(max=h - 1), x.clamp(max=w - 1))
                activated = route_mask[candidate.reshape(-1)].view(H, W)
                best = torch.where(y_ok & x_ok & activated, torch.minimum(best, candidate), best)
        owner[hr_frame] = torch.minimum(owner[hr_frame], best)

    # Re-index from (t, y, x) to native high-res order, matching the feature buffer.
    cells = native_cell_map(T, H, W, merge, device).reshape(-1)
    flat = owner.reshape(-1)
    out = torch.full((n_hi,), -1, dtype=torch.long, device=device)
    has_owner = flat < n_lo
    out[cells[has_owner]] = flat[has_owner]
    return out
