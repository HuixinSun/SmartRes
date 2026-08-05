"""Splice a variable-length visual sequence into a prompt and rebuild what indexes by position."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

# <|endoftext|>. Padding is masked out, so the exact id only matters for tools that
# decode the padded ids for inspection.
_PAD_TOKEN_ID = 151643


@dataclass
class SplicedBatch:
    """The rebuilt batch, plus what callers need to realign anything position-indexed."""

    inputs_embeds: torch.Tensor          # [B, L, H]
    input_ids: torch.Tensor              # [B, L]   placeholders repeated to the true count
    attention_mask: torch.Tensor         # [B, L]
    cache_position: torch.Tensor         # [L]
    lengths: torch.Tensor                # [B]      unpadded length of each sample
    # (start, end, n_visual, n_placeholder) per sample in UNPADDED coordinates, or None
    # for a sample with no image. Left padding shifts these by (L - lengths[b]).
    spans: List[Optional[Tuple[int, int, int, int]]] = field(default_factory=list)


def _visual_token_count(assembled_length: int, spatial_merge_size: int) -> int:
    """Assembled ViT tokens -> language-model tokens."""
    return assembled_length // (spatial_merge_size ** 2)


def _left_pad(tensors: List[torch.Tensor], length: int, pad_value: float) -> torch.Tensor:
    out = []
    for t in tensors:
        deficit = length - t.size(0)
        if deficit > 0:
            shape = (deficit,) + tuple(t.shape[1:])
            pad = torch.full(shape, pad_value, device=t.device, dtype=t.dtype)
            t = torch.cat([pad, t], dim=0)
        out.append(t)
    return torch.stack(out, dim=0)


def splice_visual_sequence(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    visual_embeds: torch.Tensor,
    assembled_lengths: List[int],
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> SplicedBatch:
    """Replace each prompt's placeholder run with its assembled visual tokens."""
    batch_size = inputs_embeds.size(0)
    if attention_mask is None:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )

    embeds_out, ids_out, mask_out, spans = [], [], [], []
    cursor = 0  # read head into visual_embeds

    for b in range(batch_size):
        ids_b, emb_b, mask_b = input_ids[b], inputs_embeds[b], attention_mask[b]
        positions = torch.nonzero(ids_b == image_token_id, as_tuple=False).flatten()

        if positions.numel() == 0:
            embeds_out.append(emb_b)
            ids_out.append(ids_b)
            mask_out.append(mask_b)
            spans.append(None)
            continue

        n_visual = _visual_token_count(assembled_lengths[b], spatial_merge_size)
        span = visual_embeds[cursor:cursor + n_visual]
        cursor += n_visual
        if span.size(0) != n_visual:
            raise ValueError(
                f"sample {b}: assembly announced {n_visual} visual tokens but only "
                f"{span.size(0)} were available -- assembled_lengths does not match "
                f"visual_embeds"
            )

        # The placeholders are one contiguous run; replace it wholesale.
        start, end = int(positions[0]), int(positions[-1])
        n_placeholder = int(positions.numel())

        embeds_out.append(torch.cat([emb_b[:start], span, emb_b[end + 1:]], dim=0))
        ids_out.append(torch.cat([
            ids_b[:start],
            torch.full((n_visual,), image_token_id, device=ids_b.device, dtype=ids_b.dtype),
            ids_b[end + 1:],
        ], dim=0))
        mask_out.append(torch.cat([
            mask_b[:start],
            torch.ones(n_visual, device=mask_b.device, dtype=mask_b.dtype),
            mask_b[end + 1:],
        ], dim=0))
        spans.append((start, end, n_visual, n_placeholder))

    lengths = torch.tensor([e.size(0) for e in embeds_out], device=inputs_embeds.device)
    max_len = int(lengths.max())

    return SplicedBatch(
        inputs_embeds=_left_pad(embeds_out, max_len, 0.0),
        input_ids=_left_pad(ids_out, max_len, _PAD_TOKEN_ID),
        attention_mask=_left_pad(mask_out, max_len, 0),
        cache_position=torch.arange(max_len, device=inputs_embeds.device, dtype=torch.long),
        lengths=lengths,
        spans=spans,
    )


def realign_labels(
    labels: torch.Tensor,
    spans: List[Optional[Tuple[int, int, int, int]]],
    lengths: torch.Tensor,
    max_len: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Rebuild ``labels`` for a spliced batch."""
    out = []
    for b, span in enumerate(spans):
        row = labels[b]
        if span is None:
            new = row
        else:
            start, end, n_visual, _ = span
            new = torch.cat([
                row[:start],
                torch.full((n_visual,), ignore_index, device=row.device, dtype=row.dtype),
                row[end + 1:],
            ], dim=0)
        deficit = max_len - new.size(0)
        if deficit > 0:
            pad = torch.full((deficit,), ignore_index, device=new.device, dtype=new.dtype)
            new = torch.cat([pad, new], dim=0)
        out.append(new)
    return torch.stack(out, dim=0)
