"""Make a stock Qwen2.5-VL accept SmartRes' variable-length visual sequence.

The stock model scatters visual features into the placeholder run, which requires the two
to have the same length. SmartRes decides that length at run time, so the run is replaced
wholesale instead and everything indexed by position is rebuilt around it.
"""

from types import MethodType
from typing import List, Optional, Tuple

import torch

from .sequence import realign_labels, splice_visual_sequence


def _vision_span_positions(
    length: int, grid_t: int, grid_h: int, grid_w: int, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The t/h/w channels for a vision span of ``length`` tokens over a (t, h, w) grid.

    The raster walk is truncated to the span, so a token's h/w come from its position in
    the sequence rather than from where it sits in the frame. That is what the released
    checkpoint was trained under; keep it unless the checkpoint changes.
    """
    t_index = (
        torch.arange(grid_t, device=device, dtype=torch.long)
        .view(-1, 1, 1).expand(-1, grid_h, grid_w).flatten()
    )
    h_index = (
        torch.arange(grid_h, device=device, dtype=torch.long)
        .view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    )
    w_index = (
        torch.arange(grid_w, device=device, dtype=torch.long)
        .view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    )

    flat_len = int(t_index.numel())
    if length < flat_len:
        return t_index[:length], h_index[:length], w_index[:length]
    if length > flat_len and flat_len > 0:
        repeats = (length + flat_len - 1) // flat_len
        return (
            t_index.repeat(repeats)[:length],
            h_index.repeat(repeats)[:length],
            w_index.repeat(repeats)[:length],
        )
    return t_index, h_index, w_index


def spliced_rope_index(
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    image_token_id: int,
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D M-RoPE positions for a batch whose vision spans have already been spliced in.

    Images only: the grounding datasets this release ships carry one image per sample and
    no video, and a video span would need its own temporal handling.
    """
    device = input_ids.device
    batch_size, padded_len = input_ids.shape
    position_ids = torch.ones(3, batch_size, padded_len, dtype=torch.long, device=device)
    rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

    for b in range(batch_size):
        ids = input_ids[b]
        keep = attention_mask[b] == 1 if attention_mask is not None else torch.ones_like(ids, dtype=torch.bool)
        ids = ids[keep]

        positions = torch.nonzero(ids == image_token_id, as_tuple=False).flatten()
        chunks: List[torch.Tensor] = []
        cursor = 0  # read head into `ids`
        offset = 0  # next free position index

        if positions.numel():
            start, end = int(positions[0]), int(positions[-1])
            span_len = end - start + 1

            if start > cursor:
                text = torch.arange(start - cursor, device=device, dtype=torch.long) + offset
                chunks.append(text.expand(3, -1))
                offset += start - cursor

            grid = grid_thw[b] if grid_thw.dim() == 2 else grid_thw
            grid_t = int(grid[0])
            grid_h = int(grid[1]) // spatial_merge_size
            grid_w = int(grid[2]) // spatial_merge_size
            t_index, h_index, w_index = _vision_span_positions(
                span_len, grid_t, grid_h, grid_w, device
            )
            chunks.append(torch.stack([t_index, h_index, w_index]) + offset)
            offset += int(max(t_index.max(), h_index.max(), w_index.max())) + 1
            cursor = end + 1

        if cursor < ids.numel():
            text = torch.arange(ids.numel() - cursor, device=device, dtype=torch.long) + offset
            chunks.append(text.expand(3, -1))
            offset += ids.numel() - cursor

        flat = torch.cat(chunks, dim=1) if chunks else torch.zeros(3, 0, dtype=torch.long, device=device)
        position_ids[:, b, keep] = flat
        rope_deltas[b] = int(flat.max()) + 1 - int(ids.numel()) if flat.numel() else 0

    return position_ids, rope_deltas


def install_runtime(model, image_token_id: Optional[int] = None):
    """Route the language model around the stock scatter. Returns ``model``.

    Call after :func:`install_smartres`; both are idempotent per model instance.
    """
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "visual"):
        raise AttributeError(
            "install_runtime expects a Qwen2_5_VLForConditionalGeneration whose .model "
            "holds the vision tower"
        )
    if getattr(inner, "_smartres_runtime", False):
        return model

    if image_token_id is None:
        image_token_id = model.config.image_token_id
    merge = inner.visual.spatial_merge_size
    stock_forward = inner.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, pixel_values=None, image_grid_thw=None,
                pixel_frames_hr=None, hr_grid_thw=None, labels=None,
                cache_position=None, **kwargs):
        routed = (
            inputs_embeds is None
            and input_ids is not None
            and pixel_values is not None
            and pixel_frames_hr is not None
            and hr_grid_thw is not None
        )
        if not routed:
            return stock_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                cache_position=cache_position, **kwargs
            )

        visual_embeds, assembled = self.visual(
            pixel_values, grid_thw=image_grid_thw,
            pixel_frames_hr=pixel_frames_hr, hr_grid_thw=hr_grid_thw,
            text_prompt=kwargs.pop("text_prompt", None),
        )
        embeds = self.get_input_embeddings()(input_ids)
        spliced = splice_visual_sequence(
            input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask,
            visual_embeds=visual_embeds.to(embeds.dtype), assembled_lengths=list(assembled),
            image_token_id=image_token_id, spatial_merge_size=merge,
        )
        # Positions come from the high-resolution grid: the spliced span is a walk over it.
        position_ids, rope_deltas = spliced_rope_index(
            spliced.input_ids, hr_grid_thw, spliced.attention_mask, image_token_id, merge
        )
        self.rope_deltas = rope_deltas
        self._smartres_spliced = spliced

        if labels is not None:
            labels = realign_labels(
                labels, spliced.spans, spliced.lengths, spliced.input_ids.size(1)
            )
        return stock_forward(
            inputs_embeds=spliced.inputs_embeds,
            attention_mask=spliced.attention_mask,
            position_ids=position_ids,
            cache_position=spliced.cache_position,
            labels=labels,
            **kwargs,
        )

    inner.forward = MethodType(forward, inner)
    inner._smartres_runtime = True
    _install_generation_bridge(model)
    return model


def _install_generation_bridge(model):
    """Prefill through the splice, then let the stock decode loop run on embeddings.

    ``generate`` tracks its own ids, which no longer line up with the spliced prefill, so
    the visual pass happens here and the model sees ``inputs_embeds``.
    """
    if getattr(model, "_smartres_generate", False):
        return
    stock_generate = model.generate

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 image_grid_thw=None, pixel_frames_hr=None, hr_grid_thw=None, **kwargs):
        if input_ids is None or pixel_frames_hr is None or hr_grid_thw is None:
            return stock_generate(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs
            )

        inner = self.model
        visual_embeds, assembled = inner.visual(
            pixel_values, grid_thw=image_grid_thw,
            pixel_frames_hr=pixel_frames_hr, hr_grid_thw=hr_grid_thw,
        )
        embeds = inner.get_input_embeddings()(input_ids)
        spliced = splice_visual_sequence(
            input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask,
            visual_embeds=visual_embeds.to(embeds.dtype), assembled_lengths=list(assembled),
            image_token_id=self.config.image_token_id,
            spatial_merge_size=inner.visual.spatial_merge_size,
        )
        position_ids, rope_deltas = spliced_rope_index(
            spliced.input_ids, hr_grid_thw, spliced.attention_mask,
            self.config.image_token_id, inner.visual.spatial_merge_size,
        )
        inner.rope_deltas = rope_deltas
        return stock_generate(
            inputs_embeds=spliced.inputs_embeds,
            attention_mask=spliced.attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    model.generate = MethodType(generate, model)
    model._smartres_generate = True
