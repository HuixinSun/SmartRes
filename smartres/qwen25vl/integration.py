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
        rope_deltas[b] = int(flat.max()) + 1 - padded_len if flat.numel() else 0

    return position_ids, rope_deltas


def _prepare(inner, input_ids, attention_mask, pixel_values, image_grid_thw,
             high_res_pixels, high_res_grid, text_prompt, image_token_id, merge):
    """Run the vision tower and rebuild the batch around its variable-length output."""
    visual_embeds, assembled = inner.visual(
        pixel_values, grid_thw=image_grid_thw,
        pixel_frames_hr=high_res_pixels, hr_grid_thw=high_res_grid,
        text_prompt=text_prompt,
    )
    embeds = inner.get_input_embeddings()(input_ids)
    spliced = splice_visual_sequence(
        input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask,
        visual_embeds=visual_embeds.to(embeds.dtype), assembled_lengths=list(assembled),
        image_token_id=image_token_id, spatial_merge_size=merge,
    )
    # Positions come from the high-resolution grid: the spliced span is a walk over it.
    position_ids, rope_deltas = spliced_rope_index(
        spliced.input_ids, high_res_grid, spliced.attention_mask, image_token_id, merge
    )
    inner.rope_deltas = rope_deltas
    return spliced, position_ids


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
    if getattr(model, "_smartres_runtime", False):
        return model

    if image_token_id is None:
        image_token_id = model.config.image_token_id
    merge = inner.visual.spatial_merge_size
    stock_inner = inner.forward
    stock_outer = model.forward

    def inner_forward(self, *args, position_ids=None, inputs_embeds=None, **kwargs):
        """Qwen2.5-VL drops position_ids on every generation step; take back ours."""
        if inputs_embeds is not None and position_ids is None:
            stashed = getattr(self, "_smartres_prefill_positions", None)
            if stashed is not None and stashed.shape[-1] == inputs_embeds.size(1):
                position_ids = stashed
                self._smartres_prefill_positions = None
        return stock_inner(*args, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs)

    def outer_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                      inputs_embeds=None, pixel_values=None, image_grid_thw=None,
                      pixel_frames_hr=None, hr_grid_thw=None, labels=None,
                      cache_position=None, text_prompt=None, instruction=None, **kwargs):
        """The splice lives here because this is where the loss is built from labels."""
        routed = (
            inputs_embeds is None and input_ids is not None and pixel_values is not None
            and pixel_frames_hr is not None and hr_grid_thw is not None
        )
        if routed:
            spliced, positions = _prepare(
                inner, input_ids, attention_mask, pixel_values, image_grid_thw,
                pixel_frames_hr, hr_grid_thw, text_prompt, image_token_id, merge,
            )
            if labels is not None:
                labels = realign_labels(
                    labels, spliced.spans, spliced.lengths, spliced.input_ids.size(1)
                )
            inner._smartres_prefill_positions = positions
            output = stock_outer(
                inputs_embeds=spliced.inputs_embeds,
                attention_mask=spliced.attention_mask,
                cache_position=spliced.cache_position,
                labels=labels, **kwargs,
            )
        else:
            output = stock_outer(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                inputs_embeds=inputs_embeds, pixel_values=pixel_values,
                image_grid_thw=image_grid_thw, labels=labels,
                cache_position=cache_position, **kwargs,
            )

        # The routing terms are produced by the tower and belong in the trained loss.
        routing = getattr(inner.visual, "loss_mts", None)
        if getattr(output, "loss", None) is not None and torch.is_tensor(routing):
            output.loss = output.loss + routing.to(output.loss.device)
        inner.visual.loss_mts = None
        return output

    inner.forward = MethodType(inner_forward, inner)
    model.forward = MethodType(outer_forward, model)
    model._smartres_runtime = True
    _install_generation_bridge(model, inner, image_token_id, merge)
    return model


def _install_generation_bridge(model, inner, image_token_id, merge):
    """Prefill through the splice, then let the stock decode loop run on embeddings.

    ``generate`` tracks its own ids, which no longer line up with the spliced prefill, so
    the visual pass happens here and the model sees ``inputs_embeds``.
    """
    stock_generate = model.generate

    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 image_grid_thw=None, pixel_frames_hr=None, hr_grid_thw=None,
                 text_prompt=None, instruction=None, **kwargs):
        if input_ids is None or pixel_frames_hr is None or hr_grid_thw is None:
            return stock_generate(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs
            )

        spliced, positions = _prepare(
            inner, input_ids, attention_mask, pixel_values, image_grid_thw,
            pixel_frames_hr, hr_grid_thw, text_prompt, image_token_id, merge,
        )
        inner._smartres_prefill_positions = positions
        out = stock_generate(
            inputs_embeds=spliced.inputs_embeds,
            attention_mask=spliced.attention_mask,
            **kwargs,
        )
        # Generating from embeddings returns the completion alone; callers that strip a
        # prompt off the front expect the prompt to still be there.
        if isinstance(out, torch.Tensor):
            return torch.cat([input_ids, out.to(input_ids.device)], dim=1)
        return out

    model.generate = MethodType(generate, model)
