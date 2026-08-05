"""FastV token pruning via hooks. Ranks image tokens at layer k by layer k-1 attention."""

from typing import List, Optional

import torch


def _last_token_attention_to_images(
    attentions: torch.Tensor, image_start: int, image_end: int
) -> torch.Tensor:
    """Mean-over-heads attention from the last query position to the image span."""
    return attentions.mean(dim=1)[0, -1, image_start:image_end]


class _FastVState:
    """Bookkeeping shared between the two hooks."""

    def __init__(self, layer: int, keep_ratio: float) -> None:
        self.layer = layer
        self.keep_ratio = keep_ratio
        self.attentions: Optional[torch.Tensor] = None
        self.image_span: Optional[tuple] = None
        self.handles: List = []
        self.kept: Optional[int] = None
        self.total: Optional[int] = None


def _capture_attention(state: _FastVState):
    def hook(module, args, kwargs, output):
        # The layer returns (hidden_states, attn_weights, ...) when asked for attentions.
        if isinstance(output, tuple) and len(output) > 1 and torch.is_tensor(output[1]):
            state.attentions = output[1]
        return output

    return hook


def _prune(state: _FastVState):
    def hook(module, args, kwargs):
        # Decode steps carry a single query position; nothing to prune.
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.shape[1] <= 1:
            return None
        if state.attentions is None or state.image_span is None:
            return None

        start, end = state.image_span
        if end > hidden.shape[1]:
            return None

        scores = _last_token_attention_to_images(state.attentions, start, end)
        n_image = end - start
        n_keep = max(1, int(n_image * state.keep_ratio))
        keep_in_span = scores.topk(n_keep).indices.sort().values + start

        device = hidden.device
        keep = torch.cat([
            torch.arange(start, device=device),
            keep_in_span,
            torch.arange(end, hidden.shape[1], device=device),
        ])

        state.kept, state.total = n_keep, n_image
        state.image_span = (start, start + n_keep)   # the span the next layers will see

        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden[:, keep]
        for name in ("position_ids", "cache_position"):
            value = kwargs.get(name)
            if torch.is_tensor(value):
                kwargs[name] = value[..., keep] if value.dim() > 1 else value[keep]
        embeddings = kwargs.get("position_embeddings")
        if isinstance(embeddings, tuple) and torch.is_tensor(embeddings[0]):
            kwargs["position_embeddings"] = tuple(e[..., keep, :] for e in embeddings)
        mask = kwargs.get("attention_mask")
        if torch.is_tensor(mask) and mask.dim() == 4:
            kwargs["attention_mask"] = mask[:, :, keep][..., keep]
        return (), kwargs

    return hook


def install_fastv(model, k: int = 2, keep_ratio: float = 0.5, image_token_id: int = 151655):
    """Prune image tokens at decoder layer ``k``, keeping ``keep_ratio`` of them."""
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    if not 1 <= k < len(layers):
        raise ValueError(f"k must be in [1, {len(layers)}), got {k}")

    state = _FastVState(k, keep_ratio)

    # Locate the image span from input_ids before the decoder runs.
    def find_span(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if torch.is_tensor(ids):
            positions = torch.nonzero(ids[0] == image_token_id, as_tuple=False).flatten()
            state.image_span = (
                (int(positions[0]), int(positions[-1]) + 1) if positions.numel() else None
            )
        state.attentions = None
        return None

    state.handles.append(model.register_forward_pre_hook(find_span, with_kwargs=True))
    state.handles.append(
        layers[k - 1].register_forward_hook(_capture_attention(state), with_kwargs=True)
    )
    state.handles.append(layers[k].register_forward_pre_hook(_prune(state), with_kwargs=True))
    return state


def remove_fastv(state: _FastVState) -> None:
    """Undo :func:`install_fastv`."""
    for handle in state.handles:
        handle.remove()
    state.handles.clear()
