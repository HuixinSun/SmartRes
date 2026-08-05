"""Lightweight router. Scores low-resolution patches, thresholds into M = STE(S > tau)."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingOutput:
    """Losses are None outside training."""

    mask: torch.Tensor                      # [N] in {0, 1}, straight-through
    score: torch.Tensor                     # [N] raw logits, pre-sigmoid
    loss_route: Optional[torch.Tensor] = None
    loss_hinge: Optional[torch.Tensor] = None

    @property
    def activated_ratio(self) -> float:
        """Fraction of low-resolution patches routed to high resolution."""
        return float(self.mask.detach().float().mean())


def straight_through_threshold(
    values: torch.Tensor, tau: float, temperature: float = 1.0, hard: bool = True
) -> torch.Tensor:
    """``values > tau`` in the forward pass, sigmoid gradient in the backward pass."""
    soft = torch.sigmoid((values - tau) / temperature)
    if not hard:
        return soft
    return (values >= tau).to(soft.dtype) + (soft - soft.detach())


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
    if num_layers == 1:
        return nn.Linear(input_dim, output_dim)
    dims = [input_dim] + [hidden_dim] * (num_layers - 1)
    layers: list = []
    for i, (n_in, n_out) in enumerate(zip(dims, dims[1:] + [output_dim])):
        layers.append(nn.Linear(n_in, n_out))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Router(nn.Module):
    """Per-patch resolution router."""

    def __init__(
        self,
        embed_dim: int,
        tau: float = 0.5,
        temperature: float = 1.0,
        margin: float = 1.0,
        pos_weight: float = 5.0,
        lambda_route: float = 1.0,
        lambda_hinge: float = 5.0,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.temperature = temperature
        self.margin = margin
        self.pos_weight = pos_weight
        self.lambda_route = lambda_route
        self.lambda_hinge = lambda_hinge

        # Sole parameters of the router: [D -> D -> D -> 1].
        self.score_head = _mlp(embed_dim, embed_dim, 1, num_layers=num_layers)

        # Start neutral at sigmoid(0) = 0.5, so the feature statistics set the initial
        # activation rate rather than a biased head.
        last = self.score_head[-1] if isinstance(self.score_head, nn.Sequential) else self.score_head
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(
        self, features: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> RoutingOutput:
        """Score every low-resolution patch and threshold it."""
        score = self.score_head(features).squeeze(-1)
        # tau lives in probability space, so squash first. The losses stay on the logits.
        mask = straight_through_threshold(
            score.sigmoid(), self.tau, temperature=self.temperature, hard=True
        )
        if target is None:
            return RoutingOutput(mask=mask, score=score)

        target = target.to(score.dtype)
        loss_route, loss_hinge = self._objective(score, target)
        return RoutingOutput(
            mask=mask, score=score, loss_route=loss_route, loss_hinge=loss_hinge
        )

    def _objective(self, score: torch.Tensor, target: torch.Tensor):
        """Weighted BCE plus the margin regulariser."""
        foreground = target > 0.5
        background = ~foreground

        # One-class samples carry no separation signal. Return a differentiable zero
        # rather than skipping, so the graph stays intact under DDP.
        if not bool(foreground.any()) or not bool(background.any()):
            zero = score.sum() * 0.0
            return zero, zero

        bce = F.binary_cross_entropy_with_logits(score, target, reduction="none")
        weight = torch.where(
            foreground, torch.full_like(bce, self.pos_weight), torch.ones_like(bce)
        )
        loss_route = (bce * weight).mean()

        gap = score[foreground].mean() - score[background].mean()
        loss_hinge = F.relu(self.margin - gap)
        return loss_route, loss_hinge

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, margin={self.margin}, "
            f"pos_weight={self.pos_weight}, lambda_hinge={self.lambda_hinge}"
        )
