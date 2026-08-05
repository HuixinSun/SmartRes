"""FastV baseline: prune image tokens in the LLM by attention score."""

from .fastv import install_fastv, remove_fastv

__all__ = ["install_fastv", "remove_fastv"]
