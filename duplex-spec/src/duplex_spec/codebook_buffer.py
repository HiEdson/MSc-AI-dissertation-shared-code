"""Inner codebook buffer.

Moshi's depformer generates the codebooks of a single frame *sequentially*:
codebook 0 is sampled, fed back in to condition codebook 1, and so on, all
conditioned on the same `transformer_out`. The outer speculative buffer, though,
reasons one *frame* at a time. This helper bridges the two: it accumulates the
per-codebook outputs as the depformer loop runs, then assembles them into a
single frame once all codebooks are in, ready to hand to a StepOutput.

Two nested levels:
  - this CodebookBuffer  -> accumulates codebooks WITHIN a frame
  - SpeculativeBuffer    -> accumulates frames ACROSS the lookahead horizon

Important: the per-codebook distributions are CONDITIONAL (codebook k depends on
0..k-1), not independent. So a frame-level confidence score should combine them
as a joint quantity (we sum per-codebook entropies, i.e. the chain-rule upper
bound on joint entropy), never treat them as independent draws.

numpy only.
"""

from __future__ import annotations

import numpy as np


class CodebookBuffer:
    def __init__(self, n_codebooks: int, codebook_size: int):
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.reset()

    def reset(self) -> None:
        """Begin a new frame."""
        self._tokens = np.full(self.n_codebooks, -1, dtype=np.int64)
        self._logits = np.zeros((self.n_codebooks, self.codebook_size), dtype=np.float32)
        self._filled = 0

    def add(self, cb_index: int, logits: np.ndarray, token: int) -> None:
        """Record one codebook's logits (pre-sample) and the chosen token.

        Expects codebooks added in order 0..n_codebooks-1 (depformer order).
        """
        if cb_index != self._filled:
            raise ValueError(f"codebooks must arrive in order; expected {self._filled}, got {cb_index}")
        if logits.shape != (self.codebook_size,):
            raise ValueError(f"logits must be ({self.codebook_size},), got {logits.shape}")
        self._logits[cb_index] = logits
        self._tokens[cb_index] = token
        self._filled += 1

    def is_complete(self) -> bool:
        return self._filled == self.n_codebooks

    def assemble(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (tokens (n_codebooks,), logits (n_codebooks, codebook_size)) for the frame."""
        if not self.is_complete():
            raise RuntimeError(f"frame incomplete: {self._filled}/{self.n_codebooks} codebooks")
        return self._tokens.copy(), self._logits.copy()


def frame_entropy(logits: np.ndarray) -> float:
    """Joint-entropy proxy for one frame: SUM of per-codebook normalised entropies.

    logits: (n_codebooks, codebook_size). Normalised by log(codebook_size) so the
    result is comparable across backbones with different codebook sizes; summed
    (not averaged) because the chain rule makes the sum the natural joint measure.
    Divide by n_codebooks if you want a per-codebook mean instead.
    """
    x = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)          # (n_codebooks,)
    return float((ent / np.log(logits.shape[-1])).sum())
