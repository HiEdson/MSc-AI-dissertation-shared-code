"""The backbone interface.

Anything that satisfies `DuplexBackbone` can be driven by your speculative
head. NTPP, Moshi, Mini-Omni2 and the MockBackbone are all just
implementations of this one protocol. Your novel code imports THIS, never a
concrete model.

The streaming contract is deliberately tiny:

    reset()      -> opaque state (KV cache etc.)
    encode(wav)  -> token grid               (offline, for teacher forcing / data prep)
    decode(grid) -> wav                       (turn predicted tokens back into audio)
    step(pair, state) -> StepOutput           (ONE streaming tick)

`StepOutput.hidden` is the representation your speculative head consumes.
`StepOutput.logits` is the backbone's own next-pair distribution -- useful as a
verification signal for the speculative buffer (this is the seam where the
speculative-decoding analogy enters).

Token grids are plain numpy arrays of shape:
    (n_channels, n_frames, n_codebooks)   dtype int64
A single streaming "pair" is one frame across channels/codebooks:
    (n_channels, n_codebooks)             dtype int64
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import BackboneConfig


@dataclass
class StepOutput:
    hidden: np.ndarray   # (hidden_dim,)            -- what your head reads
    logits: np.ndarray   # (n_channels, n_codebooks, codebook_size) -- backbone's own next-pair scores


@runtime_checkable
class DuplexBackbone(Protocol):
    @property
    def config(self) -> BackboneConfig: ...

    def reset(self) -> Any:
        """Return a fresh streaming state (KV cache / RNN state / whatever)."""
        ...

    def encode(self, wav: np.ndarray) -> np.ndarray:
        """wav (n_channels, n_samples) -> tokens (n_channels, n_frames, n_codebooks)."""
        ...

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """tokens (n_channels, n_frames, n_codebooks) -> wav (n_channels, n_samples)."""
        ...

    def step(self, input_pair: np.ndarray, state: Any) -> tuple[StepOutput, Any]:
        """Advance one frame.

        input_pair: (n_channels, n_codebooks) int64 -- the frame just observed.
        Returns the StepOutput for predicting the NEXT frame, and the new state.
        """
        ...
