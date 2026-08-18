"""Moshi backbone adapter -- config + seam confirmed from kyutai-labs/moshi.

All numbers verified from moshi/moshi/models/loaders.py and models/lm.py
(not guessed).

THE SEAM (what makes this our preferred backbone):
  LMGen._step(input_tokens) returns (out_tokens, transformer_out).
  `transformer_out` is the temporal transformer's post-out_norm hidden state,
  shape [B, 1, dim=4096], and it is exactly what the depformer consumes to emit
  the 8 audio codebooks. Your speculative head reads THIS.
  Moshi even ships LMGen.step_with_extra_heads(), which runs auxiliary heads
  (nn.Linear(dim, ...)) on transformer_out -- a blessed extension point. Your
  multi-step head follows the same pattern (richer than the 6-dim extra_heads).

CACHE-FEATURES PLAN (fits 16GB):
  Run LMGen over audio in no_grad, capture transformer_out per frame, write to
  disk. Train the multi-step head on the cached 4096-d vectors with Moshi
  unloaded -> no gradients through the 8B backbone.

CAVEATS:
  - Streaming uses CUDA graphs (graphed_main/graphed_depth): fine to READ
    transformer_out, painful to modify the forward.
  - The depformer samples the 8 codebooks sequentially within a frame, so
    collecting all-codebook logits at once for the commit strategy needs a
    small adaptation (loop forward_depformer per codebook, or read pre-sample
    logits).
  - On 16GB, use a quantized checkpoint for inference (kyutai/moshiko-pytorch-q8
    ~8GB). bf16 (~16GB) leaves no headroom.

DUAL-STREAM: the LM carries n_q=16 audio codebooks = 8 user (input) + 8 Moshi
(generated). That genuine two-channel structure is why this backbone fits the
turn-anticipation hypothesis (unlike a TTS or a half-duplex small model).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backbone import StepOutput
from ..config import BackboneConfig

# Verified from models/loaders.py (SAMPLE_RATE, FRAME_RATE, bins, dim, dep_q).
MOSHI_CONFIG = BackboneConfig(
    name="moshi",
    frame_rate_hz=12.5,        # 1 frame = 80 ms; human ~200ms gap = 2.5 frames
    sample_rate=24000,
    n_channels=2,              # user (input) + Moshi (generated)
    n_codebooks=8,             # dep_q: codebooks generated per frame
    codebook_size=2048,        # quantizer "bins"
    hidden_dim=4096,           # temporal transformer dim -> transformer_out width
    has_text_stream=True,      # inner-monologue text token
)


class MoshiBackbone:
    def __init__(self, checkpoint: str = "kyutai/moshiko-pytorch-q8", device: str = "cuda"):
        # TODO: load via moshi.models.loaders (CheckpointInfo / get_moshi_lm + Mimi),
        # build an LMGen, and keep it as self._gen. hidden_dim is already known.
        raise NotImplementedError(
            "Config is known (MOSHI_CONFIG). Wire loading to moshi.models.loaders "
            "and map step() -> LMGen._step (read transformer_out). Use MockBackbone meanwhile."
        )

    @property
    def config(self) -> BackboneConfig:
        return MOSHI_CONFIG

    def reset(self) -> Any:
        # TODO: enter LMGen streaming context; return its state handle.
        ...

    def encode(self, wav: np.ndarray) -> np.ndarray:
        # TODO: Mimi.encode (models/compression.py) -> [n_channels, n_frames, n_codebooks]
        ...

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        # TODO: Mimi.decode
        ...

    def step(self, input_pair: np.ndarray, state: Any) -> tuple[StepOutput, Any]:
        # TODO: feed input_pair to LMGen._step; set
        #   StepOutput.hidden  = transformer_out[:, 0]          (4096-d)
        #   StepOutput.logits  = per-codebook depformer logits  (loop forward_depformer)
        ...
