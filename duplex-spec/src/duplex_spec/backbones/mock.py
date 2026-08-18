"""A fake backbone for developing everything that ISN'T the model.

Why this exists: on a 16GB card you do NOT want to load NTPP/Moshi every time
you tweak the speculative buffer. MockBackbone emits synthetic-but-structured
token-pairs deterministically, so you can write and unit-test the speculative
head, buffer, commit/rollback logic, and metrics on a laptop in milliseconds.
Swap in a real backbone only when the logic is proven.

It depends on numpy only -- no torch, no GPU, no downloads.

The synthetic signal has *just enough* structure to be predictable: each
channel walks a slow sinusoid through codebook index space, so a good predictor
should beat random and a bad one shouldn't. That lets your buffer's
accept/reject machinery actually be exercised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..backbone import StepOutput
from ..config import BackboneConfig


class MockBackbone:
    def __init__(self, seed: int = 0):
        # Small, fast, dual-channel, RVQ-ish. Frame rate mimics Mimi (12.5 Hz)
        # so your ms<->frame conversions get exercised realistically.
        self._cfg = BackboneConfig(
            name="mock",
            frame_rate_hz=12.5,
            sample_rate=24000,
            n_channels=2,
            n_codebooks=4,
            codebook_size=256,
            hidden_dim=64,
            has_text_stream=False,
        )
        self._rng = np.random.default_rng(seed)

    @property
    def config(self) -> BackboneConfig:
        return self._cfg

    def reset(self) -> Any:
        # State = current frame index. Real backbones return a KV cache here.
        return {"t": 0}

    def encode(self, wav: np.ndarray) -> np.ndarray:
        c = self._cfg
        n_frames = max(1, wav.shape[-1] // (c.sample_rate // int(c.frame_rate_hz)))
        return self._synth_grid(n_frames)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        c = self._cfg
        n_frames = tokens.shape[1]
        n_samples = int(n_frames * c.sample_rate / c.frame_rate_hz)
        # Mock decode = silence of the right length; real backbones synthesize audio.
        return np.zeros((c.n_channels, n_samples), dtype=np.float32)

    def step(self, input_pair: np.ndarray, state: Any) -> tuple[StepOutput, Any]:
        c = self._cfg
        t = state["t"]
        # Deterministic "hidden state" that depends on t -> reproducible tests.
        phase = 2 * np.pi * t / 50.0
        hidden = np.sin(phase + np.arange(c.hidden_dim) * 0.1).astype(np.float32)

        # Logits peaked at a slowly-moving target index per channel/codebook,
        # so the *true* next pair is learnable, not random noise.
        logits = np.full((c.n_channels, c.n_codebooks, c.codebook_size), -4.0, dtype=np.float32)
        for ch in range(c.n_channels):
            for cb in range(c.n_codebooks):
                target = int((np.sin(phase + ch + cb) * 0.5 + 0.5) * (c.codebook_size - 1))
                logits[ch, cb, target] = 6.0
        return StepOutput(hidden=hidden, logits=logits), {"t": t + 1}

    # -- helper: the "ground truth" future, so tests can score a predictor --
    def _synth_grid(self, n_frames: int) -> np.ndarray:
        c = self._cfg
        grid = np.zeros((c.n_channels, n_frames, c.n_codebooks), dtype=np.int64)
        for t in range(n_frames):
            phase = 2 * np.pi * t / 50.0
            for ch in range(c.n_channels):
                for cb in range(c.n_codebooks):
                    grid[ch, t, cb] = int((np.sin(phase + ch + cb) * 0.5 + 0.5) * (c.codebook_size - 1))
        return grid

    def true_next_pair(self, t: int) -> np.ndarray:
        """The frame the mock 'intends' at time t -- a target for your head."""
        return self._synth_grid(t + 1)[:, t, :]
