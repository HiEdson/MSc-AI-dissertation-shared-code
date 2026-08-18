"""A mock with controllable, reproducible uncertainty.

The deterministic MockBackbone can't exercise the latency-vs-rollback trade: the
greedy rollout predicts it perfectly, so precision is always 100%. This subclass
injects two *different* kinds of error, on purpose, to show what the commit
strategy can and cannot handle:

  - "honest" frames: the model is genuinely unsure -> logits are flat (HIGH
    entropy) and the realized future is random. The EntropyGate SHOULD stop
    here, and does. This is uncertainty the gate handles well.

  - "treacherous" frames: the model is CONFIDENT (low entropy, peaked logits)
    but the realized future differs anyway. The EntropyGate sails straight
    through and commits a wrong frame -> a rollback glitch. The gate CANNOT
    avoid these by tuning its threshold, because entropy gives no warning.

That second category is the entropy gate's blind spot, and the whole motivation
for the backbone-consistency gate (v1): you need to check the speculation
against the backbone's realized distribution, not just its confidence.

Everything is a pure function of (seed, frame_index), so step() and
true_next_pair() stay consistent and the whole thing is reproducible.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..backbone import StepOutput
from .mock import MockBackbone


class StochasticMockBackbone(MockBackbone):
    def __init__(
        self,
        seed: int = 0,
        p_honest: float = 0.15,
        p_treacherous: float = 0.10,
        treacherous_offset: int = 7,
    ):
        super().__init__(seed=seed)
        self._cfg = replace(self._cfg, name="stochastic-mock")
        self._seed = seed
        self.p_honest = p_honest
        self.p_treacherous = p_treacherous
        self.offset = treacherous_offset

    # --- per-frame schedule, deterministic in (seed, t) ---
    def _frame_kind(self, t: int) -> str:
        r = np.random.default_rng((self._seed, t)).random()
        if r < self.p_honest:
            return "honest"
        if r < self.p_honest + self.p_treacherous:
            return "treacherous"
        return "easy"

    def _base_target(self, t: int, ch: int, cb: int) -> int:
        phase = 2 * np.pi * t / 50.0
        return int((np.sin(phase + ch + cb) * 0.5 + 0.5) * (self._cfg.codebook_size - 1))

    def step(self, input_pair: np.ndarray, state: Any) -> tuple[StepOutput, Any]:
        c = self._cfg
        t = state["t"]
        phase = 2 * np.pi * t / 50.0
        hidden = np.sin(phase + np.arange(c.hidden_dim) * 0.1).astype(np.float32)

        kind = self._frame_kind(t)
        if kind == "honest":
            # flat logits -> normalised entropy ~1.0 -> gate stops here
            logits = np.zeros((c.n_channels, c.n_codebooks, c.codebook_size), dtype=np.float32)
        else:
            # confident: peaked at base target (easy AND treacherous look identical here)
            logits = np.full((c.n_channels, c.n_codebooks, c.codebook_size), -4.0, dtype=np.float32)
            for ch in range(c.n_channels):
                for cb in range(c.n_codebooks):
                    logits[ch, cb, self._base_target(t, ch, cb)] = 6.0
        return StepOutput(hidden=hidden, logits=logits), {"t": t + 1}

    def true_next_pair(self, t: int) -> np.ndarray:
        c = self._cfg
        kind = self._frame_kind(t)
        rng = np.random.default_rng((self._seed, t, 999))
        out = np.zeros((c.n_channels, c.n_codebooks), dtype=np.int64)
        for ch in range(c.n_channels):
            for cb in range(c.n_codebooks):
                base = self._base_target(t, ch, cb)
                if kind == "easy":
                    out[ch, cb] = base                                   # confident & correct
                elif kind == "treacherous":
                    out[ch, cb] = (base + self.offset) % c.codebook_size  # confident & WRONG
                else:
                    out[ch, cb] = int(rng.integers(0, c.codebook_size))   # honest random
        return out
