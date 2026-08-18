"""Tests for the speculative buffer pipeline. CPU-only, no torch."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec import (  # noqa: E402
    BackboneRolloutPredictor,
    EntropyGate,
    SpeculativeBuffer,
    longest_matching_prefix,
)
from duplex_spec.audio import crossfade, fade_out  # noqa: E402
from duplex_spec.backbones import MockBackbone  # noqa: E402


# ---------- audio rollback ----------

def test_fade_out_reaches_silence_and_preserves_head():
    sig = np.ones((2, 100), dtype=np.float32)
    out = fade_out(sig, 40)
    assert abs(out[:, -1]).max() < 1e-6          # last sample silent
    assert np.allclose(out[:, :60], 1.0)         # untouched head
    assert out.shape == sig.shape


def test_crossfade_equal_power_no_clip():
    a = np.ones((1, 50), dtype=np.float32)
    b = np.ones((1, 50), dtype=np.float32)
    out = crossfade(a, b, 20)
    # equal-power overlap of two equal signals stays within [~0.99, ~1.42], never clips wildly
    assert out.max() <= np.sqrt(2) + 1e-4
    assert out.shape[-1] == 50 + 50 - 20


# ---------- prefix matching ----------

def test_longest_matching_prefix():
    a = np.zeros((2, 5, 4), dtype=np.int64)
    b = np.zeros((2, 5, 4), dtype=np.int64)
    b[:, 3, :] = 7                                # diverge at frame 3
    assert longest_matching_prefix(a, b) == 3


# ---------- full buffer episode ----------

def _truth_tokens(bb, horizon):
    return np.stack([bb.true_next_pair(t) for t in range(horizon)], axis=1)


def test_conservative_gate_has_high_precision():
    bb = MockBackbone()
    buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb), EntropyGate(threshold=0.05),
                            horizon_ms=240)
    state = bb.reset()
    last = np.zeros((bb.config.n_channels, bb.config.n_codebooks), dtype=np.int64)

    truth = _truth_tokens(bb, buf.horizon)
    spec = buf.speculate(last, state)
    result = buf.resolve(spec, truth)

    # Mock is highly predictable + low entropy -> we should commit and be correct.
    assert spec.tokens.shape[1] >= 1
    assert result.n_correct == spec.tokens.shape[1]      # no wrong frames
    assert not result.glitch
    assert result.latency_saved_ms > 0


def test_threshold_trades_latency_for_safety():
    """The ablation axis: aggressive gate commits more frames than conservative."""
    bb = MockBackbone()
    last = np.zeros((bb.config.n_channels, bb.config.n_codebooks), dtype=np.int64)

    def committed(threshold):
        buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb),
                                EntropyGate(threshold=threshold), horizon_ms=240)
        spec = buf.speculate(last, bb.reset())
        return spec.tokens.shape[1]

    assert committed(0.5) >= committed(0.01)


def test_render_shapes_with_rollback():
    bb = MockBackbone()
    buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb), EntropyGate(0.2), horizon_ms=240)
    state = bb.reset()
    last = np.zeros((bb.config.n_channels, bb.config.n_codebooks), dtype=np.int64)
    spec = buf.speculate(last, state)
    # force a glitch by lying about the truth (all-different tokens)
    fake_truth = np.full_like(spec.tokens, 999) if spec.tokens.shape[1] else \
        np.zeros((bb.config.n_channels, 1, bb.config.n_codebooks), dtype=np.int64)
    result = buf.resolve(spec, fake_truth)
    wav = buf.render(spec, result)
    assert wav.shape[0] == bb.config.n_channels


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all buffer tests passed")
