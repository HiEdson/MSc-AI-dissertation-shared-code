"""Tests for the stochastic mock: it must produce the two error regimes that
make the ablation meaningful, and the entropy gate must show its blind spot."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec import BackboneRolloutPredictor, EntropyGate, SpeculativeBuffer  # noqa: E402
from duplex_spec.backbones import StochasticMockBackbone  # noqa: E402


def test_reproducible():
    a = StochasticMockBackbone(seed=1)
    b = StochasticMockBackbone(seed=1)
    assert np.array_equal(a.true_next_pair(5), b.true_next_pair(5))


def test_has_both_error_kinds():
    bb = StochasticMockBackbone(p_honest=0.2, p_treacherous=0.2, seed=3)
    kinds = {bb._frame_kind(t) for t in range(200)}
    assert {"easy", "honest", "treacherous"} <= kinds


def test_entropy_gate_blind_spot_on_treacherous():
    """A purely-treacherous stream: low entropy everywhere, yet commits are wrong.

    Tightening the threshold cannot save precision here -- that's the blind spot.
    """
    bb = StochasticMockBackbone(p_honest=0.0, p_treacherous=1.0, seed=0)
    last = np.zeros((bb.config.n_channels, bb.config.n_codebooks), dtype=np.int64)

    def precision_at(threshold):
        buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb),
                                EntropyGate(threshold), horizon_ms=320)
        state = bb.reset()
        truth = np.stack([bb.true_next_pair(t) for t in range(buf.horizon)], axis=1)
        spec = buf.speculate(last, state)
        r = buf.resolve(spec, truth)
        return spec.tokens.shape[1], r.n_correct

    tight_committed, tight_correct = precision_at(0.05)
    loose_committed, loose_correct = precision_at(0.80)
    # Confident frames -> low entropy -> gate commits even when tight...
    assert tight_committed > 0
    # ...but every committed frame is wrong (realized = base + offset).
    assert tight_correct == 0 and loose_correct == 0


def test_honest_frames_are_gated_out():
    """Pure-honest stream: high entropy -> gate refuses to commit -> no glitch."""
    bb = StochasticMockBackbone(p_honest=1.0, p_treacherous=0.0, seed=0)
    last = np.zeros((bb.config.n_channels, bb.config.n_codebooks), dtype=np.int64)
    buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb), EntropyGate(0.5), horizon_ms=320)
    spec = buf.speculate(last, bb.reset())
    assert spec.tokens.shape[1] == 0          # committed nothing -> safe


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all stochastic-mock tests passed")
