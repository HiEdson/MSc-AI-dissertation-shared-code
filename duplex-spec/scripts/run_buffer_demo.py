"""Demo: the core latency-vs-safety trade, swept over the commit threshold.

Runs many speculation episodes over the mock stream at different gate settings
and prints the table that will become a figure in your dissertation:
conservative gates save little latency but rarely glitch; aggressive gates save
more but roll back more often.

    python scripts/run_buffer_demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec import (  # noqa: E402
    BackboneRolloutPredictor,
    BufferStats,
    EntropyGate,
    SpeculativeBuffer,
)
from duplex_spec.backbones import StochasticMockBackbone  # noqa: E402


def run_episodes(threshold: float, n_episodes: int = 400, horizon_ms: float = 320.0):
    n_ch = n_cb = None
    stats = None
    last = None
    for ep in range(n_episodes):
        # Fresh, independent schedule per episode (seed=ep) -> fair Monte Carlo.
        bb = StochasticMockBackbone(seed=ep, p_honest=0.15, p_treacherous=0.10)
        buf = SpeculativeBuffer(bb, BackboneRolloutPredictor(bb),
                                EntropyGate(threshold=threshold), horizon_ms=horizon_ms)
        if stats is None:
            stats = BufferStats(horizon_frames=buf.horizon)
            n_ch, n_cb = bb.config.n_channels, bb.config.n_codebooks
            last = np.zeros((n_ch, n_cb), dtype=np.int64)
        truth = np.stack([bb.true_next_pair(k) for k in range(buf.horizon)], axis=1)
        spec = buf.speculate(last, bb.reset())
        stats.update(buf.resolve(spec, truth))
    return stats


def main():
    print(f"{'threshold':>10} {'commit_prec':>12} {'rollback_rate':>14} {'mean_saved_ms':>14}")
    print("-" * 54)
    for thr in (0.02, 0.05, 0.10, 0.20, 0.40, 0.80):
        s = run_episodes(thr)
        print(f"{thr:>10.2f} {s.commit_precision:>11.1%} {s.rollback_rate:>13.1%} "
              f"{s.mean_latency_saved_ms:>13.0f}")
    print("\nReading this: higher threshold commits through more frames -> more"
          "\nlatency saved, but precision is capped by the ~10% 'treacherous'"
          "\n(confident-but-wrong) frames the entropy gate CANNOT see. Lowering the"
          "\nthreshold does not fix those -> motivation for the consistency gate (v1).")


if __name__ == "__main__":
    main()
