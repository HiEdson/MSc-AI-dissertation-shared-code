"""Tests for the speculative-evaluation acceptance logic (pure numpy)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec.spec_eval import (  # noqa: E402
    acceptable_prefix,
    entropy_commit_len,
    episode,
    frame_acceptable,
    mean_norm_entropy,
)


def test_frame_acceptable_modes():
    a = np.array([[1, 2, 3], [4, 5, 6]])          # [C=2, Q=3]
    exact = a.copy()
    cb0_ok = a.copy(); cb0_ok[:, 1] = 99           # differ on a fine codebook only
    cb0_bad = a.copy(); cb0_bad[0, 0] = 99         # differ on coarse codebook
    assert frame_acceptable(a, exact, "exact")
    assert not frame_acceptable(a, cb0_ok, "exact")     # exact rejects any diff
    assert frame_acceptable(a, cb0_ok, "cb0")           # cb0 ignores fine codebooks
    assert not frame_acceptable(a, cb0_bad, "cb0")      # cb0 catches coarse diff
    assert frame_acceptable(a, cb0_ok, "frac", frac=0.5)  # 4/6 tokens match >= .5


def test_acceptable_prefix_stops_at_first_bad():
    # 3 frames; cb0 matches on frames 0,1, breaks on 2
    C, Q = 2, 4
    committed = np.zeros((3, C, Q), int)
    real = np.zeros((3, C, Q), int)
    real[2, 0, 0] = 7                               # coarse mismatch at frame 2
    assert acceptable_prefix(committed, real, "cb0") == 2


def test_cb0_prefix_at_least_exact_prefix():
    rng = np.random.default_rng(0)
    committed = rng.integers(0, 5, size=(5, 2, 8))
    real = committed.copy()
    real[3:] = rng.integers(0, 5, size=(2, 2, 8))   # diverge from frame 3
    assert acceptable_prefix(committed, real, "cb0") >= acceptable_prefix(committed, real, "exact")


def test_entropy_helpers():
    V = 16
    confident = np.full((2, 3, V), -10.0); confident[..., 0] = 10.0
    uniform = np.zeros((2, 3, V))
    assert mean_norm_entropy(confident) < 0.05
    assert mean_norm_entropy(uniform) > 0.95
    logits = np.stack([confident, uniform])         # [2 frames, C, Q, V]
    assert entropy_commit_len(logits, threshold=0.5) == 1   # commit frame0, stop at frame1


def test_episode_counts_correct_prefix_and_glitch():
    C, Q = 2, 4
    committed = np.zeros((3, C, Q), int)
    real = committed.copy(); real[2, 0, 0] = 9       # frame2 wrong on cb0
    n_correct, glitch, saved = episode(committed, real, n_commit=3, mode="cb0", ms_per_frame=80.0)
    assert n_correct == 2 and glitch and abs(saved - 160.0) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)


def test_stability_commit_lengths_m2_and_m3():
    import numpy as np
    from duplex_spec.spec_eval import stability_commit_lengths
    # cb0 values [N=4, K=3]; same absolute frame seen as horizon k at pos i and k+1 at i-1
    vals = np.array([[0, 5, 7],
                     [5, 7, 9],
                     [7, 99, 0],
                     [0, 0, 0]])
    pred = vals.reshape(4, 3, 1, 1)               # [N,K,C=1,Q=1]
    # m=2: pos1 both horizons converged ->2; pos2 only h0 ->1; pos0/3 ->0
    assert list(stability_commit_lengths(pred, m=2)) == [0, 2, 1, 0]
    # m=3: only pos2,h0 has 3 agreeing vantage points (7==7==7) -> 1
    assert list(stability_commit_lengths(pred, m=3)) == [0, 0, 1, 0]


def test_relaxed_commit_lengths_vote_and_strict_equivalence():
    import numpy as np
    from duplex_spec.spec_eval import relaxed_commit_lengths, stability_commit_lengths
    # min_agree == window recovers the strict gate
    strict_vals = np.array([[0, 5, 7], [5, 7, 9], [7, 99, 0], [0, 0, 0]]).reshape(4, 3, 1, 1)
    assert list(relaxed_commit_lengths(strict_vals, window=2, min_agree=2)) == \
           list(stability_commit_lengths(strict_vals, m=2))
    # case where 2-of-3 agree but not all 3: relaxed commits, strict does not
    vals = np.array([[0, 5, 99], [5, 7, 9], [7, 99, 0], [0, 0, 0]]).reshape(4, 3, 1, 1)
    assert list(relaxed_commit_lengths(vals, window=3, min_agree=2)) == [0, 0, 1, 0]
    assert list(relaxed_commit_lengths(vals, window=3, min_agree=3)) == [0, 0, 0, 0]
