"""Offline speculative-evaluation logic (pure numpy, tested).

Turns predicted token trajectories into latency-saved / rollback numbers under a
*frame-acceptance* criterion. Exact token match is too strict for real lossy
audio (per-token accuracy ~0.3 makes a full 16-token frame rarely identical), so
acceptance is configurable:

  - "exact": every (channel, codebook) token matches  (the mock's old rule)
  - "cb0":   the coarse codebook matches on both channels  (content correct;
             default — cb0 carries phonetic/semantic/turn-taking content)
  - "frac":  at least `frac` of the tokens match

Latency saved = length of the contiguous acceptable prefix of the committed
frames; a glitch/rollback happens when the policy commits past that prefix.
"""

from __future__ import annotations

import numpy as np


def frame_acceptable(committed: np.ndarray, real: np.ndarray,
                     mode: str = "cb0", frac: float = 0.5) -> bool:
    """committed, real: one frame each, shape [C, Q]."""
    if mode == "exact":
        return bool(np.array_equal(committed, real))
    if mode == "cb0":
        return bool(np.all(committed[:, 0] == real[:, 0]))
    if mode == "frac":
        return float((committed == real).mean()) >= frac
    raise ValueError(f"unknown acceptance mode: {mode}")


def acceptable_prefix(committed: np.ndarray, real: np.ndarray,
                      mode: str = "cb0", frac: float = 0.5) -> int:
    """Longest contiguous acceptable prefix. Both [n, C, Q]."""
    n = min(committed.shape[0], real.shape[0])
    for k in range(n):
        if not frame_acceptable(committed[k], real[k], mode, frac):
            return k
    return n


def mean_norm_entropy(frame_logits: np.ndarray) -> float:
    """Mean normalised entropy over a frame's [C, Q, V] logits, in [0, 1]."""
    x = frame_logits - frame_logits.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)          # [C, Q]
    return float((ent / np.log(frame_logits.shape[-1])).mean())


def entropy_commit_len(logits: np.ndarray, threshold: float) -> int:
    """How many leading frames to commit while mean entropy < threshold.

    logits: [K, C, Q, V]. Stops at the first frame whose entropy >= threshold
    (longest-accepted-prefix semantics).
    """
    n = 0
    for k in range(logits.shape[0]):
        if mean_norm_entropy(logits[k]) < threshold:
            n += 1
        else:
            break
    return n


def episode(committed: np.ndarray, real: np.ndarray, n_commit: int,
            mode: str = "cb0", frac: float = 0.5, ms_per_frame: float = 80.0):
    """One speculation episode.

    committed, real: [K, C, Q] greedy predicted vs real future frames.
    n_commit: frames the policy chose to pre-fire.
    Returns (n_correct, glitch, latency_saved_ms).
    """
    n_commit = max(0, min(n_commit, committed.shape[0]))
    acc = acceptable_prefix(committed[:n_commit], real[:n_commit], mode, frac)
    return acc, (n_commit > acc), acc * ms_per_frame


def stability_commit_lengths(pred: np.ndarray, m: int, cb: int = 0) -> np.ndarray:
    """Amendable / stability commit gate (vectorised, one conversation).

    The multi-step head predicts each absolute future frame from several vantage
    points: horizon k at position i, horizon k+1 at position i-1, and so on, each
    made with one more frame of real input. We commit horizon k at position i only
    if its prediction of that absolute frame has been IDENTICAL (on codebook `cb`,
    all channels) across the last `m` vantage points — i.e. it stopped being
    amended as evidence arrived. This is the "amendable decision" criterion:
    commit when the tentative decision has converged, not merely when it is
    confident. Larger m = require a longer convergence history = stricter/safer.

    pred: [N, K, C, Q] greedy predictions for ONE conversation (contiguous frames).
    Returns commit_len[N]: leading run of horizons that pass the stability test.
    """
    N, K, C, Q = pred.shape
    code = pred[:, :, :, cb]                              # [N, K, C]
    stable = np.zeros((N, K), dtype=bool)
    cols = K - (m - 1)                                    # horizons with enough history
    if cols > 0:
        stable[m - 1:, :cols] = True                     # validity mask (rows & horizons)
    for j in range(1, m):
        if K - j <= 0 or N - j <= 0:
            stable[:] = False
            break
        agree = np.zeros((N, K), dtype=bool)
        a = code[j:, : K - j, :]                          # position i, horizon k
        b = code[: N - j, j:K, :]                         # position i-j, horizon k+j (same frame)
        agree[j:, : K - j] = (a == b).all(axis=2)
        stable &= agree
    return np.cumprod(stable, axis=1).sum(axis=1).astype(int)   # leading-true run per position


def relaxed_commit_lengths(pred: np.ndarray, window: int, min_agree: int,
                           cb: int = 0, ent: np.ndarray = None,
                           ent_floor: float = None) -> np.ndarray:
    """Relaxed amendable gate: k-of-m agreement (with optional confidence floor).

    Like stability_commit_lengths, but instead of requiring ALL `window` recent
    vantage points to agree on the prediction of an absolute frame, it requires
    only at least `min_agree` of them to match the current (most-informed)
    prediction (codebook `cb`, both channels). min_agree == window recovers the
    strict gate; min_agree < window commits MORE frames (recovers coverage) at
    some precision cost. If `ent`/`ent_floor` are given, also require the current
    prediction's normalised entropy < ent_floor (a precision-recovery add-on).

    pred: [N, K, C, Q] for ONE conversation (contiguous frames).
    Returns commit_len[N].
    """
    N, K, C, Q = pred.shape
    code = pred[:, :, :, cb]                              # [N, K, C]
    count = np.zeros((N, K), dtype=np.int16)
    for j in range(window):
        if j == 0:
            count += 1                                    # current prediction agrees with itself
            continue
        if K - j <= 0 or N - j <= 0:
            break
        a = code[j:, : K - j, :]                          # current code[i, k]
        b = code[: N - j, j:, :]                          # vantage code[i-j, k+j] (same frame)
        agree = np.zeros((N, K), dtype=bool)
        agree[j:, : K - j] = (a == b).all(axis=2)
        count += agree
    valid = np.zeros((N, K), dtype=bool)
    cols = K - (window - 1)
    if cols > 0:
        valid[window - 1:, :cols] = True                  # full window must exist
    committable = valid & (count >= min_agree)
    if ent_floor is not None and ent is not None:
        committable &= (ent < ent_floor)
    return np.cumprod(committable, axis=1).sum(axis=1).astype(int)
