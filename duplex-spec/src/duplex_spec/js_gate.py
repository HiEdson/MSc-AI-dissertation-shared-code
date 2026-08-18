"""JS-divergence commit gate --- a distribution-based replacement for the argmax
equality test in the amendable criterion.

WHY
---
`stability_commit_lengths` commits horizon k only if the cb0 ARGMAX is byte-identical
across the last m vantage points, on all channels. Diagnostics showed 25.8% of argmax
flips have a *stable, confident* distribution underneath: the peak wobbles between two
near-tied candidates while the model's belief has not moved. The strict test discards
all of them, which is a major reason coverage sits at ~0.03 frames/speculation.

THE TEST
--------
For the same absolute future frame predicted from successive vantage points, compute
the Jensen-Shannon divergence between the two cb0 distributions:

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),   M = 0.5 (P + Q)

Commit horizon k iff, on EVERY channel and across the last m vantage points:
    JS < tau                      (the belief did not move)
  AND normalised entropy < floor  (the belief is confident, not uniformly clueless)

The entropy floor is essential: two near-uniform distributions also give JS ~ 0, so a
pure-JS gate would commit precisely the frames of maximal uncertainty.

Argmax equality is the degenerate special case where the peak happens to hold, so this
generalises the amendable criterion rather than replacing it.

USAGE
-----
Import alongside the existing gate and sweep tau the way you sweep m:

    from duplex_spec.js_gate import js_commit_lengths
    clen = js_commit_lengths(p0, m=3, tau=0.05, ent_floor=0.5)

where p0 is the per-conversation cb0 distribution array [N, K, C, V] and the returned
commit lengths have exactly the same meaning/shape as stability_commit_lengths.

NOTE: JS here uses natural log, so it is bounded [0, ln 2 ~ 0.693]. Divide by ln 2 if
you prefer a [0, 1] scale; tau values below assume the natural-log scale.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-8
_LN2 = float(np.log(2.0))


def js_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence between two distributions (natural log, bounded [0, ln2])."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=axis)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=axis)
    return 0.5 * kl_pm + 0.5 * kl_qm


def normalised_entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy normalised to [0, 1] by log V."""
    V = p.shape[axis]
    p = np.clip(p, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis) / float(np.log(V))


def js_commit_lengths(p0: np.ndarray, m: int, tau: float,
                      ent_floor: float | None = 0.5) -> np.ndarray:
    """Distribution-stability commit gate (vectorised, one conversation).

    Mirrors stability_commit_lengths exactly, but tests JS between distributions of
    the same absolute frame across vantage points instead of argmax equality.

    p0:        [N, K, C, V] cb0 probability distributions for ONE conversation
               (contiguous frames), N positions, K horizons, C channels, V codes.
    m:         number of vantage points that must agree (>= 2 to test anything).
    tau:       JS threshold; commit only where JS < tau on every channel/comparison.
    ent_floor: normalised-entropy ceiling; None disables the confidence requirement
               (NOT recommended --- a pure-JS gate commits uniform distributions).

    Returns commit_len[N]: leading run of horizons passing the test, same as the
    argmax gate, so it drops straight into the existing metrics.
    """
    N, K, C, V = p0.shape
    stable = np.zeros((N, K), dtype=bool)
    cols = K - (m - 1)
    if cols > 0:
        stable[m - 1:, :cols] = True                       # same validity mask as argmax gate
    else:
        return np.zeros(N, dtype=int)

    # confidence requirement on the CURRENT (most-informed) prediction
    if ent_floor is not None:
        ent = normalised_entropy(p0, axis=-1)              # [N, K, C]
        conf = (ent < ent_floor).all(axis=2)               # [N, K] every channel confident
        stable &= conf

    for j in range(1, m):
        if K - j <= 0 or N - j <= 0:
            stable[:] = False
            break
        agree = np.zeros((N, K), dtype=bool)
        a = p0[j:, : K - j, :, :]                          # position i,   horizon k
        b = p0[: N - j, j:K, :, :]                         # position i-j, horizon k+j (same frame)
        d = js_divergence(a, b, axis=-1)                   # [., ., C]
        agree[j:, : K - j] = (d < tau).all(axis=2)         # every channel's belief held still
        stable &= agree

    return np.cumprod(stable, axis=1).sum(axis=1).astype(int)


def argmax_commit_lengths_from_dist(p0: np.ndarray, m: int) -> np.ndarray:
    """The ORIGINAL argmax gate, recomputed from distributions.

    Useful as an in-script control: run this and js_commit_lengths on the same p0 to
    confirm any coverage gain comes from the JS test and not from a data difference.
    """
    N, K, C, V = p0.shape
    code = p0.argmax(-1)                                   # [N, K, C]
    stable = np.zeros((N, K), dtype=bool)
    cols = K - (m - 1)
    if cols > 0:
        stable[m - 1:, :cols] = True
    else:
        return np.zeros(N, dtype=int)
    for j in range(1, m):
        if K - j <= 0 or N - j <= 0:
            stable[:] = False
            break
        agree = np.zeros((N, K), dtype=bool)
        a = code[j:, : K - j, :]
        b = code[: N - j, j:K, :]
        agree[j:, : K - j] = (a == b).all(axis=2)
        stable &= agree
    return np.cumprod(stable, axis=1).sum(axis=1).astype(int)
