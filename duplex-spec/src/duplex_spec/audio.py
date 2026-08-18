"""Rollback mechanics for the speculative buffer.

When a speculation is wrong, we have already emitted audio the user is hearing.
A hard cut is audible and ugly, so rollback = fade the wrong tail out (and, if
we have a corrected continuation, crossfade into it). Getting this right is part
of the *naturalness* claim, not just the latency claim.

Pure numpy. Works the same regardless of backbone.

Equal-power (constant-energy) curves are used rather than linear because linear
crossfades dip in perceived loudness through the middle of the transition.
"""

from __future__ import annotations

import numpy as np


def _equal_power_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_out_gain, fade_in_gain), each length n, with g_out^2+g_in^2=1."""
    t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)
    fade_in = np.sin(0.5 * np.pi * t)
    fade_out = np.cos(0.5 * np.pi * t)
    return fade_out, fade_in


def fade_out(audio: np.ndarray, n: int) -> np.ndarray:
    """Fade the LAST n samples of `audio` to silence (equal-power). Returns a copy.

    audio: (..., n_samples). Used to retract wrongly pre-fired audio cleanly.
    """
    if n <= 0 or audio.shape[-1] == 0:
        return audio.copy()
    n = min(n, audio.shape[-1])
    out = audio.copy()
    g_out, _ = _equal_power_ramps(n)
    out[..., -n:] = out[..., -n:] * g_out
    return out


def crossfade(prefix: np.ndarray, suffix: np.ndarray, n: int) -> np.ndarray:
    """Splice suffix onto prefix with an n-sample equal-power crossfade.

    The tail of `prefix` (the speculated audio we are abandoning) fades out while
    the head of `suffix` (the corrected continuation) fades in, overlapping.
    prefix, suffix: (..., n_samples).
    """
    if n <= 0:
        return np.concatenate([prefix, suffix], axis=-1)
    n = min(n, prefix.shape[-1], suffix.shape[-1])
    if n == 0:
        return np.concatenate([prefix, suffix], axis=-1)

    g_out, g_in = _equal_power_ramps(n)
    head = prefix[..., :-n]
    overlap = prefix[..., -n:] * g_out + suffix[..., :n] * g_in
    tail = suffix[..., n:]
    return np.concatenate([head, overlap, tail], axis=-1)
