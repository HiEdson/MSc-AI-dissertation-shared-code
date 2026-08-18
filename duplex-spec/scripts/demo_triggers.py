"""Core of the unified demo harness: the TRIGGER interface and shared decode helper.

Every method you built is expressed as a Trigger: given the per-frame prediction state up
to frame t, it returns a boolean "release Moshi to speak this frame". The harness then runs
ONE continuous gate/release loop (drive_moshi in run_demo.py) identically for all triggers,
so the comparison is clean --- only the trigger logic differs.

Triggers (the rows the report/professor needs):
  event triggers (fire on their own logic, no acceptance rule):
    oracle       -- ground-truth handoff frames (upper bound / ceiling)
    handoff_lr   -- weak activity logistic regression (handoff_predictor.py)
    handoff_gru  -- strong GRU on h_t + distribution features (handoff_gru.py)
    entropy      -- confidence gate (baseline): release when cb0 entropy < floor
  gate triggers (commit-based; run under BOTH cb0 and top5 acceptance):
    argmax       -- amendable/stability gate (original criterion)
    js           -- JS-divergence gate (contribution)
    probe        -- trained probe (contribution)

Two axes: trigger x acceptance-rule (cb0 | top5) for the gate triggers.

Metrics kept deliberately in two groups (see run_demo.py):
  - offline PROXY: rollback%, saved_ms, coverage  (vs ground truth; a proxy for naturalness)
  - BEHAVIOURAL:   barge_in%, release_lead dist, audio  (what a listener actually judges)
Rollback is reported but NOT treated as ground-truth-natural: committing a different but
plausible token is not necessarily an unnatural turn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

_EPS = 1e-8
_LN2 = float(np.log(2.0))


# --------------------------- per-frame prediction state ---------------------------
@dataclass
class FrameState:
    """Everything a trigger might read at frame t. Precomputed once from the head so all
    triggers see identical inputs."""
    t: int
    p0_now: np.ndarray            # [C, V] cb0 distribution, horizon k=1, this vantage
    p0_prev: np.ndarray | None    # [C, V] same future frame predicted one tick earlier (or None)
    ent_now: np.ndarray           # [C] normalised entropy of p0_now
    hidden: np.ndarray | None     # [D] backbone h_t (for GRU/LR handoff triggers)
    activity: np.ndarray          # [C] 0/1 active flags at t


# --------------------------- acceptance rules ---------------------------
def accept_cb0(p0, true_tok):
    """argmax(p0) == true token, per channel. p0 [C,V], true_tok [C]."""
    return (p0.argmax(-1) == true_tok)


def accept_topk(p0, true_tok, k, ent_floor):
    """true token in confident top-k, per channel."""
    V = p0.shape[-1]
    ptrue = p0[np.arange(len(true_tok)), true_tok]
    rank = (p0 > ptrue[:, None]).sum(-1)
    ent = -(np.clip(p0, _EPS, 1) * np.log(np.clip(p0, _EPS, 1))).sum(-1) / np.log(V)
    return (rank < k) & (ent < ent_floor)


# --------------------------- JS helper ---------------------------
def js_div(p, q):
    p = np.clip(p, _EPS, 1); q = np.clip(q, _EPS, 1)
    m = 0.5 * (p + q)
    return 0.5 * (p * (np.log(p) - np.log(m))).sum(-1) + \
           0.5 * (q * (np.log(q) - np.log(m))).sum(-1)


# --------------------------- trigger base ---------------------------
class Trigger:
    """A trigger decides, per frame, whether to RELEASE Moshi (let it speak).
    `threshold` is the single knob the harness calibrates to a rollback budget."""
    name = "base"
    has_acceptance = False

    def __init__(self, threshold: float = 0.5, **kw):
        self.threshold = threshold
        self.cfg = kw

    def reset(self):
        """Clear any per-conversation state (called at the start of each conversation)."""
        pass

    def fire(self, st: FrameState) -> bool:
        raise NotImplementedError


# ---- event triggers ----
class OracleTrigger(Trigger):
    name = "oracle"
    def __init__(self, handoff_frames, lead=2, **kw):
        super().__init__(**kw)
        self.ho = set(int(f) - lead for f in handoff_frames)   # fire `lead` frames early
    def fire(self, st): return st.t in self.ho


class EntropyTrigger(Trigger):
    name = "entropy"
    # release when the model is confident (low entropy) on BOTH channels
    def fire(self, st): return bool((st.ent_now < self.threshold).all())


class HandoffModelTrigger(Trigger):
    """Wraps a learned handoff model (LR or GRU) that outputs time-to-handoff or a score.
    predict_fn(state_window) -> scalar; fire when it crosses threshold."""
    name = "handoff_model"
    def __init__(self, predict_fn, mode="time_le", **kw):
        super().__init__(**kw)
        self.predict_fn = predict_fn; self.mode = mode
    def fire(self, st):
        v = self.predict_fn(st)
        if v is None: return False
        return (v <= self.threshold) if self.mode == "time_le" else (v >= self.threshold)


# ---- gate triggers (commit-based) ----
class ArgmaxGate(Trigger):
    name = "argmax"; has_acceptance = True
    # release when cb0 argmax is stable vs the previous vantage on both channels
    def fire(self, st):
        if st.p0_prev is None: return False
        stable = (st.p0_now.argmax(-1) == st.p0_prev.argmax(-1)).all()
        conf = (st.ent_now < self.cfg.get("ent_floor", 0.5)).all()
        return bool(stable and conf)


class JSGate(Trigger):
    name = "js"; has_acceptance = True
    # release when the distribution barely moved (JS < tau) and is confident
    def fire(self, st):
        if st.p0_prev is None: return False
        d = js_div(st.p0_now, st.p0_prev)              # [C]
        moved = (d < self.threshold).all()
        conf = (st.ent_now < self.cfg.get("ent_floor", 0.5)).all()
        return bool(moved and conf)


class ProbeGate(Trigger):
    name = "probe"; has_acceptance = True
    # release when the trained probe's acceptability score exceeds threshold
    def __init__(self, score_fn, **kw):
        super().__init__(**kw)
        self.score_fn = score_fn
    def fire(self, st):
        s = self.score_fn(st)
        return s is not None and s >= self.threshold


# --------------------------- Mimi decode helper ---------------------------
def decode_tokens_to_wav(mimi, frame_tokens, device, sil_pad=2):
    """frame_tokens: [C, Q, T] int -> waveform via Mimi. Shared by the top-5 demo and the
    harness. Assumes mimi.decode expects [B, Q, T] per channel; adjust per your Stage-A code.

    Returns np.float32 waveform. This mirrors the Stage-A decode path; if your Stage-A used a
    specific streaming/exec-mask setup, replicate it here.
    """
    import torch
    C, Q, T = frame_tokens.shape
    wavs = []
    with torch.no_grad():
        for c in range(C):
            codes = torch.from_numpy(frame_tokens[c][None]).to(device).long()  # [1,Q,T]
            wav = mimi.decode(codes)                                            # [1,1,samples]
            wavs.append(wav.squeeze().cpu().numpy())
    # simple sum-mix of the two channels for a single listenable track
    n = max(len(w) for w in wavs)
    mix = np.zeros(n, np.float32)
    for w in wavs:
        mix[:len(w)] += w.astype(np.float32)
    return mix / max(np.abs(mix).max(), 1e-6)
