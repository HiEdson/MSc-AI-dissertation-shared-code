"""Commit strategies -- the accept/reject criterion.

This is the conceptual heart of the project. In speculative *decoding* you accept
a drafted token if it matches the target model's distribution: there is a ground
truth to verify against. Here there is none -- you are betting on a future that
has not happened. So "verify" must mean something else, and the choice of
meaning is your contribution. We make it a swappable strategy.

A strategy looks at a proposed trajectory (a list of per-frame StepOutputs from
the lookahead predictor) and returns how many leading frames are safe to commit
-- the "longest accepted prefix", mirroring speculative decoding's accept loop.

  v0  EntropyGate            -- commit while predictive entropy stays low. Cheap,
                                model-free, today. Weakness: confident != correct.
  v1  BackboneConsistencyGate -- STUB. Verify each speculated frame against the
                                frozen backbone's own next-pair distribution as
                                real audio arrives. The honest analogue of draft-verify.
  v2  LearnedCommitGate       -- STUB. A tiny head trained to predict "safe to fire?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .backbone import StepOutput


@dataclass
class CommitDecision:
    n_accept: int                       # leading frames safe to pre-fire
    scores: list[float] = field(default_factory=list)  # per-frame criterion value (for logging/plots)


class CommitStrategy(Protocol):
    def decide(self, trajectory: list[StepOutput]) -> CommitDecision: ...


def _normalized_entropy(logits: np.ndarray) -> float:
    """Mean entropy across channels/codebooks, normalised to [0, 1].

    logits: (n_channels, n_codebooks, codebook_size). 0 = fully confident,
    1 = uniform. Normalising by log(vocab) makes the threshold codebook-size
    independent -- important for the backbone-agnostic goal.
    """
    x = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)          # (n_channels, n_codebooks)
    return float((ent / np.log(logits.shape[-1])).mean())


class EntropyGate:
    """v0: accept leading frames while normalised entropy < threshold.

    threshold in [0,1]. Lower = more conservative (fewer pre-fires, fewer
    rollbacks). This single knob is your conservative<->aggressive ablation axis.
    """

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def decide(self, trajectory: list[StepOutput]) -> CommitDecision:
        scores: list[float] = []
        n_accept = 0
        for step in trajectory:
            h = _normalized_entropy(step.logits)
            scores.append(h)
            if h < self.threshold:
                n_accept += 1
            else:
                break  # stop at first uncertain frame (accept-prefix semantics)
        return CommitDecision(n_accept=n_accept, scores=scores)


class BackboneConsistencyGate:
    """v1 STUB -- the 'proper' draft-verify analogue.

    As each real frame arrives, compare the speculated frame against the frozen
    backbone's own step() distribution; accept while they agree within tolerance.
    Needs the live backbone, so it's built once a real backbone is wired.
    """

    def __init__(self, backbone, tol: float = 0.1):
        raise NotImplementedError("Implement against a real DuplexBackbone after NTPP is wired.")

    def decide(self, trajectory: list[StepOutput]) -> CommitDecision:  # pragma: no cover
        raise NotImplementedError
