"""Lookahead trajectory predictors.

The speculative buffer needs a K-frame lookahead trajectory to consider committing.
Where does it come from?

  BackboneRolloutPredictor -- roll the frozen backbone forward K steps,
      autoregressively. Needs no training. This is BOTH:
        (a) a working stand-in so we can build the buffer today, and
        (b) the legitimate VANILLA baseline you compare your learned head against.

  Your eventual multi-step NTPP head replaces this with a single-shot, cheaper
  trajectory prediction -- same interface, so the buffer never changes.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .backbone import DuplexBackbone, StepOutput


class TrajectoryPredictor(Protocol):
    def predict(self, last_pair: np.ndarray, state: Any, horizon: int) -> list[StepOutput]: ...


class BackboneRolloutPredictor:
    """Greedy autoregressive rollout of the backbone for `horizon` frames.

    NOTE: this mutates a *copy* of the streaming state so the caller's real
    state is untouched -- speculation must never corrupt the true context.
    """

    def __init__(self, backbone: DuplexBackbone):
        self.backbone = backbone

    def predict(self, last_pair: np.ndarray, state: Any, horizon: int) -> list[StepOutput]:
        import copy
        spec_state = copy.deepcopy(state)
        pair = last_pair.copy()
        traj: list[StepOutput] = []
        for _ in range(horizon):
            out, spec_state = self.backbone.step(pair, spec_state)
            traj.append(out)
            pair = out.logits.argmax(axis=-1)  # greedy next pair
        return traj


def trajectory_to_tokens(trajectory: list[StepOutput]) -> np.ndarray:
    """Greedy token grid (n_channels, horizon, n_codebooks) from a trajectory."""
    if not trajectory:
        return np.zeros((0,), dtype=np.int64)
    pairs = [s.logits.argmax(axis=-1) for s in trajectory]  # each (n_channels, n_codebooks)
    return np.stack(pairs, axis=1)
