"""The speculative buffer.

Lifecycle of one speculation episode (near a suspected turn-end):

  1. speculate(): ask the predictor for a K-frame lookahead trajectory, then ask
     the commit strategy how many leading frames are safe to pre-fire. Those are
     "in flight" -- audio the user is (notionally) already hearing.
  2. resolve(): the future arrives. Compare what we committed against the truth.
       - committed frames that were correct  -> latency saved (handoff shrinks)
       - committed frames that were wrong     -> rollback (fade them out): a glitch
     This asymmetry is the whole game: pre-firing wins time but an over-commit
     costs an audible artifact. The commit strategy's job is to ride that line.

Everything here is token-level and runs on CPU against the mock. `render()` shows
the audio-level rollback using the backbone's decoder + equal-power fade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import fade_out
from .backbone import DuplexBackbone, StepOutput
from .commit import CommitDecision, CommitStrategy
from .config import BackboneConfig
from .predictor import TrajectoryPredictor, trajectory_to_tokens
from .spec_eval import acceptable_prefix


def longest_matching_prefix(a: np.ndarray, b: np.ndarray) -> int:
    """Number of leading frames where token grids a,b agree fully.

    a, b: (n_channels, n_frames, n_codebooks). Compares min length.
    """
    n = min(a.shape[1], b.shape[1])
    for t in range(n):
        if not np.array_equal(a[:, t, :], b[:, t, :]):
            return t
    return n


@dataclass
class Speculation:
    trajectory: list[StepOutput]
    decision: CommitDecision
    tokens: np.ndarray            # (n_channels, n_accept, n_codebooks) committed/pre-fired


@dataclass
class EpisodeResult:
    n_committed: int
    n_correct: int
    n_rolled_back: int
    latency_saved_ms: float
    glitch: bool                  # did we pre-fire any wrong frame?


@dataclass
class BufferStats:
    n_episodes: int = 0
    frames_committed: int = 0
    frames_correct: int = 0
    frames_rolled_back: int = 0
    rollback_events: int = 0
    latency_saved_ms: float = 0.0
    horizon_frames: int = 0

    def update(self, r: EpisodeResult) -> None:
        self.n_episodes += 1
        self.frames_committed += r.n_committed
        self.frames_correct += r.n_correct
        self.frames_rolled_back += r.n_rolled_back
        self.rollback_events += int(r.glitch)
        self.latency_saved_ms += r.latency_saved_ms

    # --- the headline numbers for the dissertation ---
    @property
    def commit_precision(self) -> float:
        """Of frames we pre-fired, fraction that were correct. (Higher = fewer glitches.)"""
        return self.frames_correct / self.frames_committed if self.frames_committed else 1.0

    @property
    def rollback_rate(self) -> float:
        """Fraction of episodes that produced an audible rollback."""
        return self.rollback_events / self.n_episodes if self.n_episodes else 0.0

    @property
    def mean_latency_saved_ms(self) -> float:
        return self.latency_saved_ms / self.n_episodes if self.n_episodes else 0.0

    def summary(self) -> str:
        return (
            f"episodes={self.n_episodes}  "
            f"commit_precision={self.commit_precision:.2%}  "
            f"rollback_rate={self.rollback_rate:.2%}  "
            f"mean_latency_saved={self.mean_latency_saved_ms:.0f}ms  "
            f"horizon={self.horizon_frames}f"
        )


class SpeculativeBuffer:
    def __init__(
        self,
        backbone: DuplexBackbone,
        predictor: TrajectoryPredictor,
        strategy: CommitStrategy,
        *,
        horizon_ms: float = 240.0,
        accept: str = "exact",
        accept_frac: float = 0.5,
    ):
        self.backbone = backbone
        self.predictor = predictor
        self.strategy = strategy
        self.accept = accept          # "exact" | "cb0" | "frac" — how a pre-fired frame counts as correct
        self.accept_frac = accept_frac
        self.cfg: BackboneConfig = backbone.config
        self.horizon = self.cfg.frames_for_ms(horizon_ms)
        self.stats = BufferStats(horizon_frames=self.horizon)

    def speculate(self, last_pair: np.ndarray, state: Any) -> Speculation:
        traj = self.predictor.predict(last_pair, state, self.horizon)
        decision = self.strategy.decide(traj)
        tokens = trajectory_to_tokens(traj)[:, : decision.n_accept, :]
        return Speculation(trajectory=traj, decision=decision, tokens=tokens)

    def resolve(self, spec: Speculation, truth_tokens: np.ndarray) -> EpisodeResult:
        """truth_tokens: (n_channels, >=n_accept, n_codebooks) -- what actually happened."""
        n_committed = spec.tokens.shape[1]
        if n_committed:
            # spec.tokens / truth are [C, n, Q]; acceptable_prefix wants [n, C, Q]
            committed_fm = spec.tokens.transpose(1, 0, 2)
            truth_fm = truth_tokens.transpose(1, 0, 2)
            n_correct = acceptable_prefix(committed_fm, truth_fm, self.accept, self.accept_frac)
        else:
            n_correct = 0
        n_rolled_back = n_committed - n_correct
        # Only correctly pre-fired frames buy real handoff time; wrong ones cost a glitch.
        latency_saved = self.cfg.ms_for_frames(n_correct) if n_correct else 0.0
        result = EpisodeResult(
            n_committed=n_committed,
            n_correct=n_correct,
            n_rolled_back=n_rolled_back,
            latency_saved_ms=latency_saved,
            glitch=n_rolled_back > 0,
        )
        self.stats.update(result)
        return result

    def render(self, spec: Speculation, result: EpisodeResult) -> np.ndarray:
        """Produce the output waveform for the episode, fading out any wrong tail.

        Shows the audio-level consequence of a rollback. With the mock decoder
        this is silence, but the fade logic is real and backbone-independent.
        """
        if spec.tokens.shape[1] == 0:
            return np.zeros((self.cfg.n_channels, 0), dtype=np.float32)
        wav = self.backbone.decode(spec.tokens)              # (n_channels, n_samples)
        if result.glitch:
            fade_frames = result.n_rolled_back
            fade_samples = int(fade_frames * self.cfg.sample_rate / self.cfg.frame_rate_hz)
            wav = fade_out(wav, fade_samples)
        return wav
