"""Smoke test: proves the abstraction layer works end-to-end on CPU.

Run with:  pytest -q   (or)   python tests/test_mock_smoke.py
No GPU, no torch, no downloads required.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec import DuplexBackbone, StepOutput  # noqa: E402
from duplex_spec.backbones import MockBackbone  # noqa: E402


def test_satisfies_protocol():
    bb = MockBackbone()
    assert isinstance(bb, DuplexBackbone)  # structural check


def test_ms_frame_bridge():
    cfg = MockBackbone().config
    # at 12.5 Hz, one frame is exactly 80 ms
    assert abs(cfg.ms_for_frames(1) - 80.0) < 1e-6
    assert cfg.frames_for_ms(160) == 2   # 2 frames
    assert cfg.frames_for_ms(240) == 3   # 3 frames
    # NOTE: 200 ms = 2.5 frames -> a genuine quantisation gap. The human
    # ~200ms turn gap does NOT land on a frame boundary at 12.5 Hz; your
    # achievable target gaps are quantised to 80 ms steps. Worth a sentence
    # in the dissertation.


def test_encode_decode_shapes():
    bb = MockBackbone()
    cfg = bb.config
    wav = np.zeros((cfg.n_channels, cfg.sample_rate), dtype=np.float32)  # 1 s
    grid = bb.encode(wav)
    assert grid.shape[0] == cfg.n_channels and grid.shape[2] == cfg.n_codebooks
    back = bb.decode(grid)
    assert back.shape[0] == cfg.n_channels


def test_streaming_step_and_lookahead():
    """Roll the backbone forward and 'speculate' a short trajectory."""
    bb = MockBackbone()
    cfg = bb.config
    state = bb.reset()

    horizon_frames = cfg.frames_for_ms(240)  # ~3 frames of lookahead
    pair = np.zeros((cfg.n_channels, cfg.n_codebooks), dtype=np.int64)

    predicted = []
    for _ in range(horizon_frames):
        out, state = bb.step(pair, state)
        assert isinstance(out, StepOutput)
        assert out.hidden.shape == (cfg.hidden_dim,)
        assert out.logits.shape == (cfg.n_channels, cfg.n_codebooks, cfg.codebook_size)
        pair = out.logits.argmax(axis=-1)  # greedy next-pair from backbone
        predicted.append(pair)

    traj = np.stack(predicted, axis=1)  # (n_channels, horizon, n_codebooks)
    assert traj.shape == (cfg.n_channels, horizon_frames, cfg.n_codebooks)

    # The mock's argmax trajectory should match its declared ground truth,
    # i.e. a perfect predictor is achievable -> your head has signal to learn.
    truth = np.stack([bb.true_next_pair(t) for t in range(horizon_frames)], axis=1)
    agree = (traj == truth).mean()
    assert agree > 0.9, f"mock trajectory should be near-perfectly predictable, got {agree:.2f}"


if __name__ == "__main__":
    test_satisfies_protocol()
    test_ms_frame_bridge()
    test_encode_decode_shapes()
    test_streaming_step_and_lookahead()
    print("OK: all smoke tests passed")
    print(MockBackbone().config.summary())
