"""Tests for the inner codebook buffer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec.codebook_buffer import CodebookBuffer, frame_entropy  # noqa: E402


def _peaked_logits(n_cb, card, targets, peak=8.0):
    lg = np.full((n_cb, card), -4.0, dtype=np.float32)
    for k, t in enumerate(targets):
        lg[k, t] = peak
    return lg


def test_assembles_frame_in_order():
    n_cb, card = 8, 2048
    targets = [3, 100, 7, 500, 1, 2047, 42, 9]
    full = _peaked_logits(n_cb, card, targets)
    buf = CodebookBuffer(n_cb, card)
    for k in range(n_cb):
        buf.add(k, full[k], targets[k])
    assert buf.is_complete()
    tokens, logits = buf.assemble()
    assert tokens.tolist() == targets
    assert np.array_equal(logits, full)


def test_rejects_out_of_order():
    buf = CodebookBuffer(4, 16)
    buf.add(0, np.zeros(16, np.float32), 0)
    with pytest.raises(ValueError):
        buf.add(2, np.zeros(16, np.float32), 0)   # skipped 1


def test_incomplete_assemble_raises():
    buf = CodebookBuffer(4, 16)
    buf.add(0, np.zeros(16, np.float32), 0)
    with pytest.raises(RuntimeError):
        buf.assemble()


def test_reset_starts_new_frame():
    buf = CodebookBuffer(2, 16)
    buf.add(0, np.zeros(16, np.float32), 0)
    buf.add(1, np.zeros(16, np.float32), 0)
    assert buf.is_complete()
    buf.reset()
    assert not buf.is_complete()


def test_frame_entropy_low_for_peaked_high_for_flat():
    n_cb, card = 8, 256
    peaked = _peaked_logits(n_cb, card, [0] * n_cb, peak=12.0)
    flat = np.zeros((n_cb, card), dtype=np.float32)
    e_peaked = frame_entropy(peaked)
    e_flat = frame_entropy(flat)
    assert e_peaked < 0.5            # confident frame -> near-zero summed entropy
    assert abs(e_flat - n_cb) < 1e-3  # uniform -> each codebook ~1.0, summed ~n_cb


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and name != "test_rejects_out_of_order":
            try:
                fn()
                print(f"ok  {name}")
            except Exception as e:
                print(f"FAIL {name}: {e}")
    print("(run via pytest for the exception-raising cases)")
