"""Tests for the multi-step head. Skipped automatically if torch is absent."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from duplex_spec.head import MultiStepTPPHead, tpp_loss  # noqa: E402


def _small_head():
    # tiny config so tests are fast
    return MultiStepTPPHead(hidden_dim=32, n_channels=2, n_codebooks=3,
                            codebook_size=16, horizon=2, trunk_dim=24)


def test_forward_shape():
    head = _small_head()
    h = torch.randn(5, 32)
    out = head(h)
    assert out.shape == (5, 2, 2, 3, 16)        # [B, K, C, Q, V]


def test_loss_is_scalar_and_finite():
    head = _small_head()
    h = torch.randn(4, 32)
    tgt = torch.randint(0, 16, (4, 2, 3, 2))    # [B, C, Q, K]
    loss = tpp_loss(head(h), tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_head_can_learn_a_fixed_mapping():
    """Overfit a tiny fixed (feature -> future tokens) set: loss must drop a lot.

    This is the real check that the architecture + loss can learn at all.
    """
    torch.manual_seed(0)
    head = _small_head()
    N = 8
    H = torch.randn(N, 32)
    tgt = torch.randint(0, 16, (N, 2, 3, 2))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    first = tpp_loss(head(H), tgt).item()
    for _ in range(300):
        opt.zero_grad()
        loss = tpp_loss(head(H), tgt)
        loss.backward()
        opt.step()
    last = loss.item()

    assert last < first * 0.2, f"head failed to overfit: {first:.3f} -> {last:.3f}"
    # and it should predict the memorised tokens correctly
    pred = head(H).argmax(-1)                    # [N, K, C, Q]
    acc = (pred == tgt.permute(0, 3, 1, 2)).float().mean().item()
    assert acc > 0.9, f"memorisation accuracy too low: {acc:.2f}"


if __name__ == "__main__":
    test_forward_shape()
    test_loss_is_scalar_and_finite()
    test_head_can_learn_a_fixed_mapping()
    print("head tests passed")


def test_dep_head_teacher_forced_matches_greedy():
    import torch
    from duplex_spec.head import MultiStepDepHead
    torch.manual_seed(0)
    h = MultiStepDepHead(hidden_dim=64, n_channels=2, n_codebooks=8, codebook_size=32,
                         horizon=3, trunk_dim=48, dep_dim=32, dep_layers=2, dep_heads=4).eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        greedy = h(x)                      # [B,K,C,Q,V] greedy rollout
        toks = greedy.argmax(-1)           # the tokens greedy actually chose
        forced = h(x, toks)                # teacher-force with those same tokens
    # feeding greedy's own argmax back as teacher tokens must reproduce its logits
    assert torch.allclose(greedy, forced, atol=1e-4)
    assert greedy.shape == (4, 3, 2, 8, 32)
