"""Horizon-autoregressive head (v3) --- conditions each future frame on the PREVIOUS
predicted frame, addressing the cross-horizon conditional-independence approximation.

MOTIVATION
----------
v0 (MultiStepTPPHead) predicts every horizon in parallel from h_t:
    p(x_{t+1:t+K} | h_t) = prod_k p(x_{t+k} | h_t)
so frame t+2's prediction never sees t+1. Real speech is autoregressive --- t+2 depends
on what was actually said at t+1 --- and this approximation blurs the far-horizon
distributions the JS gate and probe read. v3 keeps the SINGLE frozen-backbone pass (the
latency argument stands: only ONE h_t is computed) but replaces the parallel per-step
projections with a small recurrent module over the K horizons, so each step's context is
conditioned on the previous step's PREDICTED frame.

    context_1 = f(h_t)
    context_k = g(context_{k-1}, embed(pred_{k-1}))     for k > 1

At training time pred_{k-1} is the TEACHER frame (teacher forcing over horizons); at
inference it is the previous step's argmax (K tiny recurrent steps, not K backbone passes
--- negligible latency). Output contract is identical to v0: logits [B,K,C,Q,V], and
tpp_loss is unchanged, so the JS gate / probe / buffer all work without modification.

Codebooks remain independent within a frame (as v0); this isolates the CROSS-HORIZON
effect. A variant conditioning both across horizons and across codebooks is possible but
would confound the two.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonARHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, n_channels: int = 2, n_codebooks: int = 8,
                 codebook_size: int = 2048, horizon: int = 4, trunk_dim: int = 1024,
                 frame_emb_dim: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.n_pos = n_channels * n_codebooks

        self.in_norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, trunk_dim)

        # recurrent cell over horizons: next context from (prev context, prev frame embedding)
        self.cell = nn.GRUCell(frame_emb_dim, trunk_dim)
        # embed the previous predicted frame: mean of per-(channel,codebook) token embeddings
        self.tok_emb = nn.Embedding(codebook_size + 1, frame_emb_dim)   # +1 = BOS/no-history
        self.bos = codebook_size
        self.frame_norm = nn.LayerNorm(frame_emb_dim)

        # shared output projections per (channel, codebook), as in v0
        self.out = nn.ModuleList(nn.Linear(trunk_dim, codebook_size) for _ in range(self.n_pos))

    def _embed_frame(self, frame_ids):
        """frame_ids [B, n_pos] int -> [B, frame_emb_dim] (mean of token embeddings)."""
        e = self.tok_emb(frame_ids)                      # [B, n_pos, emb]
        return self.frame_norm(e.mean(dim=1))            # [B, emb]

    def _project(self, ctx):
        """ctx [B, trunk] -> logits [B, n_channels, n_codebooks, V]."""
        pos = torch.stack([proj(ctx) for proj in self.out], dim=1)      # [B, n_pos, V]
        B = ctx.shape[0]
        return pos.view(B, self.n_channels, self.n_codebooks, self.codebook_size)

    def forward(self, hidden: torch.Tensor, y: torch.Tensor = None,
                tf_prob: float = 1.0) -> torch.Tensor:
        """hidden [B,H] -> logits [B,K,C,Q,V].

        y (teacher targets, layout [B, K, C, Q]) enables teacher forcing over horizons.
        tf_prob: scheduled sampling. At each step (k>0) the frame fed forward is the
          TEACHER frame with probability tf_prob, else the head's OWN greedy prediction
          (detached). tf_prob=1.0 -> pure teacher forcing (old behaviour); tf_prob<1.0
          exposes the head to its own errors during training to fight exposure bias.
          Ignored when y is None (inference: always greedy).
        """
        B = hidden.shape[0]
        dev = hidden.device
        ctx = F.gelu(self.in_proj(self.in_norm(hidden)))                 # [B, trunk] = context_1
        # first frame has no predecessor -> BOS frame embedding
        prev_ids = torch.full((B, self.n_pos), self.bos, dtype=torch.long, device=dev)

        # teacher frames, if provided: y [B,C,Q,K] -> per-step [B, n_pos]
        teacher = None
        if y is not None:
            # reorder to [B, K, C, Q] then flatten (C,Q) -> n_pos
            teacher = y.permute(0, 3, 1, 2).reshape(B, self.horizon, self.n_pos)

        steps = []
        for k in range(self.horizon):
            if k > 0:
                fe = self._embed_frame(prev_ids)                         # [B, emb]
                ctx = self.cell(fe, ctx)                                 # condition on prev frame
            logit_k = self._project(ctx)                                 # [B,C,Q,V]
            steps.append(logit_k)
            # choose the frame to feed forward (scheduled sampling)
            if teacher is not None:
                greedy_ids = logit_k.reshape(B, self.n_pos, -1).argmax(-1).detach()
                if tf_prob >= 1.0:
                    prev_ids = teacher[:, k, :]
                else:
                    # per-SAMPLE coin flip: teacher row w.p. tf_prob else own greedy row
                    use_teacher = (torch.rand(B, 1, device=dev) < tf_prob)
                    prev_ids = torch.where(use_teacher, teacher[:, k, :], greedy_ids)
            else:
                prev_ids = logit_k.reshape(B, self.n_pos, -1).argmax(-1)  # inference: greedy

        return torch.stack(steps, dim=1)                                 # [B,K,C,Q,V]
