"""Multi-step token-pair prediction head — the project's core trainable module.

From a single frozen-backbone hidden state (`transformer_out`, width H), predict
the next K frames of dual-channel token-pairs IN ONE FORWARD PASS. This is the
learned alternative to autoregressive rollout: rollout needs K sequential
backbone passes (slower, defeats the latency goal), whereas this head emits the
whole K-frame trajectory at once (the Medusa / multi-token-prediction pattern).

Output for one feature: logits of shape
    [B, horizon, n_channels, n_codebooks, codebook_size]
i.e. an independent categorical per (future-frame, channel, codebook). Codebooks
are predicted independently given the hidden state (a v0 simplification; they are
actually RVQ-dependent — the frame-level commit score recombines them jointly).

Requires torch. Kept out of the numpy-only core so the rest of the library still
runs without it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStepTPPHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_channels: int = 2,
        n_codebooks: int = 8,
        codebook_size: int = 2048,
        horizon: int = 4,
        trunk_dim: int = 1024,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.n_pos = n_channels * n_codebooks            # outputs per frame

        self.in_norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, trunk_dim)
        # one residual transform per future step (step-specific "where we look")
        self.step = nn.ModuleList(nn.Linear(trunk_dim, trunk_dim) for _ in range(horizon))
        # one output projection per (channel, codebook), SHARED across steps
        self.out = nn.ModuleList(nn.Linear(trunk_dim, codebook_size) for _ in range(self.n_pos))

    def forward(self, hidden: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """hidden [B, H] -> logits [B, horizon, n_channels, n_codebooks, codebook_size].

        `y` (teacher tokens) is accepted but ignored: codebooks are independent here,
        so the signature just matches MultiStepDepHead for a uniform training call.
        """
        x = F.gelu(self.in_proj(self.in_norm(hidden)))   # [B, trunk]
        steps = []
        for k in range(self.horizon):
            fk = x + F.gelu(self.step[k](x))             # residual step transform
            pos = torch.stack([proj(fk) for proj in self.out], dim=1)  # [B, n_pos, V]
            steps.append(pos)
        logits = torch.stack(steps, dim=1)               # [B, horizon, n_pos, V]
        B = hidden.shape[0]
        return logits.view(B, self.horizon, self.n_channels, self.n_codebooks, self.codebook_size)

"""
class MultiStepDepHead(nn.Module):
    "**"v1 head: codebooks predicted AUTOREGRESSIVELY within a frame (depformer).

    The RVQ codebooks of a frame are not independent — codebook q encodes the
    residual after codebooks <q — so v0's independent prediction discards real
    structure. Here a small causal transformer over the Q codebook positions
    predicts each codebook conditioned on the (teacher-forced) coarser ones,
    sharing one trunk/context per (future-step, channel). Output contract is
    identical to v0: logits [B, K, C, Q, V], so the eval/buffer are unchanged.

    forward(hidden, y):
      - y given  (training): teacher-forced, all Q predicted in ONE pass (causal mask).
      - y is None (inference): greedy left-to-right rollout over the Q codebooks,
        feeding each argmax into the next — returns the same [B,K,C,Q,V] logits.
    "**"

    def __init__(self, hidden_dim: int = 4096, n_channels: int = 2, n_codebooks: int = 8,
                 codebook_size: int = 2048, horizon: int = 4, trunk_dim: int = 1024,
                 dep_dim: int = 512, dep_layers: int = 2, dep_heads: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.bos = codebook_size                          # BOS id (one past the vocab)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, trunk_dim)
        self.step = nn.ModuleList(nn.Linear(trunk_dim, trunk_dim) for _ in range(horizon))
        self.chan = nn.Embedding(n_channels, trunk_dim)   # channel-specific context shift
        self.ctx_proj = nn.Linear(trunk_dim, dep_dim)
        self.tok_emb = nn.Embedding(codebook_size + 1, dep_dim)   # +1 for BOS
        self.pos_emb = nn.Embedding(n_codebooks, dep_dim)
        layer = nn.TransformerEncoderLayer(d_model=dep_dim, nhead=dep_heads,
                                           dim_feedforward=dep_dim * 2, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.dep = nn.TransformerEncoder(layer, num_layers=dep_layers)
        self.out = nn.Linear(dep_dim, codebook_size)
        mask = torch.triu(torch.full((n_codebooks, n_codebooks), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask)

    def _context(self, hidden):
        x = F.gelu(self.in_proj(self.in_norm(hidden)))    # [B, trunk]
        steps = []
        for k in range(self.horizon):
            fk = x + F.gelu(self.step[k](x))              # [B, trunk]
            steps.append(fk[:, None, :] + self.chan.weight[None, :, :])   # [B, C, trunk]
        ctx = torch.stack(steps, dim=1)                   # [B, K, C, trunk]
        return self.ctx_proj(ctx)                         # [B, K, C, dep]

    def forward(self, hidden: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        ctx = self._context(hidden)                       # [B,K,C,D]
        B, K, C, D = ctx.shape
        Q, V = self.n_codebooks, self.codebook_size
        M = B * K * C
        ctx = ctx.reshape(M, D)
        pos = self.pos_emb(torch.arange(Q, device=hidden.device))         # [Q,D]
        if y is not None:                                 # teacher-forced, one pass
            yt = y.reshape(M, Q)
            bos = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
            inp = torch.cat([bos, yt[:, :-1]], dim=1)     # shift-right: predicts q from <q
            seq = self.tok_emb(inp) + pos[None] + ctx[:, None, :]          # [M,Q,D]
            h = self.dep(seq, mask=self.causal)
            return self.out(h).view(B, K, C, Q, V)
        cur = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
        logits = []                                       # greedy rollout
        for q in range(Q):
            seq = self.tok_emb(cur) + pos[None, :q + 1, :] + ctx[:, None, :]
            h = self.dep(seq, mask=self.causal[:q + 1, :q + 1])
            lg = self.out(h[:, -1])                       # [M,V]
            logits.append(lg)
            cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], dim=1)
        return torch.stack(logits, dim=1).view(B, K, C, Q, V)

"""
class MultiStepDepHead(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 4096, 
        n_channels: int = 2, 
        n_codebooks: int = 8,
        codebook_size: int = 2048, 
        horizon: int = 4, 
        trunk_dim: int = 2048,   # INCREASED from 1024
        dep_dim: int = 1024,     # INCREASED from 512
        dep_layers: int = 4,     # INCREASED from 2
        dep_heads: int = 8,      # INCREASED from 4
        dropout: float = 0.1     # ADDED regularization
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.bos = codebook_size                          
        
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, trunk_dim)
        
        # ADDED: Explicit conditioning on the current frame (y0)
        self.y0_emb = nn.Embedding(codebook_size, trunk_dim)
        
        self.step = nn.ModuleList(nn.Linear(trunk_dim, trunk_dim) for _ in range(horizon))
        self.chan = nn.Embedding(n_channels, trunk_dim)   
        self.ctx_proj = nn.Linear(trunk_dim, dep_dim)
        
        self.tok_emb = nn.Embedding(codebook_size + 1, dep_dim)   
        self.pos_emb = nn.Embedding(n_codebooks, dep_dim)
        
        self.drop = nn.Dropout(dropout) # ADDED
        
        # INCREASED capacity and added dropout to the transformer
        layer = nn.TransformerEncoderLayer(
            d_model=dep_dim, 
            nhead=dep_heads,
            dim_feedforward=dep_dim * 2, 
            batch_first=True,
            activation="gelu", 
            dropout=dropout
        )
        self.dep = nn.TransformerEncoder(layer, num_layers=dep_layers)
        self.out = nn.Linear(dep_dim, codebook_size)
        
        mask = torch.triu(torch.full((n_codebooks, n_codebooks), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask)

    def _context(self, hidden, y0=None):
        x = F.gelu(self.in_proj(self.in_norm(hidden)))    # [B, trunk]
        
        # ADDED: Inject y0 knowledge if provided. 
        # Sums the embeddings of the current frame's tokens across channels and codebooks.
        if y0 is not None:
            # y0 shape: [B, C, Q]. Clamp to handle potential padding/out-of-bounds
            y0_clamped = torch.clamp(y0, 0, self.codebook_size - 1)
            y0_feats = self.y0_emb(y0_clamped).sum(dim=(1, 2)) # [B, trunk]
            x = x + y0_feats
            
        x = self.drop(x)
        
        steps = []
        for k in range(self.horizon):
            fk = x + self.drop(F.gelu(self.step[k](x)))   # [B, trunk]
            steps.append(fk[:, None, :] + self.chan.weight[None, :, :])   # [B, C, trunk]
        ctx = torch.stack(steps, dim=1)                   # [B, K, C, trunk]
        return self.ctx_proj(ctx)                         # [B, K, C, dep]

    # ADDED y0 to the signature
    def forward(self, hidden: torch.Tensor, y0: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        ctx = self._context(hidden, y0)                   # [B,K,C,D]
        B, K, C, D = ctx.shape
        Q, V = self.n_codebooks, self.codebook_size
        M = B * K * C
        ctx = ctx.reshape(M, D)
        pos = self.pos_emb(torch.arange(Q, device=hidden.device))         # [Q,D]
        
        if y is not None:                                 # teacher-forced, one pass
            yt = y.reshape(M, Q)
            bos = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
            inp = torch.cat([bos, yt[:, :-1]], dim=1)     
            seq = self.tok_emb(inp) + pos[None] + ctx[:, None, :]          
            h = self.dep(seq, mask=self.causal)
            return self.out(h).view(B, K, C, Q, V)
            
        cur = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
        logits = []                                       # greedy rollout
        for q in range(Q):
            seq = self.tok_emb(cur) + pos[None, :q + 1, :] + ctx[:, None, :]
            h = self.dep(seq, mask=self.causal[:q + 1, :q + 1])
            lg = self.out(h[:, -1])                       # [M,V]
            logits.append(lg)
            cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], dim=1)
        return torch.stack(logits, dim=1).view(B, K, C, Q, V)


def tpp_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over all (frame, channel, codebook) positions.

    logits:  [B, K, C, Q, V]
    targets: [B, C, Q, K]  (Stage-A future_targets layout: channels, codebooks, time)
    """
    B, K, C, Q, V = logits.shape
    tgt = targets.permute(0, 3, 1, 2).contiguous()       # [B, K, C, Q]
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))


# --- integration: trained head as a one-pass trajectory predictor ---
import numpy as np  # noqa: E402
from .backbone import StepOutput  # noqa: E402


class HeadPredictor:
    def __init__(self, head: "MultiStepTPPHead", device: str = "cpu"):
        self.head = head.to(device).eval()
        self.device = device

    def predict(self, last_pair, state, horizon):
        h = np.asarray(state["hidden"], dtype=np.float32)
        
        # ADDED: Extract y0 from last_pair for inference. 
        # Assumes last_pair can be cast to [C, Q]. Adjust if last_pair structure differs in your buffer.
        y0_tensor = None
        if last_pair is not None:
            y0_np = np.asarray(last_pair, dtype=np.int64)
            y0_tensor = torch.from_numpy(y0_np)[None].to(self.device) # [1, C, Q]

        with torch.no_grad():
            # Pass y0 to the head
            lo = self.head(
                hidden=torch.from_numpy(h)[None].to(self.device),
                y0=y0_tensor
            )  
        lo = lo[0].cpu().numpy()                                        
        K = min(horizon, lo.shape[0])
        return [StepOutput(hidden=h, logits=lo[k]) for k in range(K)]
"""
class HeadPredictor:
    "**"Adapt a trained MultiStepTPPHead to the SpeculativeBuffer's predictor API.

    Unlike BackboneRolloutPredictor (K sequential backbone passes), this emits the
    whole K-frame trajectory in ONE forward pass — the latency argument of the
    project. It reads the current hidden state from state['hidden'] (a cached
    transformer_out vector) and ignores last_pair.

    Per-frame StepOutput.logits has shape [n_channels, n_codebooks, codebook_size],
    matching what the mock produced, so the existing commit strategies and buffer
    work unchanged.
    "**"

    def __init__(self, head: "MultiStepTPPHead", device: str = "cpu"):
        self.head = head.to(device).eval()
        self.device = device

    def predict(self, last_pair, state, horizon):
        h = np.asarray(state["hidden"], dtype=np.float32)
        with torch.no_grad():
            lo = self.head(torch.from_numpy(h)[None].to(self.device))  # [1,K,C,Q,V]
        lo = lo[0].cpu().numpy()                                        # [K,C,Q,V]
        K = min(horizon, lo.shape[0])
        return [StepOutput(hidden=h, logits=lo[k]) for k in range(K)]
        
 """
