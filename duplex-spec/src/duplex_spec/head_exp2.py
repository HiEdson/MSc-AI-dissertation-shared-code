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
        trunk_dim: int = 2048,   # WIDENED capacity
        dropout: float = 0.1     # ADDED regularization
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.n_pos = n_channels * n_codebooks            

        self.in_norm = nn.LayerNorm(hidden_dim)
        
        # HALVED the projection to make room for y0 concatenation
        self.in_proj = nn.Linear(hidden_dim, trunk_dim // 2)
        
        # ADDED: Explicit conditioning on the current frame (y0)
        self.y0_emb = nn.Embedding(codebook_size, trunk_dim // 2)
        self.drop = nn.Dropout(dropout)
        
        self.step = nn.ModuleList(nn.Linear(trunk_dim, trunk_dim) for _ in range(horizon))
        self.out = nn.ModuleList(nn.Linear(trunk_dim, codebook_size) for _ in range(self.n_pos))

    # UPDATED signature to accept y0 and y
    def forward(self, hidden: torch.Tensor, y0: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        """
        `y` is accepted but ignored to maintain a uniform training call. 
        `y0` is actively concatenated to inject the current discrete frame state.
        """
        x = F.gelu(self.in_proj(self.in_norm(hidden)))   # [B, trunk // 2]
        
        # EXPERIMENT 1: Inject y0 via Concatenation
        if y0 is not None:
            y0_clamped = torch.clamp(y0, 0, self.codebook_size - 1)
            y0_feats = self.y0_emb(y0_clamped).sum(dim=(1, 2)) # [B, trunk // 2]
            x = torch.cat([x, y0_feats], dim=-1)               # [B, trunk]
        else:
            # Fallback for greedy eval if y0 is totally missing
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)
            
        x = self.drop(x)
        
        steps = []
        for k in range(self.horizon):
            fk = x + self.drop(F.gelu(self.step[k](x)))             
            pos = torch.stack([proj(fk) for proj in self.out], dim=1)  # [B, n_pos, V]
            steps.append(pos)
            
        logits = torch.stack(steps, dim=1)               # [B, horizon, n_pos, V]
        B = hidden.shape[0]
        return logits.view(B, self.horizon, self.n_channels, self.n_codebooks, self.codebook_size)
        
        
class MultiStepCascadedMLPHead(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 4096, 
        n_channels: int = 2, 
        n_codebooks: int = 8,
        codebook_size: int = 2048, 
        horizon: int = 4, 
        trunk_dim: int = 2048,   
        dep_dim: int = 512,      # Kept small for blazing fast execution
        dropout: float = 0.1     
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.horizon = horizon
        self.bos = codebook_size                          
        
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, trunk_dim // 2)
        
        # y0 conditioning (The fix from earlier is baked in)
        self.y0_emb = nn.Embedding(codebook_size, trunk_dim // 2)
        self.post_cat_norm = nn.LayerNorm(trunk_dim)
        
        self.step = nn.ModuleList(nn.Linear(trunk_dim, trunk_dim) for _ in range(horizon))
        self.chan = nn.Embedding(n_channels, trunk_dim)   
        self.ctx_proj = nn.Linear(trunk_dim, dep_dim)
        
        self.tok_emb = nn.Embedding(codebook_size + 1, dep_dim)   
        self.pos_emb = nn.Embedding(n_codebooks, dep_dim)
        self.drop = nn.Dropout(dropout)
        
        # THE MAGIC: A tiny MLP block replaces the heavy Transformer
        self.q_cell = nn.Sequential(
            nn.Linear(dep_dim, dep_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dep_dim, dep_dim)
        )
        self.out = nn.Linear(dep_dim, codebook_size)

    def _context(self, hidden, y0=None):
        x = F.gelu(self.in_proj(self.in_norm(hidden)))    
        
        if y0 is not None:
            y0_clamped = torch.clamp(y0, 0, self.codebook_size - 1)
            y0_feats = self.y0_emb(y0_clamped).mean(dim=(1, 2)) 
            x = torch.cat([x, y0_feats], dim=-1)               
        else:
            x = torch.cat([x, torch.zeros_like(x)], dim=-1)
            
        x = self.post_cat_norm(x)
        x = self.drop(x)
        
        steps = []
        for k in range(self.horizon):
            fk = x + self.drop(F.gelu(self.step[k](x)))   
            steps.append(fk[:, None, :] + self.chan.weight[None, :, :])   
        ctx = torch.stack(steps, dim=1)                   
        return self.ctx_proj(ctx)                         

    def forward(self, hidden: torch.Tensor, y0: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        ctx = self._context(hidden, y0)                   # [B,K,C,D]
        B, K, C, D = ctx.shape
        Q, V = self.n_codebooks, self.codebook_size
        M = B * K * C
        ctx = ctx.reshape(M, D)
        pos = self.pos_emb(torch.arange(Q, device=hidden.device))         
        
        # TRAINING: Fast Parallel Teacher Forcing
        if y is not None:                                 
            yt = y.reshape(M, Q)
            bos = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
            inp = torch.cat([bos, yt[:, :-1]], dim=1)     
            
            # Sequence of states [M, Q, D]
            seq = self.tok_emb(inp) + pos[None] + ctx[:, None, :]          
            
            # Apply the MLP cell across the codebook dimension
            h = seq + self.q_cell(seq)
            return self.out(h).view(B, K, C, Q, V)
            
        # INFERENCE: Blazing Fast Sequential Rollout
        cur = torch.full((M, 1), self.bos, dtype=torch.long, device=hidden.device)
        logits = []                                       
        for q in range(Q):
            # Embed the previous codebook's prediction
            seq_step = self.tok_emb(cur[:, -1:]) + pos[None, q:q+1, :] + ctx[:, None, :]
            
            # Step the MLP
            h_step = seq_step + self.q_cell(seq_step)
            
            # Predict this codebook
            lg = self.out(h_step[:, -1])                       
            logits.append(lg)
            
            # Append the prediction to feed into the next codebook step
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

