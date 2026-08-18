"""Backbone-agnostic configuration.

The whole point of this project's "model-agnostic" claim lives here: every
backbone (NTPP, Moshi, Mini-Omni2, the mock) describes itself with the SAME
small struct. Your speculative head and buffer read ONLY this config plus the
hidden states a backbone emits -- never anything model-specific.

CRITICAL DESIGN RULE: reason about lookahead in *milliseconds*, never in
*tokens*. Codecs run at different frame rates (Moshi's Mimi = 12.5 Hz,
NTPP's RVQ is different), so "4 frames ahead" is a different amount of real
time on each model. Define the horizon in ms; convert to frames per backbone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneConfig:
    name: str

    # --- temporal resolution of the audio token stream ---
    frame_rate_hz: float        # acoustic frames emitted per second
    sample_rate: int            # raw audio sample rate (e.g. 24000)

    # --- token grid shape ---
    n_channels: int             # 2 for dual-channel (user + agent)
    n_codebooks: int            # RVQ depth (Mimi=8); 1 for plain VQ
    codebook_size: int          # vocab per codebook (e.g. 4096)

    # --- model internals your head needs to know ---
    hidden_dim: int             # width of the hidden states your head reads
    has_text_stream: bool       # Moshi's inner monologue present? (optional)

    # ---- helpers: the ms <-> frames bridge that keeps you portable ----
    def frames_for_ms(self, ms: float) -> int:
        """How many acoustic frames correspond to `ms` milliseconds."""
        return max(1, round(self.frame_rate_hz * ms / 1000.0))

    def ms_for_frames(self, n_frames: int) -> float:
        """Real time (ms) represented by `n_frames` frames."""
        return 1000.0 * n_frames / self.frame_rate_hz

    def summary(self) -> str:
        txt = " +text" if self.has_text_stream else ""
        return (
            f"[{self.name}] {self.frame_rate_hz:g}Hz  "
            f"{self.n_channels}ch x {self.n_codebooks}cb x {self.codebook_size}v  "
            f"hidden={self.hidden_dim}{txt}  "
            f"(1 frame = {self.ms_for_frames(1):.1f} ms)"
        )
