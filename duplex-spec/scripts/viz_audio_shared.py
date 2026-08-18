"""Visualise a STEREO audio file as two speaker lanes (user = left, Moshi = right).

Point it at a generated .wav (stereo, as audio_demo_triggers.py now writes) and it draws the
two-channel waveform timeline --- every place each speaker is active, overlaps included. No
tokens, no labels; just the audio.

Usage:
    python scripts/viz_audio.py --audio demo_trigger_audio/clip00_h5617_js.wav \
        --names "User,GPT-Live" --out demo_viz/clip00_js.svg

    # compare several conditions stacked (e.g. oracle vs js vs reactive):
    python scripts/viz_audio.py --audio a_oracle.wav a_js.wav a_reactive.wav \
        --row-labels oracle,js,reactive --out demo_viz/compare.svg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def envelope(x, n_bins):
    if len(x) == 0:
        return np.zeros(n_bins)
    step = max(1, len(x) // n_bins)
    env = np.array([np.abs(x[i:i+step]).max() for i in range(0, step*n_bins, step)][:n_bins])
    if len(env) < n_bins:
        env = np.pad(env, (0, n_bins - len(env)))
    return env


def load_stereo(path):
    import soundfile as sf
    wav, sr = sf.read(path, always_2d=True)          # [samples, ch]
    if wav.shape[1] == 1:                             # mono -> duplicate so it still renders
        wav = np.repeat(wav, 2, axis=1)
    return wav.astype(np.float32), sr


def build_svg(channel_envs, names, dur_ms, title=None):
    """channel_envs: list of 1D arrays (one per lane). Returns SVG string."""
    C = len(channel_envs)
    n_bins = len(channel_envs[0])
    W, laneH, gap, padL, padT, padB, padR = 1200, 110, 26, 100, 40, 44, 20
    H = padT + C * laneH + (C - 1) * gap + padB
    innerW = W - padL - padR
    bw = innerW / n_bins
    colors = ["#5FA84E", "#3F7FE0", "#C9761F", "#8250C0"]
    bgs = ["#EAF4E2", "#E8F0FE", "#FBEEDF", "#F1EBFA"]
    gmax = max(1e-6, max(e.max() for e in channel_envs))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    if title:
        svg.append(f'<text x="{padL}" y="20" font-size="13" fill="#444">{title}</text>')

    for c in range(C):
        y0 = padT + c * (laneH + gap); mid = y0 + laneH / 2
        col = colors[c % len(colors)]; bg = bgs[c % len(bgs)]
        svg.append(f'<rect x="{padL-8}" y="{y0}" width="{innerW+16}" height="{laneH}" '
                   f'rx="12" fill="{bg}"/>')
        svg.append(f'<text x="14" y="{mid+4}" font-size="14" fill="#333">{names[c]}</text>')
        svg.append(f'<line x1="{padL}" y1="{mid:.1f}" x2="{padL+innerW}" y2="{mid:.1f}" '
                   f'stroke="#ccc" stroke-width="0.5"/>')
        env = channel_envs[c] / gmax
        for i in range(n_bins):
            a = env[i]
            if a <= 0.01:
                continue
            h = a * (laneH * 0.44)
            xx = padL + i * bw
            svg.append(f'<rect x="{xx:.2f}" y="{mid-h:.2f}" width="{max(bw*0.7,0.5):.2f}" '
                       f'height="{2*h:.2f}" rx="0.6" fill="{col}"/>')

    for k in range(0, 9):
        frac = k / 8; xx = padL + frac * innerW; ms = int(dur_ms * frac)
        svg.append(f'<text x="{xx:.1f}" y="{H-14}" font-size="10" fill="#888" '
                   f'text-anchor="middle">{ms}ms</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True, help="stereo wav(s)")
    ap.add_argument("--names", default="User,GPT-Live", help="two channel labels")
    ap.add_argument("--row-labels", default="", help="if multiple audios, a label per file")
    ap.add_argument("--shared-user", action="store_true",
                    help="show the user lane ONCE (from first file), then each file's Moshi lane")
    ap.add_argument("--bins-per-sec", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=Path("demo_viz/audio.svg"))
    args = ap.parse_args()

    names = args.names.split(",")
    lanes, lane_names = [], []
    dur_ms = 0
    row_labels = args.row_labels.split(",") if args.row_labels else None

    if args.shared_user:
        # user lane once (from first file, left channel), then each file's Moshi (right channel)
        for ai, ap_ in enumerate(args.audio):
            wav, sr = load_stereo(ap_)
            dur_ms = max(dur_ms, int(1000 * len(wav) / sr))
            n_bins = int(len(wav) / sr * args.bins_per_sec)
            if ai == 0:
                lanes.append(envelope(wav[:, 0], n_bins)); lane_names.append(names[0])
            lbl = row_labels[ai] if row_labels and ai < len(row_labels) else f"cond{ai}"
            lanes.append(envelope(wav[:, 1], n_bins)); lane_names.append(lbl)
    else:
        for ai, ap_ in enumerate(args.audio):
            wav, sr = load_stereo(ap_)
            dur_ms = max(dur_ms, int(1000 * len(wav) / sr))
            n_bins = int(len(wav) / sr * args.bins_per_sec)
            prefix = f"{row_labels[ai]}: " if row_labels and ai < len(row_labels) else ""
            for ch in range(2):
                lanes.append(envelope(wav[:, ch], n_bins))
                lane_names.append(f"{prefix}{names[ch]}")

    # pad all lanes to same bin count
    m = max(len(l) for l in lanes)
    lanes = [np.pad(l, (0, m - len(l))) for l in lanes]

    svg = build_svg(lanes, lane_names, dur_ms,
                    title=(None if len(args.audio) == 1 else "comparison"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"[svg] {args.out}  ({len(lanes)} lanes, {dur_ms}ms)")
    try:
        import cairosvg
        png = args.out.with_suffix(".png")
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=2)
        print(f"[png] {png}")
    except Exception:
        print("  (pip install cairosvg --break-system-packages  for PNG)")


if __name__ == "__main__":
    main()
