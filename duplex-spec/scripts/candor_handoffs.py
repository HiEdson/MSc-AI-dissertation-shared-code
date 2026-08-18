"""Build real handoff labels from CANDOR transcripts (backbiter turn model), aligned to
the 80 ms token frames. Replaces the silence-heuristic labels, whose 1.1% prevalence and
noise capped the predictor at AUC 0.69.

A handoff = a turn boundary where the FLOOR changes speaker: turn i ends (speaker A) and
turn i+1 is spoken by speaker B (A != B). We take the moment of transition as the START of
turn i+1 (when B begins). Backchannels are already excluded by the backbiter model, so
"mhm"/"yeah" do not count as taking the floor --- exactly the false positives that poisoned
the heuristic.

Alignment
---------
manifest.jsonl maps conv_id -> speaker_L, speaker_R, token_path, n_frames, seconds.
frame = round(t_seconds * n_frames / seconds)  (== round(t*12.5), but derived per-conv so
it is exact even if the rate drifts). Transcript `speaker` is matched to channel 0 (L) or
1 (R) via the manifest.

Output per conversation: handoff_frames (int array) + the channel that TAKES the floor.
These drive handoff_predictor.py in --label-mode transcript.

Usage:
    python scripts/candor_handoffs.py \
        --manifest tokens/manifest.jsonl \
        --candor-root "/home/ec25045/Documents/thesis/Candor dataset" \
        --turn-model backbiter --out handoff_labels/
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np


def load_manifest(path):
    convs = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            convs[d["conv_id"]] = d
    return convs


def read_turns(csv_path):
    """Return list of (speaker, start, stop) for real turns, in order."""
    turns = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                turns.append((row["speaker"], float(row["start"]), float(row["stop"])))
            except (KeyError, ValueError):
                continue
    return turns


def handoffs_for_conv(turns, spk_L, spk_R):
    """Yield (handoff_frame_time_seconds, taker_channel) for floor changes.
    taker_channel: 0 if speaker_L takes the floor, 1 if speaker_R."""
    out = []
    for i in range(1, len(turns)):
        prev_spk = turns[i - 1][0]
        cur_spk, cur_start, _ = turns[i]
        if cur_spk != prev_spk:                       # floor changed hands
            if cur_spk == spk_L:
                taker = 0
            elif cur_spk == spk_R:
                taker = 1
            else:
                continue                              # unknown speaker id; skip
            out.append((cur_start, taker))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--candor-root", type=Path, required=True)
    ap.add_argument("--turn-model", default="backbiter",
                    choices=["backbiter", "audiophile", "cliffhanger"])
    ap.add_argument("--out", type=Path, default=Path("handoff_labels"))
    args = ap.parse_args()

    convs = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    n_ok = n_missing = 0
    total_ho = 0
    summary = {}

    for conv_id, meta in convs.items():
        csv_path = (args.candor_root / conv_id / "transcription" /
                    f"transcript_{args.turn_model}.csv")
        if not csv_path.exists():
            n_missing += 1
            continue
        turns = read_turns(csv_path)
        if not turns:
            n_missing += 1
            continue
        n_frames = int(meta["n_frames"]); seconds = float(meta["seconds"])
        rate = n_frames / seconds                     # frames per second (~12.5)
        hos = handoffs_for_conv(turns, meta["speaker_L"], meta["speaker_R"])
        # to frame indices, clipped to valid range
        ho_frame = []; ho_taker = []
        for t, taker in hos:
            fr = int(round(t * rate))
            if 0 <= fr < n_frames:
                ho_frame.append(fr); ho_taker.append(taker)
        ho_frame = np.array(ho_frame, np.int64); ho_taker = np.array(ho_taker, np.int8)
        np.savez(args.out / f"{conv_id}.npz",
                 handoff_frames=ho_frame, taker_channel=ho_taker,
                 n_frames=n_frames, rate=rate)
        n_ok += 1; total_ho += len(ho_frame)
        summary[conv_id] = {"handoffs": int(len(ho_frame)), "n_frames": n_frames,
                            "per_min": len(ho_frame) / (seconds / 60.0)}

    print(f"[done] {n_ok} conversations labelled, {n_missing} missing transcripts")
    if n_ok:
        pm = np.mean([s["per_min"] for s in summary.values()])
        print(f"[stats] {total_ho} handoffs total, {pm:.1f} handoffs/min average")
        print(f"        (silence-heuristic gave 1.1% frame prevalence; transcript should")
        print(f"         be cleaner and better-timed)")
    (args.out / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[out] labels + _summary.json in {args.out}/")


if __name__ == "__main__":
    main()
