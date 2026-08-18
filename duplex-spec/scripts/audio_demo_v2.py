"""Gate-triggered AUDIO demo --- drives Moshi using the JS gate or the probe as the release
trigger, with the ORACLE as a side-by-side reference, so the timing can be HEARD.

The harness (run_demo.py) showed the gates fire on ~13k confident frames anywhere, mostly
mid-speech. A raw "release on any confident frame" trigger is a firehose, not turn-taking.
So here the trigger is ARMED: it may only release when Moshi is currently LISTENING (silent)
and the user is winding down; once it releases it DISARMS until the next silence --- one
release per turn. That is the honest realisation of "use the gate as a handoff trigger".

For each handoff clip we render up to three conditions on the SAME audio:
    oracle : release `lead` frames before the true handoff (ideal timing, reference)
    js     : release when the JS gate goes confident while armed
    probe  : release when the probe score crosses threshold while armed

Expected honest outcome: js/probe coverage is low (13%/7%), so they will MISS some turns ---
Moshi stays silent where oracle speaks. That hesitancy is the informative result: it shows
the gate is conservative as a trigger, motivating a hybrid. Listen for: does the gate release
NEAR the oracle moment, late, or not at all?

Usage:
    PYTHONPATH=src python scripts/audio_demo_triggers.py \
        --tokens tokens_eval/<conv>.npy --feats pairs_eval/<conv>.npz \
        --labels handoff_labels_eval/<conv>.npz --head head_v0.pt \
        --triggers oracle,js,probe --probe probe_top5.npz \
        --js-tau 0.2 --probe-thr 0.8 --lead 3 \
        --n-clips 4 --clip-frames 80 --device cuda --out demo_trigger_audio/
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_triggers as dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--feats", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--probe", type=Path)
    ap.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-q8")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--triggers", default="oracle,js,probe")
    ap.add_argument("--js-tau", type=float, default=0.2)
    ap.add_argument("--probe-thr", type=float, default=0.8)
    ap.add_argument("--ent-floor", type=float, default=0.5)
    ap.add_argument("--entropy-thr", type=float, default=0.6,
                    help="entropy trigger: release when norm-entropy < this (both channels)")
    ap.add_argument("--reactive-lag", type=int, default=6,
                    help="reactive baseline releases this many frames AFTER the handoff")
    ap.add_argument("--lead", type=int, default=3)
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--clip-frames", type=int, default=80)
    ap.add_argument("--arm-silence", type=int, default=3,
                    help="frames of system silence required to re-arm the trigger")
    ap.add_argument("--out", type=Path, default=Path("demo_trigger_audio"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true",
                    help="deterministic generation (use_sampling=False) so reruns are identical")
    ap.add_argument("--temp", type=float, default=0.8,
                    help="Moshi sampling temperature (ignored if --greedy)")
    ap.add_argument("--prime", type=int, default=8,
                    help="frames of REAL-audio warm-up before each clip (A)")
    ap.add_argument("--start", default=None,
                    help="explicit clip start frame(s), comma-separated (from find_active). "
                         "If omitted, clips are picked around handoff labels.")
    ap.add_argument("--vad-db", type=float, default=-40,
                    help="energy VAD threshold (dB) for real-speech arming")
    ap.add_argument("--min-talk", type=int, default=12,
                    help="once released, keep speaking at least this many frames (anti-stutter)")
    ap.add_argument("--soft-gate", action="store_true",
                    help="B: keep Moshi running during gated frames (discard its audio) so it "
                         "stays responsive on release, instead of hard-forcing silence")
    args = ap.parse_args()

    import torch
    import soundfile as sf
    from moshi.models import LMGen, loaders
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from duplex_spec.head import MultiStepTPPHead, MultiStepDepHead

    # --- head (for gate decisions) ---
    ck = torch.load(args.head, map_location=args.device)
    K = ck["horizon"]
    Head = MultiStepDepHead if ck.get("head_type") == "dep" else MultiStepTPPHead
    head = Head(hidden_dim=ck["hidden_dim"], n_channels=ck["n_channels"],
                n_codebooks=ck["n_codebooks"], codebook_size=2048, horizon=K)
    head.load_state_dict(ck["state_dict"]); head.to(args.device).eval()
    logV = float(np.log(2048.0))

    # --- Moshi + Mimi ---
    print("[load] Moshi + Mimi ...")
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    lm = ckpt.get_moshi(device=args.device, dtype=getattr(torch, args.dtype))
    mimi = ckpt.get_mimi(device=args.device)
    try:
        from moshi.utils.quantize import QLinear
        for m in lm.modules():
            if isinstance(m, QLinear):
                m.weight_scb.data = m.weight_scb.data.float()
    except Exception:
        pass

    tokens = np.load(args.tokens); C, Q, T = tokens.shape
    d = np.load(args.feats); feats, frames = d["feats"], d["frames"]
    ho = np.sort(np.load(args.labels)["handoff_frames"].astype(int))
    ho = ho[(ho > args.clip_frames) & (ho < T - args.clip_frames)]

    # --- ENERGY-based speech activity per channel (full-duplex channels always carry signal,
    #     so the token-modal test over-counts; energy on decoded audio gives true speech). ---
    sr = mimi.sample_rate
    spf = int(sr / 12.5)
    def speech_flags(ch, chunk=1500):
        import torch as _t
        flags = []
        with _t.no_grad():
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                seg = _t.from_numpy(tokens[ch:ch+1, :, s:e]).to(args.device).long()
                wav = mimi.decode(seg).squeeze().cpu().numpy()
                nf = len(wav) // spf
                fr = wav[:nf*spf].reshape(nf, spf)
                db = 20*np.log10(np.sqrt((fr**2).mean(1)+1e-12)+1e-12)
                flags.append((db > args.vad_db).astype(np.float32))
                del seg
                if args.device == "cuda": _t.cuda.empty_cache()
        a = np.concatenate(flags)
        # pad to T (last partial frame)
        if len(a) < T: a = np.concatenate([a, np.zeros(T-len(a), np.float32)])
        return a[:T]
    print("[vad] computing energy-based speech activity (both channels) ...")
    user_speech = speech_flags(0)          # [T] 0/1 real user speech

    # clip selection: explicit --start windows, else around handoff labels
    if args.start is not None:
        starts = [int(x) for x in str(args.start).split(",")]
        pick = [(s + args.clip_frames // 2) for s in starts]   # pseudo-'handoff' = window centre
        clip_starts = starts
    else:
        if len(ho) == 0:
            sys.exit("no handoffs with enough context; pass --start instead")
        import numpy.random as _r
        rng0 = _r.default_rng(args.seed)
        pick = list(np.sort(rng0.choice(ho, min(args.n_clips, len(ho)), replace=False)))
        clip_starts = [int(h) - args.clip_frames // 2 for h in pick]

    # --- precompute per-frame cb0 distributions from the head (for gate decisions) ---
    #     map frame index -> row in feats
    fr2row = {int(fr): r for r, fr in enumerate(frames)}
    def dist_at(t):
        """cb0 distribution [C,V] predicted at frame t (horizon 1), or None if not available."""
        r = fr2row.get(t)
        if r is None:
            return None
        with torch.no_grad():
            x = torch.from_numpy(feats[r:r+1].astype(np.float32)).to(args.device)
            p = torch.softmax(head(x)[:, 0, :, 0, :], dim=-1)[0]     # [C,V]
        return p.cpu().numpy()

    # probe scorer (optional)
    probe_score = None
    if args.probe and "probe" in args.triggers:
        pr = np.load(args.probe, allow_pickle=True)
        w, b, mu, sd = pr["w"], float(pr["b"]), pr["mu"], pr["sd"]
        def probe_score(p0_now, p0_prev):
            srt = np.sort(p0_now, -1)
            ent = (-(np.clip(p0_now,1e-8,1)*np.log(np.clip(p0_now,1e-8,1))).sum(-1)/np.log(2048)).mean()
            top1 = srt[:,-1].mean(); margin = top1 - srt[:,-2].mean()
            js_prev = 0.0 if p0_prev is None else float(dt.js_div(p0_now,p0_prev).mean())
            has = 0.0 if p0_prev is None else 1.0
            feat = np.array([ent, top1, margin, 0.0, js_prev, 0.0, 0.0, has])
            return float(1/(1+np.exp(-(((feat-mu)/sd)@w+b))))

    needed = lm.num_codebooks - lm.dep_q - 1
    # per-CODEBOOK silence token (each codebook has its own silence value!)
    sil_cb = np.array([[int(np.bincount(tokens[c, q, :]).argmax()) for q in range(Q)]
                       for c in range(C)])                       # [C, Q]
    sil = [int(sil_cb[c, 0]) for c in range(C)]                  # cb0 silence, for activity checks
    dev = args.device; sr = mimi.sample_rate
    args.out.mkdir(parents=True, exist_ok=True)

    def user_frame(t):
        return torch.from_numpy(tokens[0, :, t]).to(dev).long()[None, :, None][:, :needed]
    def silence_moshi():
        # each of Moshi's dep codebooks gets its OWN silence token, not cb0's for all
        vals = torch.tensor(sil_cb[1, :lm.dep_q], dtype=torch.long, device=dev)
        return vals[None, :, None]                               # [1, dep_q, 1]

    def gate_condition(trigger, t, prev):
        """Does `trigger` want to fire at frame t? Returns (fire_bool, dist_at_t)."""
        p = dist_at(t)
        if p is None:
            return False, prev
        ent = -(np.clip(p,1e-8,1)*np.log(np.clip(p,1e-8,1))).sum(-1)/np.log(2048)
        fire = False
        if trigger == "js" and prev is not None:
            jsd = dt.js_div(p, prev)
            fire = bool((jsd < args.js_tau).all() and (ent < args.ent_floor).all())
        elif trigger == "entropy":
            fire = bool((ent < args.entropy_thr).all())
        elif trigger == "probe" and probe_score is not None:
            fire = probe_score(p, prev) >= args.probe_thr
        return fire, p

    def release_schedule(trigger, start, stop, handoffs):
        """Return a boolean array speaking[start:stop]: True where Moshi is released to speak.
        CONTINUOUS: the trigger is evaluated every frame. It fires when ARMED and its condition
        holds while the user is active; once speaking it continues until the user goes quiet
        for `arm_silence` frames, then RE-ARMS and can fire again at the next handoff. This lets
        Moshi take multiple turns across the clip, like a real system."""
        n = stop - start
        speaking = np.zeros(n, bool)
        if trigger == "baseline":
            speaking[:] = True            # vanilla Moshi: never gated, decides on its own
            return speaking
        # oracle / reactive fire at EACH handoff in the span (label-driven), also multi-turn
        if trigger in ("oracle", "reactive"):
            off = -args.lead if trigger == "oracle" else args.reactive_lag
            for h in handoffs:
                rel = h + off
                for t in range(max(start, rel), stop):
                    speaking[t - start] = True     # speak from release to end-of-turn...
                # ...until next handoff (so it re-listens for the following turn)
            # trim: stop speaking at the next handoff so turns alternate
            nxt = sorted(handoffs)
            for h in nxt:
                # after a handoff belonging to the OTHER speaker, user resumes -> gate silent
                pass
            return speaking
        # gate triggers: continuous arm/fire/re-arm
        prev = None; armed = True; silent_run = 0; talk_run = 0
        for t in range(start, stop):
            user_active = user_speech[t] > 0.5      # ENERGY-based: real speech, not token-change
            fire, prev = gate_condition(trigger, t, prev)
            i = t - start
            if armed and (fire or (not user_active)):
                speaking[i] = True
                armed = False; silent_run = 0; talk_run = 1
            elif not armed and talk_run < args.min_talk:
                speaking[i] = True; talk_run += 1     # hold speaking (anti-stutter)
            elif not armed:
                # keep speaking while user quiet; re-arm after a run of user silence
                if not user_active:
                    silent_run += 1
                    speaking[i] = True
                    if silent_run >= args.arm_silence:
                        armed = True; silent_run = 0
                else:
                    speaking[i] = True     # user talking again over us; keep until re-arm window
        return speaking

    def _decode(codes_QT):
        with torch.no_grad():
            w = mimi.decode(torch.from_numpy(codes_QT[None]).to(dev).long()).squeeze().cpu().numpy()
        return w.astype(np.float32)

    def generate(speaking, start, stop):
        """Return a STEREO waveform. `speaking` is a per-frame boolean over [start,stop): where
        True Moshi generates (released); where False it is gated silent (listening). Supports
        MULTIPLE turns per clip, like a real system."""
        gen_cols = []
        lm_gen = LMGen(lm, use_sampling=(not args.greedy), temp=args.temp)
        with torch.no_grad(), lm_gen.streaming(1):
            # A: warm up on REAL user audio (no forced silence) so Moshi's state reflects the
            #    actual conversation entering the clip, not an artificial quiet room.
            for t in range(max(0, start - args.prime), start):
                _ = lm_gen._step(user_frame(t))
            for t in range(start, stop):
                gated = not speaking[t - start]
                if gated and not args.soft_gate:
                    # HARD gate (original): force silence tokens (disrupts state)
                    _ = lm_gen._step(user_frame(t), depformer_replace_tokens=silence_moshi())
                    gen_cols.append(sil_cb[1, :Q].astype(np.int64))
                else:
                    # released, OR soft-gated: let Moshi step normally (state stays live)
                    out = lm_gen._step(user_frame(t))
                    if out is None:
                        gen_cols.append(sil_cb[1, :Q].astype(np.int64))
                    else:
                        gen = out[0] if isinstance(out, tuple) else out
                        col = gen[0, 1:9, 0].cpu().numpy().astype(np.int64)
                        # soft-gate: keep state live but DON'T play Moshi's audio while gated
                        gen_cols.append(sil_cb[1, :Q].astype(np.int64) if gated else col)
        moshi_codes = np.stack(gen_cols, axis=1)                 # [Q, win]
        user_codes = tokens[0, :Q, start:stop].astype(np.int64)  # [Q, win] real user audio
        wav_moshi = _decode(moshi_codes)
        wav_user = _decode(user_codes)
        n = max(len(wav_user), len(wav_moshi))
        # STEREO: user = left channel, Moshi = right channel (separable for visualisation)
        stereo = np.zeros((n, 2), np.float32)
        stereo[:len(wav_user), 0] = wav_user
        stereo[:len(wav_moshi), 1] = wav_moshi
        peak = max(np.abs(stereo).max(), 1e-6)
        return stereo / peak

    trigs = args.triggers.split(",")
    manifest = []
    for n, cstart in enumerate(clip_starts):
        start = max(0, cstart); stop = min(T, start + args.clip_frames)
        clip_ho = [int(x) for x in ho if start <= int(x) < stop]
        entry = {"clip": n, "start": start, "handoff_frames": clip_ho, "turns": {}}
        h = start
        for trig in trigs:
            speaking = release_schedule(trig, start, stop, clip_ho)
            wav = generate(speaking, start, stop)
            fn = args.out / f"clip{n:02d}_h{h}_{trig}.wav"
            sf.write(fn, wav, sr)
            turns = int(np.sum(np.diff(speaking.astype(int)) == 1) + (1 if speaking[0] else 0))
            entry["turns"][trig] = turns
            entry.setdefault("files", []).append(fn.name)
        manifest.append(entry)
        tt = ", ".join(f"{t}={entry['turns'][t]}" for t in trigs)
        print(f"  clip{n:02d} handoffs={clip_ho}: turns taken -> {tt}")

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[out] {len(pick)} clips x {len(trigs)} triggers in {args.out}/")
    print("  Per clip, compare the trigger wavs against oracle:")
    print("   oracle = ideal timing (releases just before the true handoff)")
    print("   js/probe = gate-decided timing. 'never' = gate stayed silent through the turn.")
    print("  Listen for: does the gate release NEAR oracle, LATE, or MISS the turn entirely?")
    print("  Low coverage (js 13%, probe 7%) means MISSES are expected --- that hesitancy is")
    print("  the honest finding: the commit gate is conservative as a handoff trigger.")


if __name__ == "__main__":
    main()
