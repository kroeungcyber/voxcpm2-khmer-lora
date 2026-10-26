#!/usr/bin/env python3
"""
Build VoxCPM2 fine-tuning manifests (train.jsonl / val.jsonl) from a folder of
audio files plus a transcripts file.

Transcripts file is a TSV:  <filename>\t<transcript>   (one per line).

The audio is (optionally) resampled to 16 kHz mono WAV via ffmpeg, because the
VoxCPM AudioVAE encoder expects exactly `sample_rate` (16000).

Usage:
    python data/make_manifest.py \
        --audio_dir data/audio \
        --transcripts data/transcripts.tsv \
        --output_dir data \
        --resample

Output:
    data/train.jsonl   # {"audio": "/abs/path.wav", "text": "..."}
    data/val.jsonl     # only if --val_ratio > 0
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


def resample_to_16k(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    # 16 kHz, mono, 16-bit PCM wav
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def duration_of(path: Path):
    if HAS_SF:
        try:
            return round(float(sf.info(str(path)).duration), 2)
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="folder containing the raw audio files")
    ap.add_argument("--transcripts", required=True, help="TSV: filename<TAB>transcript")
    ap.add_argument("--output_dir", default="data", help="where to write train.jsonl/val.jsonl")
    ap.add_argument("--resample", action="store_true", help="resample to 16 kHz mono via ffmpeg")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="fraction held out for val (0 = no split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = out_dir / "audio_16k"

    rows = []
    with open(args.transcripts, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"[warn] skipping malformed line: {line!r}", file=sys.stderr)
                continue
            fname, text = parts[0].strip(), "\t".join(parts[1:]).strip()
            src = audio_dir / fname
            if not src.exists():
                # also allow the transcript to hold a full path
                alt = Path(fname)
                src = alt if alt.exists() else src
            if not src.exists():
                print(f"[warn] audio not found, skipping: {fname}", file=sys.stderr)
                continue
            audio_path = src
            if args.resample:
                dst = norm_dir / (src.stem + ".wav")
                audio_path = resample_to_16k(src, dst)
            rows.append({
                "audio": str(audio_path),
                "text": text,
                "duration": duration_of(audio_path),
            })

    if not rows:
        print("No rows built — check --audio_dir and --transcripts.", file=sys.stderr)
        sys.exit(1)

    # strip None durations
    for r in rows:
        if r["duration"] is None:
            r.pop("duration", None)

    total = sum(r.get("duration", 0) or 0 for r in rows)
    print(f"built {len(rows)} utterances ({total/60:.2f} min of audio)")

    if args.val_ratio > 0:
        import random
        random.Random(args.seed).shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_ratio))
        val, train = rows[:n_val], rows[n_val:]
        write_jsonl(out_dir / "val.jsonl", val)
        write_jsonl(out_dir / "train.jsonl", train)
        print(f"train: {len(train)}  val: {len(val)}")
    else:
        write_jsonl(out_dir / "train.jsonl", rows)


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
