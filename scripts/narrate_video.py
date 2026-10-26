#!/usr/bin/env python3
"""
Narrate an instructional video script with the fine-tuned VoxCPM2 Khmer voice.

Input: a plain-text script — one spoken segment per line (blank lines / # comments
ignored). Output per-segment WAVs, a concatenated narration track, and an SRT
caption file with estimated timings.

Usage:
    python scripts/narrate_video.py \
        --lora_ckpt checkpoints/khmer_lora/step_0001000 \
        --script prompts/video_script.txt \
        --output_dir output/narration

Then mux narration over a silent screen recording:
    ffmpeg -i screen.mp4 -i output/narration/narration.wav \
        -c:v copy -c:a aac -shortest final.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from voxcpm.core import VoxCPM

try:
    from voxcpm.model.voxcpm2 import LoRAConfig
except ImportError:
    from voxcpm.model.voxcpm import LoRAConfig


def ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_ckpt", required=True)
    ap.add_argument("--base_model", default="")
    ap.add_argument("--script", required=True, help="text file, one segment per line")
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--cfg_value", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = Path(args.lora_ckpt)
    info = json.loads((ckpt / "lora_config.json").read_text(encoding="utf-8"))
    base = args.base_model or info.get("base_model") or "openbmb/VoxCPM2"
    lora_cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None

    print(f"Loading base model: {base}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=base,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt),
    )

    segments = [
        l.strip()
        for l in Path(args.script).read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    if not segments:
        print("No segments found in script.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = model.tts_model.sample_rate

    parts, srt, t = [], [], 0.0
    for i, text in enumerate(segments):
        wav = model.generate(
            text=text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.timesteps,
            denoise=False,
            seed=args.seed,
        )
        seg_path = out / f"seg_{i:03d}.wav"
        sf.write(str(seg_path), wav, sr)
        dur = len(wav) / sr
        srt.append(f"{i + 1}\n{ts(t)} --> {ts(t + dur)}\n{text}\n")
        t += dur
        parts.append(wav)
        print(f"[{i + 1}/{len(segments)}] {dur:.2f}s  {text[:50]}", file=sys.stderr)

    full = np.concatenate(parts)
    sf.write(str(out / "narration.wav"), full, sr)
    (out / "narration.srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"Wrote {len(parts)} segments + narration.wav + narration.srt -> {out}")


if __name__ == "__main__":
    main()
