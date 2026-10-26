#!/usr/bin/env python3
"""
Generate speech with a fine-tuned VoxCPM2 LoRA voice.

Reads the checkpoint's lora_config.json (which stores base_model + lora config)
and synthesizes `--text` in Khmer or any of the 30 supported languages.

Usage:
    python scripts/generate.py \
        --lora_ckpt checkpoints/khmer_lora/step_0001000 \
        --text "សូមស្វាគមន៍មកកាន់វគ្គបណ្តុះបណ្តាល។" \
        --output output/hello.wav
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from voxcpm.core import VoxCPM

# VoxCPM2 uses the v2 LoRAConfig; fall back to v1 for older checkpoints.
try:
    from voxcpm.model.voxcpm2 import LoRAConfig
except ImportError:
    from voxcpm.model.voxcpm import LoRAConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_ckpt", required=True, help="LoRA checkpoint dir (has lora_config.json)")
    ap.add_argument("--base_model", default="", help="override base model path")
    ap.add_argument("--text", required=True, help="text to synthesize")
    ap.add_argument("--output", default="output.wav")
    ap.add_argument("--cfg_value", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10, help="diffusion inference steps")
    ap.add_argument("--max_len", type=int, default=600)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    ckpt = Path(args.lora_ckpt)
    info = json.loads((ckpt / "lora_config.json").read_text(encoding="utf-8"))
    base = args.base_model or info.get("base_model") or "openbmb/VoxCPM2"
    lora_cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None

    print(f"Loading base model: {base}", file=sys.stderr)
    print(f"Loading LoRA from:  {ckpt}", file=sys.stderr)
    model = VoxCPM.from_pretrained(
        hf_model_id=base,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt),
    )

    wav = model.generate(
        text=args.text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.timesteps,
        max_len=args.max_len,
        denoise=False,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, model.tts_model.sample_rate)
    print(
        f"Saved {out} ({len(wav) / model.tts_model.sample_rate:.2f}s, "
        f"{model.tts_model.sample_rate} Hz)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
