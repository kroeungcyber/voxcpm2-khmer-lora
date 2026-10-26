# VoxCPM2 Khmer LoRA — personalized Khmer voice for instructional video

Fine-tune [OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (a 2B,
tokenizer-free, diffusion-autoregressive TTS model with **native Khmer**) using
**LoRA**, to create a *stable, reusable, personalized Khmer narrator voice* —
then use it to narrate instructional videos end-to-end, on [RunPod](https://runpod.io).

**What you get:**

| Artifact | Purpose |
|---|---|
| `data/make_manifest.py` | Build `train.jsonl` from audio + Khmer transcripts (16 kHz resample) |
| `configs/finetune_lora_khmer.yaml` | LoRA fine-tuning config (r=32, LM+DiT adapters) |
| `scripts/train.sh` | One command: clone VoxCPM → install → download weights → train |
| `scripts/generate.py` | Text → speech using the fine-tuned LoRA voice |
| `scripts/narrate_video.py` | Narrate a script file → per-segment WAVs + narration track + SRT captions |
| `runpod/` | Dockerfile + startup script for one-click GPU training |

---

## Why LoRA (vs. VoxCPM2's built-in voice cloning)

VoxCPM2 already clones a voice from a short reference clip. That's great for
*one-off* audio. LoRA fine-tuning is better when you need:

1. **A stable, reusable voice** — no reference clip needed at inference time.
2. **Consistent Khmer pronunciation & style** — the model adapts to *your*
   voice's Khmer, not the averaged voice of the base model.
3. **Domain adaptation** — instructional/lecture delivery style baked in.

LoRA only trains ~0.5% of the parameters (r=32 adapters on the language model
and diffusion decoder), so a single 24GB GPU is enough and training is fast on
**5–10 minutes** of audio.

---

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.5.0, CUDA ≥ 12.0
- NVIDIA GPU: **16 GB VRAM minimum, 24 GB+ recommended**
- `ffmpeg` + `libsndfile1` (for audio IO/resampling)
- A [Hugging Face](https://huggingface.co) account (VoxCPM2 is public, but a
  token avoids download throttling) — set `HF_TOKEN`.

---

## Repo layout

```
voxcpm2-khmer-lora/
├── configs/finetune_lora_khmer.yaml   # LoRA training config
├── data/
│   ├── README.md                      # how to record + transcribe the dataset
│   └── make_manifest.py               # audio + TSV -> train.jsonl/val.jsonl
├── scripts/
│   ├── train.sh                       # end-to-end training entrypoint
│   ├── generate.py                    # LoRA inference (text -> wav)
│   └── narrate_video.py               # script -> narration + SRT
├── runpod/
│   ├── Dockerfile                     # custom GPU image
│   └── start.sh                       # pod entrypoint (checks GPU + data)
├── prompts/                           # sample Khmer transcripts + video script
└── requirements.txt
```

---

## Step 1 — Prepare the dataset (5–10 min of your voice)

Follow [`data/README.md`](data/README.md):

1. Record **one speaker** (you) reading Khmer instructional phrases —
   start from [`prompts/khmer_sample_transcripts.txt`](prompts/khmer_sample_transcripts.txt).
2. Split into 2–10 s WAVs in `data/audio/`.
3. Write `data/transcripts.tsv` (`filename<TAB>text`, exact match to speech).
4. Build the manifest:

```bash
python data/make_manifest.py \
    --audio_dir data/audio \
    --transcripts data/transcripts.tsv \
    --output_dir data \
    --resample
```

---

## Step 2 — Train on RunPod

### Option A — Pod + template (fastest)

1. In RunPod → **Pods → Deploy**.
2. GPU: **RTX 4090 (24GB)** is cheapest; **A6000 (48GB)** or **A100 40GB** for headroom.
3. Template: **RunPod PyTorch 2.5** (CUDA 12.x).
4. Attach a **Network Volume** at `/workspace` (persists model + checkpoints).
5. Put the repo at `/workspace` (the scripts assume this root):

```bash
cd /workspace
git clone https://github.com/kroeungcyber/voxcpm2-khmer-lora.git .
# upload/place your data/audio + data/transcripts.tsv, then:
python data/make_manifest.py --audio_dir data/audio --transcripts data/transcripts.tsv --output_dir data --resample
```

6. Run training (via the pod's terminal):

```bash
export HF_TOKEN=hf_...        # optional, avoids rate limits
bash scripts/train.sh
```

### Option B — Custom Docker image

Build and push `runpod/Dockerfile` to a registry, then deploy it as a
custom image (mount `/workspace` and `HF_TOKEN`). The container runs
`runpod/start.sh` on boot: GPU check → data check → `scripts/train.sh`.

---

## Step 3 — Generate speech with the trained voice

The training run writes checkpoints to `checkpoints/khmer_lora/step_NNNN/`,
each containing `lora_weights.safetensors` + `lora_config.json`.

```bash
python scripts/generate.py \
    --lora_ckpt checkpoints/khmer_lora/step_0001000 \
    --text "សូមស្វាគមន៍មកកាន់វគ្គបណ្តុះបណ្តាល។" \
    --output output/hello.wav
```

---

## Step 4 — Narrate an instructional video

```bash
python scripts/narrate_video.py \
    --lora_ckpt checkpoints/khmer_lora/step_0001000 \
    --script prompts/video_script_sample.txt \
    --output_dir output/narration
```

Produces `seg_*.wav`, `narration.wav`, and `narration.srt` (captions with
timings). Mux over your silent screen recording:

```bash
ffmpeg -i screen.mp4 -i output/narration/narration.wav \
    -c:v copy -c:a aac -shortest final.mp4
```

---

## GPU sizing

| GPU | VRAM | Training (r=32, bs=2) | Notes |
|---|---|---|---|
| RTX 4090 | 24 GB | ✅ (may need `batch_size: 1`) | Cheapest |
| A6000 | 48 GB | ✅ comfortable | Best value |
| A100 40/80GB | 40/80 GB | ✅ | Fastest |

If you hit OOM, lower `batch_size` to `1` and raise `grad_accum_steps` to keep
the effective batch size (≈16) the same.

---

## Troubleshooting

- **`sample_rate mismatch`** — the config's `sample_rate` must equal the
  AudioVAE encoder rate. Keep `16000` and always resample your data.
- **`lora_config.json not found`** — you're pointing at a checkpoint without the
  LoRA metadata; use a `step_NNNN` directory produced by the training script.
- **OOM during training** — reduce `batch_size`, or set `enable_proj: true` off
  and shrink `r` (e.g. 16).
- **Khmer mispronunciation** — usually a data problem: more clean audio, exact
  transcripts, consistent speaker. 10+ minutes beats 5.

---

## References

- Model: <https://huggingface.co/openbmb/VoxCPM2>
- Code & training script: <https://github.com/OpenBMB/VoxCPM>
- Fine-tuning guide: <https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html>

Apache-2.0 licensed model — free for commercial use.
