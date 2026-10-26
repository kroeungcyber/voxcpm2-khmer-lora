# RunPod Fine-Tuning — Step-by-Step

Turnkey guide for training your personalized Khmer voice on RunPod, using the
`voxcpm2-khmer-lora` project. Assumes you have already prepared the dataset
locally (see `../data/README.md`): `data/audio/*.wav` + `data/transcripts.tsv`.

---

## 0. Prerequisites

- A [RunPod](https://runpod.io) account with credits.
- A [Hugging Face](https://huggingface.co) token (Settings → Access Tokens → Read).
  VoxCPM2 is public, but a token avoids download throttling.
- Your dataset ready on your **local** computer (audio + `transcripts.tsv`).

---

## 1. Create a Network Volume (persistent storage)

Pods are ephemeral — anything outside `/workspace` on a *Network Volume* is
deleted when you stop the pod. Store model + checkpoints + data on a volume.

1. RunPod console → **Storage** → **Network Volume** → **Create**.
2. Size: **40 GB** (VoxCPM2 weights ~5 GB + checkpoints + audio). Region: closest.
3. Note the volume name.

---

## 2. Deploy the GPU pod

1. RunPod console → **Pods** → **Deploy**.
2. **GPU:**
   - `RTX 4090` (24 GB) — cheapest, works (lower `batch_size` to 1 if OOM).
   - `A6000` (48 GB) — recommended, comfortable headroom.
   - `A100 40GB` — fastest.
3. **Template:** `RunPod PyTorch 2.5` (CUDA 12.x). If absent, use the newest
   `RunPod PyTorch` template, or the **custom image** path (`../runpod/Dockerfile`).
4. **Attach storage:** in the pod config, select **Network Volume** → your volume
   → mount path **`/workspace`**.
5. **Environment variables** (optional but recommended):
   - `HF_TOKEN` = `hf_...`
6. **Deploy.**

---

## 3. Connect to the pod

- Easiest: click the pod → **Connect** → **Web Terminal** (in-browser shell).
- Or SSH (copy the SSH command from the pod's **Connect** dialog).

---

## 4. Get the repo onto the pod

In the pod terminal:

```bash
cd /workspace
git clone https://github.com/kroeungcyber/voxcpm2-khmer-lora.git .
```

This puts `configs/`, `scripts/`, `data/`, `prompts/`, `runpod/` at `/workspace`,
which is what the scripts expect.

---

## 5. Upload your dataset

Your local `data/audio/*.wav` and `data/transcripts.tsv` need to reach
`/workspace/data/`. Two options:

**Option A — SCP (recommended, many small files):**

On your **local** machine, tar the data first, then copy:

```bash
tar czf data.tar.gz data/audio data/transcripts.tsv
scp -P <SSH_PORT> data.tar.gz root@<POD_IP>:/workspace/
```

Then in the pod:

```bash
cd /workspace && tar xzf data.tar.gz
```

**Option B — RunPod file upload:** the pod's file browser / JupyterLab lets you
drag-and-drop, but it's slow for many files — use SCP for anything > a few MB.

---

## 6. Build the training manifest

In the pod:

```bash
cd /workspace
python data/make_manifest.py \
    --audio_dir data/audio \
    --transcripts data/transcripts.tsv \
    --output_dir data \
    --resample
```

You should see `wrote /workspace/data/train.jsonl` with ~60–100 utterances.

---

## 7. Run training

```bash
cd /workspace
bash runpod/start.sh      # checks GPU + data, then runs scripts/train.sh
```

Or directly: `bash scripts/train.sh`.

What happens (automatically):
1. Clones `OpenBMB/VoxCPM`, `pip install -e .`
2. Downloads `openbmb/VoxCPM2` → `/workspace/models/VoxCPM2` (~5 GB)
3. Runs `train_voxcpm_finetune.py` with the LoRA config

**GPU check:** `nvidia-smi` should show your GPU. If training OOMs, edit
`configs/finetune_lora_khmer.yaml`: set `batch_size: 1` and `grad_accum_steps: 16`.

---

## 8. Monitor

```bash
tail -f /workspace/checkpoints/khmer_lora/train.log
```

Checkpoints appear at `/workspace/checkpoints/khmer_lora/step_NNNN/`, each with
`lora_weights.safetensors` + `lora_config.json`.

TensorBoard (optional):

```bash
tensorboard --logdir /workspace/logs --port 6006 --bind_all
```

Then in RunPod, click the pod → **Connect** → expose port **6006** → open the URL.

---

## 9. Stop the pod + download the checkpoint

- **Stop the pod** the moment training finishes (you pay per hour).
- Download the best checkpoint to your local machine:

```bash
scp -r -P <SSH_PORT> root@<POD_IP>:/workspace/checkpoints/khmer_lora/step_0001000 ./checkpoint
```

(Or keep it on the Network Volume; it persists after the pod stops.)

---

## 10. Generate speech with your new voice

Locally (or on any small GPU pod):

```bash
pip install voxcpm soundfile
python scripts/generate.py \
    --lora_ckpt ./checkpoint \
    --text "សូមស្វាគមន៍មកកាន់វគ្គបណ្តុះបណ្តាល។" \
    --output hello.wav
```

For video narration, see `../scripts/narrate_video.py`.

---

## Cost & time expectations

| Item | Estimate |
|---|---|
| Dataset prep | done locally, free |
| VoxCPM2 download | ~5 GB, a few min |
| Training (5–10 min audio, r=32) | ~30–60 min on A6000 |
| RTX 4090 | ~$0.34/hr community / ~$0.69/hr secure |
| A6000 (48 GB) | ~$0.79/hr |

Total for one training run: **~$1–2** on a 4090. Prices vary — check the pod
picker before deploying.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sample_rate mismatch` | your audio isn't 16 kHz — rerun `make_manifest.py --resample` |
| CUDA OOM | `batch_size: 1`, `grad_accum_steps: 16` |
| Slow download | set `HF_TOKEN`; retry |
| `train.jsonl not found` | you skipped step 6 — build the manifest first |
| Poor voice quality | more clean audio, exact transcripts, consistent speaker |
