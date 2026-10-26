# Dataset preparation

VoxCPM2 LoRA fine-tuning needs **5–10 minutes** of clean, single-speaker audio
paired with exact Khmer transcripts. Quality and speaker-consistency matter far
more than quantity.

## 1. Record the voice

- **One speaker** for the whole dataset (the voice you want to personalize).
- Record in a quiet room; phone or USB mic is fine.
- Speak naturally at a steady pace — this will be a narrator voice for videos.
- Cover the instructional domain: course intros, step-by-step instructions,
  reminders, conclusions. Start from `../prompts/khmer_sample_transcripts.txt`.

## 2. Split and name the audio

Split into short utterances (2–10 s each). Put them in `data/audio/`:

```
data/audio/
  001.wav
  002.wav
  ...
```

## 3. Transcribe exactly

Create `data/transcripts.tsv` — one line per file, `filename<TAB>text`:

```
001.wav	ជំរាបសួរ សូមស្វាគមន៍មកកាន់វគ្គបណ្តុះបណ្តាលនេះ។
002.wav	ថ្ងៃនេះ យើងនឹងរៀនអំពីបញ្ញាសិប្បនិម្មិត។
```

The transcript must match what is **actually spoken** — word for word. This is
the most common cause of poor fine-tuning results.

## 4. Build the manifest

```bash
python data/make_manifest.py \
    --audio_dir data/audio \
    --transcripts data/transcripts.tsv \
    --output_dir data \
    --resample            # convert to 16 kHz mono (AudioVAE requirement)
```

This writes `data/train.jsonl`:

```jsonl
{"audio": "/workspace/data/audio_16k/001.wav", "text": "ជំរាបសួរ ...", "duration": 3.2}
```

## Notes

- **Audio format:** 16 kHz mono WAV (the `--resample` flag does this with ffmpeg).
- **Duration field:** optional, but speeds up any length filtering during training.
- **Val split:** add `--val_ratio 0.1` to hold out 10% for validation, then set
  `val_manifest: /workspace/data/val.jsonl` in the config.
- **Khmer spelling:** keep it consistent (the exact text isn't spoken phonetically
  by the model — it learns the mapping from your audio, so consistency matters).
