# EchoForge AI

A text-to-speech engine with voice cloning support, rebranded from the open-source F5-TTS project.

## Stack

- **Language**: Python 3.10+
- **Package**: `echoforge_tts` (installable as `echoforge-tts`)
- **UI**: Gradio web interface
- **Models**: DiT (F5-TTS backbone) and UNetT (E2 TTS backbone)
- **Runtime**: PyTorch; GPU strongly recommended

## Package structure

```
src/echoforge_tts/
  api.py              # Main F5TTS class
  model/              # Model definitions (do NOT modify internals)
  infer/              # Inference: CLI, Gradio UI, utilities
  train/              # Training, finetuning: CLI, Gradio UI
  eval/               # Evaluation scripts
  configs/            # YAML model configs (EchoForge_*.yaml)
  runtime/            # Triton/TRT-LLM deployment
  scripts/            # Utility scripts
```

## Running the Gradio UI

```bash
pip install -e .
echoforge_infer-gradio --port 7860 --host 0.0.0.0
```

GPU (CUDA/ROCm) required for practical inference speed. Model weights (~1–4 GB) are downloaded from Hugging Face (SWivid/F5-TTS) on first run.

## CLI inference

```bash
echoforge_infer-cli --model EchoForge_v1_Base \
  --ref_audio ref.wav \
  --ref_text "Transcription of reference." \
  --gen_text "Text to synthesize."
```

## Attribution

EchoForge AI is built on top of F5-TTS (https://github.com/SWivid/F5-TTS), originally created by Yushen Chen and contributors, licensed under MIT.

## User preferences

- Surface-level rebrand only — do NOT touch model architecture files (dit.py, mmdit.py, unett.py, cfm.py, modules.py internals, trainer.py training logic)
- HuggingFace download paths pointing to `SWivid/F5-TTS` must remain unchanged
- MIT LICENSE copyright notice must remain unchanged
