# EchoForge AI — Complete Internal Architecture Audit
### (F5-TTS backbone ka poora post-mortem: Data → Training → Fine-tuning → Inference → Eval → Deployment)

**Audit date:** 16 July 2026
**Scope:** Poore `src/echoforge_tts/` tree ka har `.py` file (~40 files, ~10,400 lines) padha gaya — model backbones, training loop, dataset prep, inference pipelines, evaluation scripts, web UIs (Gradio), socket server, aur Triton/TensorRT-LLM deployment path.
**Kya rebrand touch nahi hua (confirmed dobara):** HuggingFace checkpoint paths (`SWivid/F5-TTS`, `SWivid/E2-TTS`), MIT license, Triton model repo internals.

---

## 0. Seedha jawab pehle (TL;DR — reality check)

Tune bola tha "you know better than me, per jo maine kaha ushke hisab se bhi soch lena" — to yahi hai wo:

1. **Backbone F5-TTS hi hai, aur rahega.** Architecture (DiT/MMDiT/UNetT + Conditional Flow Matching) research-grade hai — ye tera "insaan ki backbone" wala analogy sahi hai, per is backbone ko badalna matlab naya paper likhna hoga. Realistic goal hai: is backbone ko **samajhna + fine-tune karna + apne use-case (Hindi/Hinglish, Telegram bot voices) ke liye specialize karna** — replace karna nahi.
2. **Sabse bada real gap jo maine dhoonda:** tokenizer/G2P pipeline **hardcoded Chinese-centric** hai (`is_chinese()` range check + `rjieba` + `pypinyin`). Hindi/Devanagari ke liye koi phonemization nahi hai — text seedha byte ya char tokenizer mein chala jayega. Agar Hindi voice quality achhi chahiye, ye sabse important cheez hai jo touch karni padegi (detail Section 4, item #13 aur Section 6 mein).
3. Maine is audit ke dauraan **naye risk points bhi dhoonde** jo pehle kabhi flag nahi hue the — un mein sabse serious hai finetune WebUI ka training launch `subprocess.Popen(cmd, shell=True)` pattern aur ek **unauthenticated raw TCP socket server** jo `0.0.0.0` pe bind hota hai. Dono production-relevant hain kyuki tu ise ek bechne-wali service bana raha hai. Detail Section 5 mein — ye "bugs" nahi balki "hardening gaps" hain, security-audit ki tarah treat kar.
4. Yeh document teen kaam karta hai: **(a)** poora workflow explain karta hai starting-to-end, **(b)** har algorithm ko rate karta hai apni honest opinion ke sath, **(c)** ek alag block mein sirf "kya-kya ho sakta hai" wale naye feature ideas deta hai — jo tune specifically maanga tha.

---

## 1. Bird's-Eye View — Poora System Ek Nazar Mein

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA PREP          train/datasets/prepare_*.py                │
│  Raw audio+text  →  filter (dur 0.3-30s, repetition, bad chars)         │
│                  →  Arrow file (raw.arrow) + duration.json + vocab.txt  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: TOKENIZATION        model/utils.py                            │
│  Text → pinyin (ZH) / char / byte / custom vocab → int index tensor     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: BACKBONE            model/backbones/{dit,mmdit,unett}.py      │
│  Text-embed + Audio(mel)-embed + Time-embed → Transformer blocks        │
│  → predicted flow vector (same shape as mel)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: CORE ALGORITHM      model/cfm.py  (Conditional Flow Matching) │
│  TRAIN: mel + noise ka interpolation → backbone predicts flow → MSE loss│
│  INFER: pure noise → ODE solver (Euler) → 5-32 steps → clean mel        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
┌───────────────────────────┐               ┌───────────────────────────────┐
│  STAGE 5a: TRAINING LOOP   │               │  STAGE 5b: INFERENCE          │
│  model/trainer.py          │               │  infer/utils_infer.py         │
│  Accelerate + EMA + AdamW  │               │  mel → Vocoder (Vocos/BigVGAN)│
│  + LR warmup/decay         │               │  → raw waveform (.wav)        │
└───────────────────────────┘               └───────────────────────────────┘
              │                                             │
┌───────────────────────────┐               ┌───────────────────────────────┐
│  STAGE 6: FINE-TUNING      │               │  STAGE 7: SERVING              │
│  train/finetune_cli.py     │               │  infer_gradio.py (Web UI)     │
│  train/finetune_gradio.py  │               │  api.py (Python SDK)           │
│  (dataset UI + vocab       │               │  socket_server.py (streaming)  │
│   extension + subprocess   │               │  runtime/triton_trtllm/       │
│   launch of train.py)      │               │  (production GPU serving)     │
└───────────────────────────┘               └───────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 8: EVALUATION          eval/*.py                                 │
│  WER (Whisper/Paraformer ASR) + SIM (WavLM+ECAPA speaker embedding)     │
│  + UTMOS (naturalness MOS predictor)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer-wise summary table

| Layer | Kaam kya hai | Key files | Kis language/framework pe |
|---|---|---|---|
| Data prep | Raw datasets ko Arrow format mein convert karna, duration + vocab nikalna | `train/datasets/prepare_*.py` | HF `datasets`, multiprocessing |
| Tokenizer | Text → integer sequence | `model/utils.py` | `rjieba`, `pypinyin` |
| Mel extraction | Audio waveform → mel-spectrogram | `model/modules.py` (`MelSpec`) | `torchaudio`, `vocos`/BigVGAN |
| Backbone | Neural network jo flow predict karta hai | `model/backbones/*.py` | PyTorch, `x_transformers` |
| Core algorithm | Training loss + inference sampling | `model/cfm.py` | `torchdiffeq` |
| Trainer | Distributed training loop | `model/trainer.py` | HuggingFace `accelerate`, `ema_pytorch` |
| Inference utils | Preprocessing + batching + vocoding | `infer/utils_infer.py` | `pydub`, `transformers` (Whisper) |
| Web UI | Gradio-based demo/product UI | `infer/infer_gradio.py`, `train/finetune_gradio.py` | `gradio` |
| SDK | Simple Python class for external use | `api.py` | — |
| Real-time serving | Raw TCP streaming | `socket_server.py`, `socket_client.py` | Python `socket` |
| Production serving | GPU-optimized inference server | `runtime/triton_trtllm/` | NVIDIA Triton + TensorRT-LLM |
| Evaluation | Quality metrics | `eval/*.py` | `jiwer`, WavLM, UTMOS |


---

## 2. Complete Workflow — Step-by-Step (Ek AI Model Kaise Paida Hoti Hai)

Tune bola tha "ek AI model ki starting training se suru hoti hai fine tuning hoti hai response generate karna vagera hota hai" — to yahi hai poora lifecycle, exact jaisa code mein hai:

### 2.1 — Dataset Preparation (`train/datasets/prepare_*.py`)

Yeh sab scripts same output format banate hain — ek **Arrow file** (`raw.arrow`, HuggingFace `datasets` library ka fast columnar binary format), `duration.json`, aur `vocab.txt`.

| Script | Kis dataset ke liye | Filtering logic |
|---|---|---|
| `prepare_emilia.py` / `prepare_emilia_v2.py` | Emilia (ZH+EN, multilingual wild-speech dataset) | Duplicate/synthesized-sounding clips ko hardcoded UUID set se exclude karta hai; `repetition_found()` se "aaaaa..." jaisi corrupted transcripts filter karta hai |
| `prepare_libritts.py` | LibriTTS (English, studio-clean) | Duration 0.4–30s ke bahar drop |
| `prepare_ljspeech.py` | LJSpeech (single speaker EN) | Same duration filter |
| `prepare_wenetspeech4tts.py` | WenetSpeech4TTS (Mandarin) | txt+wav pairing via directory scan |
| `prepare_csv_wavs.py` | **Tera apna custom dataset** — `audio_file|text` CSV format | Duration 0.3–30s, multi-worker (`concurrent.futures`) |

**Kyun important hai tere liye:** Agar Hindi voice data se fine-tune karna hai, to `prepare_csv_wavs.py` hi tera entry point hai — isse ek CSV banake (absolute wav paths + text) directly Arrow dataset mil jayega. Isi function ko `finetune_gradio.py`'s `create_metadata()` bhi internally reuse karta hai jab tu Web UI se project banata hai.

### 2.2 — Tokenization (`model/utils.py`)

Do cheezein hoti hain text ke sath:

1. **`convert_char_to_pinyin()`** — agar tokenizer `"pinyin"` hai (jo default hai `EchoForge_v1_Base` config mein), to `rjieba` (Rust-based Chinese word segmenter) text ko words mein todta hai, phir Chinese characters ko `pypinyin` (tone-marked pinyin, jaise "ni3 hao3") mein convert karta hai. Non-Chinese (Latin alphabets) chars ko as-is chhod deta hai.
2. **`get_tokenizer()`** — ek `vocab.txt` file (ek line = ek token) padh ke `{char: index}` dictionary banata hai. Idx `0` hamesha space `" "` ke liye reserved hai (unknown-token fallback).

⚠️ **Yahi wo jagah hai jahan Devanagari (Hindi) ka koi native support nahi hai** — `is_chinese()` function sirf `\u3100`–`\u9fff` range check karta hai. Hindi text `pypinyin`/`rjieba` se guzregi hi nahi (wo sirf CJK detect karte hain), isliye Hindi characters seedhe raw chars ki tarah vocab mein jaayenge — jo "char" tokenizer mode mein theek chalega, lekin usmein Chinese jaisa tone/phoneme normalization nahi milega.

### 2.3 — Mel-Spectrogram Extraction (`model/modules.py :: MelSpec`)

Audio waveform (24kHz) ko mel-spectrogram mein convert kiya jata hai — ye "target" hai jo model ko generate karna seekhna hai. Do backend supported:
- **Vocos** style (`get_vocos_mel_spectrogram`) — default, lightweight
- **BigVGAN** style (`get_bigvgan_mel_spectrogram`) — alternate vocoder ke sath compatible

Params: `n_fft=1024`, `hop_length=256`, `win_length=1024`, `n_mel_channels=100`, `sample_rate=24000`.

### 2.4 — Model Backbone (3 options — `model/backbones/`)

Text-embedding, noised-audio-embedding, aur time-embedding ko combine karke ek Transformer chalata hai jo **flow vector** predict karta hai (detail Section 4 mein). Teen architecture variants available hain:

| Backbone | Style | Use hota hai kab | Params (Base config) |
|---|---|---|---|
| `DiT` (`dit.py`) | Diffusion Transformer — text aur audio dono ek hi sequence mein concat, single-stream attention | `EchoForge_v1_Base` / `EchoForge_Base` (production model) | dim=1024, depth=22, heads=16 |
| `MMDiT` (`mmdit.py`) | Multi-Modal DiT — Stable-Diffusion-3 style **joint attention** (text aur audio separate streams, shared attention op) | Experimental/newer variant | Configurable |
| `UNetT` (`unett.py`) | Flat U-Net-style Transformer with skip connections | `E2TTS_Base` (E2-TTS paper ka architecture) | dim=1024, depth=24, heads=16 |

### 2.5 — Core Algorithm: Conditional Flow Matching (`model/cfm.py`)

Ye "dil" hai poore system ka.

**Training (`CFM.forward`):**
1. `x1` = real mel-spectrogram (ground truth)
2. `x0` = pure Gaussian noise (same shape)
3. Random `time ∈ [0,1]` sample karo
4. `φ_t = (1-t)·x0 + t·x1` — ye ek **straight line interpolation** hai noise se real audio tak (isko "rectified flow" kehte hain)
5. `flow = x1 - x0` — target hai ki model ye direction predict kare
6. Ek random contiguous span (70-100% of length) ko mask karke model ko "infilling" sikhaya jata hai — matlab model sirf masked hisse ko predict karta hai, baaki (reference audio wala hissa) already given hota hai
7. Loss = MSE(predicted_flow, actual_flow), sirf masked region pe

**Inference (`CFM.sample`):**
1. Pure noise se shuru karo (`y0`)
2. `torchdiffeq.odeint()` — ek ODE solver (Euler method default) — jo `nfe_step` (default 32, kam bhi chal sakta hai) steps mein noise ko dheere-dheere real mel-spectrogram ki taraf le jaata hai
3. Har step pe backbone se predicted flow milta hai, wahi "velocity field" hai jo ODE solver follow karta hai
4. Reference audio ka mel hamesha "fixed" rehta hai (`cond`), sirf naya text wala hissa generate hota hai

### 2.6 — Training Loop (`model/trainer.py`)

- **Framework:** HuggingFace `accelerate` (multi-GPU/DDP ready, though tu single Colab GPU pe chalayega)
- **Optimizer:** `AdamW` (fused=True) ya `bitsandbytes` 8-bit AdamW (kam VRAM ke liye)
- **LR Schedule:** `SequentialLR` = Linear warmup (20,000 updates) → Linear decay to near-zero
- **EMA:** `ema_pytorch` library — training weights ke ek "smoothed average" copy rakhi jaati hai jo actual inference mein use hoti hai (zyada stable output deta hai)
- **Batching:** `DynamicBatchSampler` — frame-count (total audio duration) ke hisaab se batch size **dynamically** adjust hoti hai, chhote samples zyada batch mein fit ho jaate hain (padding waste kam)
- **Checkpointing:** Har `save_per_updates` (default 50k) pe poora checkpoint save, `keep_last_n_checkpoints` se rotation

### 2.7 — Fine-tuning (`train/finetune_cli.py` + `train/finetune_gradio.py`)

Fine-tuning **wahi `Trainer` class reuse karta hai**, bas do farak:
1. Pretrained checkpoint pehle download/copy hota hai (`hf://SWivid/F5-TTS/...`)
2. Learning rate chhota hota hai (`1e-5` vs scratch training ka `7.5e-5`)

`finetune_gradio.py` ek **poori Web-based data-prep + training-launch UI** hai — bahut zyada wiring hai jo tu Colab/Replit se already use kar raha hoga:
- Audio **auto-slicing** (`Slicer` class, RVC-Boss/GPT-SoVITS se liya gaya) — lambi recording ko silence detect karke chhote clips mein todta hai
- Auto-**transcription** (Whisper ASR)
- **Vocabulary extension** (naye characters/symbols ko existing model mein add karna — detail Section 4 #19)
- Training subprocess launch: `accelerate launch finetune_cli.py --args...`

### 2.8 — Inference (`infer/utils_infer.py`, `infer/infer_cli.py`, `api.py`)

1. **Reference audio preprocess** — silence-trim, 12-second clip limit, agar `ref_text` nahi diya to Whisper se auto-transcribe (result cached by MD5 hash)
2. **Long text chunking** (`chunk_text`) — bada text sentence-boundary pe chhote pieces mein todta hai (max ~135 UTF-8 bytes per chunk by default)
3. **Batch inference** — sab chunks `ThreadPoolExecutor` se **parallel** generate hote hain
4. **Vocoding** — mel-spectrogram → raw waveform (`vocos.decode()` ya BigVGAN forward)
5. **Cross-fade stitching** — chunks ko concatenate karte waqt 0.15s ka linear cross-fade lagaya jaata hai taaki joints sunayi na de

### 2.9 — Evaluation (`eval/*.py`)

Teen standard TTS metrics:
- **WER** (Word Error Rate) — generated audio ko Whisper/Paraformer se wapas transcribe karke ground-truth text se compare
- **SIM** (Speaker Similarity) — WavLM-large + ECAPA-TDNN embeddings ka cosine similarity (kya generated voice reference jaisi sunayi de rahi hai)
- **UTMOS** — ek pretrained naturalness-predictor model jo "kitna real lag raha hai" score deta hai (1-5 MOS scale)

### 2.10 — Serving/Deployment

| Method | File | Use-case |
|---|---|---|
| Gradio Web demo | `infer_gradio.py` | Tera abhi wala rebrand target — browser-based UI |
| Python SDK | `api.py` (`F5TTS` class) | Script/notebook se direct import karke use karna |
| Raw TCP streaming | `socket_server.py` + `socket_client.py` | Low-latency real-time streaming (chunk-by-chunk audio) |
| Production GPU server | `runtime/triton_trtllm/` | NVIDIA Triton + TensorRT-LLM — high-throughput multi-user serving (tune isko already "justified reason se untouched" chhoda hai) |


---

## 3. File-by-File Map (Library kaunsi, kyu, kya karti hai — yahan par)

### `model/` — Core Model Code

| File | Lines | Kya hai | Library used | Kyun / Yahan kya kaam |
|---|---|---|---|---|
| `model/utils.py` | 219 | Tokenization + math helpers | `rjieba` (Rust Chinese segmenter), `pypinyin` (tone-marked romanization), `torch` | Text ko model-readable integers mein badalta hai; masking helpers training ke liye |
| `model/modules.py` | 862 | Saare neural building blocks | `torch`, `x_transformers` (RMSNorm import), `torchaudio`, `vocos`, BigVGAN | Attention, RoPE, AdaLN, ConvNeXtV2, FeedForward — ye sab "LEGO blocks" hain jo backbones use karte hain |
| `model/backbones/dit.py` | 371 | Main production architecture | — | Text+audio single-stream Transformer; production model (`EchoForge_v1_Base`) yahi use karta hai |
| `model/backbones/mmdit.py` | 262 | Alternate joint-attention architecture | — | SD3-style — text aur audio ko separate "streams" mein rakhta hai, phir joint attention |
| `model/backbones/unett.py` | 307 | E2-TTS ka U-Net-Transformer | — | Skip-connections wala flat U-Net, "add" ya "concat" mode |
| `model/cfm.py` | 302 | Core algorithm — training loss + inference sampling | `torchdiffeq` (ODE solver) | Rectified Flow Matching — poore system ka "physics engine" |
| `model/trainer.py` | 442 | Training loop orchestration | `accelerate`, `ema_pytorch`, `wandb`, `bitsandbytes` (optional) | Multi-GPU training, checkpointing, logging |
| `model/dataset.py` | 334 | Data loading + batching | HF `datasets`, `torch.utils.data` | Arrow file se PyTorch Dataset, dynamic frame-based batching |

### `infer/` — Inference & Serving

| File | Lines | Kya hai | Library used | Kyun / Yahan kya kaam |
|---|---|---|---|---|
| `infer/utils_infer.py` | 620 | Shared inference pipeline (CLI + Gradio dono use karte hain) | `pydub` (silence detect), `transformers` (Whisper pipeline), `vocos` | Ref audio preprocessing, batched chunked inference, cross-fade |
| `infer/infer_cli.py` | 389 | Command-line inference tool | `tomli` (config), `hydra` | `[voice_name]` tags se multi-voice scripts parse karta hai |
| `infer/infer_gradio.py` | 1227 | Full Web UI (jo tune already restyle kiya) | `gradio` | 4 tabs: Basic TTS, Multi-Speech (emotions), Voice Chat (LLM+TTS), Credits |
| `infer/speech_edit.py` | 236 | Mel-domain speech editing (sirf ek portion re-generate karna) | — | "nature" ko "optimist" se replace karne jaisa demo — poore audio ko regenerate kiye bina |

### `train/` — Training & Fine-tuning

| File | Lines | Kya hai | Library used | Kyun / Yahan kya kaam |
|---|---|---|---|---|
| `train/train.py` | 82 | Scratch-training entry point | `hydra` (YAML config management) | Config file se model+trainer banake `.train()` call karta hai |
| `train/finetune_cli.py` | 214 | Fine-tune entry point (argparse-based) | `cached_path` (HF download) | Pretrained checkpoint download/copy, phir trainer chalata hai |
| `train/finetune_gradio.py` | 1881 | Poori fine-tuning Web UI | `gradio`, `librosa`, `psutil`, `safetensors` | Dataset creation, audio slicing, vocab extension, training subprocess launch, GPU monitoring |
| `train/datasets/prepare_*.py` | ~800 total | 6 alag dataset-specific prep scripts | HF `datasets`, `ProcessPoolExecutor` | Raw dataset → Arrow format |

### `eval/` — Quality Metrics

| File | Kya karta hai | Library |
|---|---|---|
| `eval/utils_eval.py` | Shared WER + SIM logic | `jiwer`, WavLM (via ECAPA) |
| `eval/ecapa_tdnn.py` | Speaker verification model architecture | Standalone PyTorch |
| `eval/eval_seedtts_testset.py` / `eval_librispeech_test_clean.py` | Standard TTS benchmarks pe evaluate karna | Multi-GPU parallel (`multiprocessing`) |
| `eval/eval_utmos.py` | Naturalness MOS score | `torch.hub` (SpeechMOS model) |

### Deployment/Extra

| File | Kya hai |
|---|---|
| `api.py` | Simple `F5TTS` Python class — pura pipeline ek object mein wrap kiya hua |
| `socket_server.py` / `socket_client.py` | Raw TCP socket pe streaming audio chunks bhejna — low-latency real-time use-case ke liye |
| `runtime/triton_trtllm/` | TensorRT-LLM conversion scripts + Triton model repo config — production-scale GPU serving (already justified as untouched) |
| `scripts/count_params_gflops.py`, `count_max_epoch*.py` | Utility scripts — model size aur training-time estimate karne ke liye |

---

## 4. Complete Algorithm Inventory — Rating aur Improvement Suggestion

Har algorithm: **kya hai → kahan hai → meri rating (/10) → kyun → improvement**.
Rating sirf "ye algorithm apne aap mein kitna solid/well-implemented hai" pe hai — "tere use-case ke liye kitna fit hai" alag baat hai (wo Section 6 mein).

### 4.1 — Conditional Flow Matching / Rectified Flow
**Kahan:** `model/cfm.py` (`forward` + `sample`)
**Kya hai:** Diffusion models ka successor — noise se real data tak seedhi line (linear interpolation path) sikhata hai, curved diffusion path ke bajaye. Isse kam steps mein (5-32 vs diffusion ke 50-1000) achha result milta hai.
**Rating: 9/10** — ye field mein state-of-the-art approach hai (2024-25 ka), Stable Diffusion 3, Flux jaisे image models bhi isi family se hain.
**Improvement:** Distillation ya consistency-model training add karke 1-4 step inference tak le ja sakte ho (abhi minimum practical ~5-16 steps hai via EPSS). Real-time Telegram bot ke liye ye latency directly kam karega.

### 4.2 — Classifier-Free Guidance (dual independent dropout)
**Kahan:** `model/cfm.py` forward (`audio_drop_prob=0.3`, `cond_drop_prob=0.2`)
**Kya hai:** Training ke time randomly audio-condition ya text-condition (ya dono) ko "drop" kiya jaata hai, taaki inference ke time model conditional aur unconditional dono prediction de sake, aur unke difference ko amplify (`cfg_strength`) karke sharper output milta hai.
**Rating: 8/10** — VoiceBox paper se liya gaya standard technique, sahi implement hai.
**Improvement:** `cfg_strength` fixed hai per-request; ise per-chunk adaptively tune karna (chhote text pe zyada guidance, lambe pe kam) quality-consistency improve kar sakta hai.

### 4.3 — Euler ODE Solver via `torchdiffeq`
**Kahan:** `model/cfm.py :: sample()`
**Kya hai:** Numeric integration se noise → clean mel path solve karta hai. Default method `"euler"` (simplest), `"midpoint"` bhi option hai.
**Rating: 7/10** — kaam karta hai, but Euler sabse basic/crude ODE solver hai.
**Improvement:** Higher-order solvers (RK4, ya adaptive-step solvers) try karo — kam steps mein better accuracy mil sakti hai, especially agar tu low-NFE (5-8 steps) real-time use-case target kar raha hai.

### 4.4 — Empirically Pruned Step Sampling (EPSS)
**Kahan:** `model/utils.py :: get_epss_timesteps()`
**Kya hai:** Hardcoded lookup table hai (n=5,6,7,10,12,16 steps ke liye) jo non-uniform timestep spacing deta hai — early steps mein bade jumps, baad mein chhote (fine detail) steps.
**Rating: 6/10** — kaam karta hai lekin ye "empirically pruned" (manually tuned by trial) hai, kisi dusre dataset/domain (jaise Hindi voices) ke liye optimal nahi hoga.
**Improvement:** Apne fine-tuned Hindi model ke liye ye timesteps **khud recompute karna chahiye** — ye ek one-time offline search hai (grid search on validation set), fir permanently better low-step quality milegi.

### 4.5 — Sway Sampling
**Kahan:** `model/cfm.py`, formula: `t + coef*(cos(π/2·t) - 1 + t)`
**Kya hai:** ODE timesteps ko ek cosine-based curve se "sway" (shift) karta hai — negative coefficient se early timesteps ki taraf zyada weight milta hai.
**Rating: 7/10** — original F5-TTS paper ka contribution hai, kaam karta hai.
**Improvement:** Koi bada change nahi chahiye, ye already fine-tuned hai.

### 4.6 — Rotary Position Embedding (RoPE)
**Kahan:** `model/modules.py` (`precompute_freqs_cis`, `get_pos_embed_indices`) + `x_transformers.RotaryEmbedding`
**Kya hai:** Position information ko attention ke Q/K vectors ko rotate karke encode karta hai (absolute position embedding ke bajaye).
**Rating: 9/10** — LLM world (LLaMA etc.) ka industry-standard hai, lambi sequences pe achha generalize karta hai.
**Improvement:** Kuch nahi — well-implemented hai, dono text aur audio pe alag-alag RoPE apply ho raha hai jo correct hai.

### 4.7 — Adaptive LayerNorm (AdaLN-Zero)
**Kahan:** `model/modules.py :: AdaLayerNorm`, `AdaLayerNorm_Final`
**Kya hai:** Time-step embedding se normalization ke scale/shift parameters generate hote hain (zero-initialized), taaki training shuru mein model ek identity function ki tarah behave kare aur gradually seekhe.
**Rating: 9/10** — DiT paper (image diffusion) se aaya technique, zero-init training stability ke liye critical hai.

### 4.8 — ConvNeXtV2 Block + GRN (Global Response Normalization)
**Kahan:** `model/modules.py :: ConvNeXtV2Block`, `GRN`
**Kya hai:** Text embeddings pe extra "local context" modeling ke liye depthwise convolution blocks — attention se pehle text ko thoda "smooth" karte hain.
**Rating: 8/10** — solid CNN design choice hai for local feature mixing.

### 4.9 — DiT Backbone (main production model)
**Kahan:** `model/backbones/dit.py`
**Kya hai:** 22-layer Transformer, single-stream (text+audio concatenated), 1024-dim, 16 heads. Optional "long skip connection" (U-ViT style).
**Rating: 8/10** — solid, production-tested (F5-TTS ka main published model).
**Notable:** Isme ek "zipvoice-style average upsampling" feature bhi hai (`average_upsampling` param) jo naye research se liya gaya lagta hai — text tokens ko target audio length ke hisaab se repeat karta hai instead of learned interpolation. Ye default off hai.

### 4.10 — MM-DiT (Joint Attention, Stable-Diffusion-3 style)
**Kahan:** `model/backbones/mmdit.py`
**Rating: 7/10** — architecturally interesting alternative, but is repo mein ye "less battle-tested" lagta hai (koi published checkpoint iske liye dikha nahi in code).
**Improvement:** Agar experiment karna hai to yahi sabse achhi jagah hai naya architecture try karne ke liye, kyunki ye already separate text/audio streams rakhta hai — multilingual conditioning add karna yahan easier hoga.

### 4.11 — UNetT (Flat U-Net Transformer, E2-TTS)
**Kahan:** `model/backbones/unett.py`
**Rating: 7/10** — skip connections (`add` ya `concat` mode) achhi tarah implement hain, `assert len(skips) == 0` jaisa safety check bhi hai.

### 4.12 — Attention: Dual Backend (PyTorch SDPA / FlashAttention)
**Kahan:** `model/modules.py :: AttnProcessor`
**Kya hai:** Runtime pe `attn_backend="torch"` (native `scaled_dot_product_attention`) ya `"flash_attn"` (variable-length, padding-free attention) choose kar sakte ho.
**Rating: 8/10** — flexibility achhi hai, code mein khud warning bhi hai ki `torch` backend + mask enabled = zyada GPU memory.
**Improvement:** Colab/free-tier GPU pe agar `flash-attn` install ho sakta hai, to training/inference dono memory-efficient ho jayenge — worth trying agar OOM issues aa rahe hain.

### 4.13 — Text-to-Phoneme: Pinyin + Jieba (Chinese-specific G2P)
**Kahan:** `model/utils.py :: convert_char_to_pinyin()`
**Rating: 6/10 (Chinese ke liye 9/10, Hindi ke liye 0/10)** — jo bhi hai wo apne scope (Chinese) mein bahut achha hai, lekin **ye "universal" nahi hai** — dusri script ke liye koi provision nahi.
**Improvement:** ⭐ **Sabse high-value improvement tere liye.** Devanagari ke liye ek G2P layer chahiye. Options: (a) simple char-level tokenizer (jo already supported hai as `tokenizer="char"`, no G2P needed — sabse aasan), (b) proper Hindi phonemizer library (jaise `indic_nlp_library` ya espeak-ng Hindi voice) integrate karna for better prosody. (a) se zyada waqt nahi lagega, (b) research-grade result dega.

### 4.14 — ByT5-style UTF-8 Byte Tokenizer
**Kahan:** `model/utils.py :: list_str_to_tensor()`
**Kya hai:** Har character ko uske raw UTF-8 bytes mein todta hai — vocabulary-free approach, kisi bhi language/script ke liye kaam karta hai bina training ke.
**Rating: 7/10** — language-agnostic hone ka fayda hai, lekin Devanagari jaisi complex script (jahan ek character multiple Unicode codepoints se milke banta hai — matra/conjuncts) byte-level mein bikhar sakta hai aur model ko zyada seekhna padega.

### 4.15 — Dynamic Frame-Based Batch Sampler
**Kahan:** `model/dataset.py :: DynamicBatchSampler`
**Kya hai:** Samples ko duration ke hisaab se sort karke, ek "frame budget" (total audio length) ke andar jitne samples fit ho jaate hain utna ek batch banata hai — chhote clips ka batch bada, lambe clips ka chhota.
**Rating: 9/10** — bahut efficient technique hai, padding waste minimize karti hai, reproducible shuffling (seed+epoch based) bhi hai.

### 4.16 — EMA (Exponential Moving Average) Weights
**Kahan:** `model/trainer.py`, `ema_pytorch` library
**Rating: 9/10** — standard best-practice, inference mein hamesha EMA weights use hote hain (`use_ema=True` default) jo raw training weights se zyada stable output dete hain.

### 4.17 — Linear Warmup + Linear Decay LR Schedule
**Kahan:** `model/trainer.py` (`SequentialLR` = `LinearLR` + `LinearLR`)
**Rating: 7/10** — simple aur reliable, lekin modern alternatives (cosine decay, ya warmup+constant+decay) kabhi-kabhi thoda better converge karte hain.
**Improvement:** Fine-tuning ke liye (jahan tu already kam updates kar raha hoga), cosine schedule try karna easy experiment hai.

### 4.18 — Gradient Accumulation + Clipping
**Kahan:** `model/trainer.py` (`accelerator.accumulate`, `clip_grad_norm_`)
**Rating: 8/10** — sahi se implemented, chhote-GPU (Colab) training ke liye essential hai.

### 4.19 — Vocabulary/Embedding Expansion for Fine-tuning
**Kahan:** `train/finetune_gradio.py :: expand_model_embeddings()`, `vocab_extend()`
**Kya hai:** Agar naya language/symbols add karne hain jo original vocab mein nahi the, ye function pretrained embedding table ko badi bana deta hai — purane tokens ke weights preserve, naye tokens ko random-init karta hai.
**Rating: 8/10** — ye exactly wahi mechanism hai jo tujhe **Hindi ke liye chahiye hoga** agar tu "char" tokenizer se naya vocab banata hai.
**Improvement:** Random init ke bajaye, naye tokens ko phonetically/visually similar existing tokens ke embeddings se initialize karna (nearest-neighbor init) convergence tez kar sakta hai — abhi pure `torch.randn()` use ho raha hai.

### 4.20 — Audio Silence Slicer (RMS-based, GPT-SoVITS derived)
**Kahan:** `train/finetune_gradio.py :: Slicer` class
**Kya hai:** RMS energy ke through silence detect karke lambi recordings ko automatically chhote utterances mein todta hai — dataset creation ke liye.
**Rating: 8/10** — well-tested algorithm (RVC-Boss/GPT-SoVITS se borrowed, credit bhi comment mein hai), production-grade hai.

### 4.21 — Reference-Audio Silence Trimming + 12s Clip Cap
**Kahan:** `infer/utils_infer.py :: preprocess_ref_audio_text()`
**Kya hai:** Do-pass silence detection (pehle strict, phir loose threshold) se reference audio ko max 12 seconds tak trim karta hai — zyada lamba reference audio quality/speed dono kharab karta hai.
**Rating: 7/10** — practical heuristic hai, kaam karti hai. Result MD5-hash se cache hota hai (same file dobara process nahi hota).

### 4.22 — Chunked Parallel Inference (ThreadPoolExecutor)
**Kahan:** `infer/utils_infer.py :: infer_batch_process()`
**Kya hai:** Lamba text sentence-boundary pe chunks mein todta hai, phir sab chunks **parallel threads** mein generate hote hain (GPU pe queue ho jaate hain, lekin Python-side dispatch overhead kam hota hai).
**Rating: 7/10** — kaam karta hai, lekin GIL ki wajah se real parallelism sirf I/O/GPU-wait ke time hi milta hai, CPU-bound preprocessing serialize hi rahega.
**Improvement:** Agar batch dimension pe hi model ko ek saath multiple chunks feed kar do (true batched GPU inference, thread ke bajaye), throughput better ho sakta hai bade texts ke liye.

### 4.23 — Cross-fade Waveform Stitching
**Kahan:** `infer/utils_infer.py`, linear fade in/out (0.15s default)
**Rating: 7/10** — simple linear cross-fade, audible clicks avoid karta hai. Equal-power (cosine) cross-fade thoda smoother sunayi de sakta hai linear ke bajaye — minor upgrade.

### 4.24 — Mel-Domain Speech Editing / Infilling
**Kahan:** `infer/speech_edit.py`
**Kya hai:** Poore audio ko regenerate kiye bina, sirf specific time-ranges (frame-level precision) ko "zero out" karke naye text se re-generate karta hai — CFM ka masking mechanism training ke corresponding hai.
**Rating: 8/10** — clever reuse hai training-time masking ka inference-time editing ke liye. Requires external forced-aligner (ctc-forced-aligner) for word-timing — ye repo mein bundled nahi hai, manual step hai.

### 4.25 — Auto Batch-Size / Epoch Calculator
**Kahan:** `train/finetune_gradio.py :: calculate_train()`
**Kya hai:** GPU memory dekh ke automatically batch-size, epochs, warmup-updates suggest karta hai (heuristic formula: `38400 * (avg_gpu_memory - 5) / 75`).
**Rating: 5/10** — ye ek **hardcoded magic-number heuristic** hai, kisi specific GPU/dataset pe tune kiya gaya lagta hai. Chhoti Colab GPU (T4, 15GB) ya bade Hindi dataset pe accurate estimate nahi dega.
**Improvement:** Isse sirf ek "starting point suggestion" maan, khud manually monitor kar (OOM aaye to batch size kam kar) — blindly is calculator pe depend mat kar.

### 4.26 — WER Evaluation (ASR round-trip)
**Kahan:** `eval/utils_eval.py :: run_asr_wer()`
**Kya hai:** Generated audio ko wapas Whisper (English) ya FunASR Paraformer (Chinese) se transcribe karke, original text se `jiwer` library se Word Error Rate nikalta hai.
**Rating: 8/10** — industry-standard TTS evaluation approach.
**Note:** Hindi ke liye `lang` parameter mein sirf `"zh"` aur `"en"` supported hain code mein — Hindi WER eval ke liye ek third branch add karna padega (Whisper multilingual mode use karke, jo already Hindi support karta hai).

### 4.27 — Speaker Similarity (WavLM + ECAPA-TDNN)
**Kahan:** `eval/utils_eval.py :: run_sim()`, `eval/ecapa_tdnn.py`
**Kya hai:** Dono audios (generated + reference) ko ek pretrained speaker-embedding model se pass karke, unke embeddings ka cosine similarity nikalta hai — "kya ye same voice lag rahi hai" ka objective score.
**Rating: 9/10** — ye gold-standard technique hai voice cloning quality measure karne ke liye, language-agnostic hai (Hindi voices pe bhi bina modification ke chalega).

### 4.28 — UTMOS Naturalness Score
**Kahan:** `eval/eval_utmos.py`
**Kya hai:** Ek pretrained model jo predict karta hai "insaan is audio ko kitna natural rate karenge" (1-5 scale), bina human listeners ke.
**Rating: 7/10** — useful proxy metric, lekin ye English/Japanese data pe trained model hai (SpeechMOS) — Hindi audio pe iske scores literal MOS ke barabar reliable nahi honge, sirf relative comparison (before/after fine-tune) ke liye trust karna.

### 4.29 — `lru_cache` on Gradio `infer()`
**Kahan:** `infer/infer_gradio.py`
**Kya hai:** Same exact parameters (ref audio path, text, seed, etc.) se dobara request aaye to cached result turant return ho jaata hai.
**Rating: 6/10** — demo/UI ke liye theek hai, lekin **production risk** hai (Section 5 mein detail).

### 4.30 — Raw TCP Socket Streaming Protocol
**Kahan:** `socket_server.py`, `socket_client.py`
**Kya hai:** Custom binary protocol — text bhejo, audio chunks (`struct.pack` floats) wapas stream hoke aate hain, `b"END"` se stream khatam hone ka signal.
**Rating: 6/10** — kaam karega, lekin bahut "raw" hai — koi authentication, encryption, ya reconnection handling nahi (Section 5 mein detail).


---

## 5. Backbone Health Check — "Insaan ki Backbone Damage" Wale Angle Se

Tune analogy di thi ki insaan ki backbone damage ho to poora sharir kaam ka nahi rehta — to yeh section wahi karta hai: **jo cheezein poore system ko production mein tod sakti hain**, unki list. Ye "improvement suggestions" nahi hain (wo Section 6 mein hain) — ye actual **risk/gap findings** hain jo maine is audit mein pehli baar note ki.

| # | Kahan | Kya risk hai | Kitna serious | Kyun matter karta hai tere case mein |
|---|---|---|---|---|
| 1 | `train/finetune_gradio.py :: start_training()` | Training command `subprocess.Popen(cmd, shell=True)` se launch hoti hai — `cmd` ek f-string hai jisme `dataset_name`, `exp_name` jaise fields directly interpolate hote hain | 🟠 Medium | Agar kabhi project-name field user-facing input bane (jaise multi-user Telegram bot backend), shell injection possible hai. Abhi single-user Colab/Replit context mein low risk hai, lekin agar isse multi-tenant service banata hai to fix zaroori |
| 2 | `socket_server.py :: start_server()` | `0.0.0.0` pe bind, **koi authentication/encryption nahi**, koi rate-limiting nahi | 🔴 High (agar public-facing deploy kare) | Koi bhi jo tere server ka IP+port jaanta hai, free mein TTS generate kar sakta hai (compute abuse) ya tera GPU exhaust kar sakta hai. Agar Render/cloud pe expose karega to firewall/VPN ke peeche rakhna zaroori |
| 3 | `infer/infer_gradio.py :: infer()` | `@lru_cache(maxsize=1000)` — cache kabhi clear nahi hota, aur agar model checkpoint hot-reload kare (jaise fine-tune ke baad same session mein) to **purana cached audio wapas mil sakta hai** | 🟡 Low-Medium | Demo ke liye fine, lekin agar tu isko live product banata hai jahan model beech mein switch/update hota hai, cache invalidation add karna padega |
| 4 | `model/utils.py :: convert_char_to_pinyin` | Sirf Chinese G2P hardcoded — Hindi/Devanagari ke liye koi phoneme normalization nahi | 🟠 Medium | Tera core product goal (Hindi TTS) is gap se directly affect hota hai — already Section 4.13 mein detail hai |
| 5 | `train/finetune_gradio.py :: calculate_train()` | GPU-memory-based batch-size heuristic mein hardcoded magic numbers (`38400`, `75`, `-5`) | 🟡 Low | Colab T4/A100 alag-alag hardware pe wrong estimate deke OOM crash ya bahut slow training de sakta hai |
| 6 | `infer/utils_infer.py` | `asr_pipe` aur `_ref_audio_cache`/`_ref_text_cache` **module-level global variables** hain, thread-safe nahi | 🟡 Low-Medium | Agar Gradio/Flask multiple concurrent requests handle kare (jaise tera Telegram bot ek saath kai users serve kare), race condition se galat cached ref-audio kisi aur user ko mil sakta hai |
| 7 | `model/trainer.py` | `duration_predictor` parameter accept karta hai lekin actual training loop mein sirf ek `# TODO. add duration predictor training` comment hai — feature stubbed hai, kaam nahi karta | 🟢 Info only | Isse "missing feature" samajh, bug nahi — agar future mein explicit duration control chahiye, ye complete karna padega |
| 8 | Poore codebase mein | Koi centralized **audio watermarking / provenance marking** nahi hai generated output pe | 🟡 Low (abhi), 🟠 Medium (jab product ban jaye) | Voice-cloning product ko commercially bechte waqt "AI-generated" disclosure/watermark ek responsible-AI aur kai jagah legal requirement ban raha hai (EU AI Act jaisी regulations) |

**Note:** #1 aur #2 ko main "backbone fracture" level ki cheez kahunga agar tu isko multi-user service banata hai — abhi tere solo-dev Colab workflow mein inka practical impact zero hai, lekin jis din ye "PR Bot Services" ka live product bane, ye do items pehle fix karne chahiye.


---

## 6. Suggestions Block — Sirf Possibilities (Naye Features/Algorithms/Workflow Angle Se)

⚠️ **Ye block "karna hai" wali list nahi hai — sirf "kya-kya kar sakte hain" wali list hai**, jaisa tune specifically maanga tha. Koi priority order nahi, koi commitment nahi — sirf exploration ke liye options.

### 6.1 — Language & Tokenization
- **Devanagari-native G2P layer** — Hindi text ko phonemes mein todne ke liye `indic_nlp_library` ya `epitran` jaisi library integrate karna, taaki model matra/conjuncts ko sahi se samjhe (abhi ka gap, Section 4.13)
- **Code-switch detection** — tera Hinglish communication style dekhte hue, ek auto-detector jo ek hi sentence mein Hindi aur English words ko alag-alag pronounce kare (abhi sab ek hi tokenizer se guzarta hai)
- **Multi-script vocab** — ek hi model mein Devanagari + Latin dono ko support karne wala combined vocab.txt (jaise Emilia_ZH_EN ka combined ZH+EN vocab hai, waisa hi Hindi+English ke liye ban sakta hai)

### 6.2 — Speed / Latency
- **Step distillation** — abhi 5-32 ODE steps lagte hain; consistency-model ya progressive-distillation training se 1-4 steps tak la sakte ho, real-time Telegram bot ke liye game-changer
- **ONNX / TensorRT export for the DiT backbone specifically** (abhi Triton path sirf existing F5-TTS ke liye hai) — CPU-only ya edge deployment ke liye
- **KV-caching across streaming chunks** — abhi har chunk independently generate hota hai; agar consecutive chunks ke beech context share ho, prosody continuity better ho sakti hai

### 6.3 — Voice & Personalization
- **Fixed speaker-embedding table** — agar ShreeGen bot ke liye 5-10 "persona voices" fix karni hain, unke embeddings ko ek chhoti lookup table mein precompute karke rakhna (bar-bar reference audio process karne ke bajaye) — latency aur consistency dono improve
- **LoRA/PEFT-based fine-tuning** — poora model fine-tune karne ke bajaye, sirf ek chhota adapter train karna — Colab/free-GPU ke liye kaafi zyada practical, aur multiple voices ko alag-alag chhoti LoRA files mein rakh sakte ho (base model shared)
- **Emotion/prosody control tokens** — text mein `{happy}`, `{sad}` jaise tags (jo `app_multistyle` mein already partially exist karte hain) ko aur structured banake ek proper conditioning signal banana

### 6.4 — Product/Integration (ShreeGen ke sath connection)
- **Voice-chat pipeline reuse** — `infer_gradio.py`'s `app_chat` (jo Qwen/Phi local LLM use karta hai) ka structure copy karke, usme apna ShreeGen/Gemini pipeline plug karna — matlab ek **voice-in, voice-out Telegram bot** jo already-existing chat-brain ko use kare
- **Multi-character narration** — `app_multistyle` ka JSON-tag system CineVerse Hub ke content ke liye reuse ho sakta hai (movie/story narration jahan alag characters ki alag voice ho)
- **Batch queue system** — Telegram bot jaise multi-user context ke liye ek proper job-queue (Redis/Celery jaisा kuch) jo raw `ThreadPoolExecutor` ke bajaye zyada robust ho concurrent requests ke liye

### 6.5 — Evaluation & Quality
- **Hindi WER branch** — `eval/utils_eval.py`'s `run_asr_wer()` mein Whisper multilingual mode se Hindi support add karna (already Whisper Hindi karta hai, bas ek `elif lang == "hi"` branch chahiye)
- **A/B listening test harness** — do model versions (base vs fine-tuned) ke output side-by-side compare karne ka simple Gradio tool banana

### 6.6 — Responsible AI / Compliance
- **Audio watermarking** — generated audio mein ek inaudible watermark embed karna (jaise Meta ka AudioSeal, ya simple approach) taaki AI-generated content traceable rahe — agar product commercially scale karta hai to ye future-proofing hai
- **Usage disclosure metadata** — generated `.wav` files ke sath ek metadata tag (ID3/vorbis comment) jo bataye "EchoForge AI se generate hua"


---

## 7. Closing — Seedha Reality Check (Jaisa Maine Shuru Mein Bola Tha)

Kuch cheezein clear kar deta hu bina sugarcoat ke, jaisa tune khud maanga hai:

1. **"Apna engine" banane ka matlab kya hona chahiye — ek honest definition:** Iss codebase ko fully samajh lena (jo tu abhi kar raha hai) + isko Hindi/Hinglish data pe fine-tune kar lena = ek **legitimately customized/specialized product**, jo research aur engineering ke hisaab se real kaam hai. Lekin ye "F5-TTS se completely independent apna architecture" nahi hai — wo architecture research (naya paper-level contribution) hai, alag scale ka kaam hai. In dono cheezon mein confuse mat ho — donon valid goals hain, par different hain.

2. **Sabse practical agla step** (agar tu poochta to main yahi kehta): Section 4.13 wala Hindi tokenizer gap fix karo pehle, chhote dataset (even 1-2 hours ki apni recorded Hindi voice) pe `char` tokenizer se ek fine-tune try karo. Ye tujhe turant pata chal jayega ki backbone Hindi ke sath kaisa behave karta hai — bina bade architecture-level kaam ke.

3. **Documentation vs Training — dono alag kaam hain.** Ye document tujhe "engine kaise kaam karti hai" samjhata hai. Actual fine-tuning shuru karne ke liye tujhe dataset chahiye (Hindi audio+text pairs), GPU time chahiye, aur patience chahiye (epochs mein hours-days lagte hain even chhote dataset pe). Ye document us kaam ko easy nahi banata — sirf informed banata hai.

4. **Deep UI/UX redesign jo pending hai** — wo alag track hai, is document se independent. Jab bhi chaho wapas us pe focus kar sakte hain.

---

## Quick Reference — Kis Sawaal Ka Jawab Kahan Milega

| Sawaal | Section |
|---|---|
| Poora system kaise fit baithta hai ek dusre ke saath? | Section 1 |
| Training se lekar audio-output tak step-by-step kya hota hai? | Section 2 |
| Kis file mein kya hai, kaunsi library kyun use hui? | Section 3 |
| Har algorithm kitna achha hai, kya improve ho sakta hai? | Section 4 |
| Kya production mein tootne wali cheezein hain? | Section 5 |
| Naye features/ideas kya try kar sakte hain (sirf possibilities)? | Section 6 |
| Honest big-picture assessment? | Section 7 |

---
*Document compiled by Claude — EchoForge AI internal architecture audit, based on full read of `src/echoforge_tts/` (~40 files, ~10,400 lines) uploaded 16 July 2026.*
