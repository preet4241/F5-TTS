# EchoForge AI — Base-Model Switch: F5-TTS (330M) → Raon-OpenTTS-1B

## Goal
Current base-model (F5-TTS `EchoForge_v1_Base`, 330M params) ko replace karna hai **Raon-OpenTTS-1B** se — jo F5-TTS ke exact DiT architecture pe based hai (author-confirmed, code unmodified), sirf scaled-up hai (1.05B params), aur already 510K-hours English speech pe pretrained hai. Humein isko scratch-train NAHI karna — sirf isse base bana ke baad mein Hindi-data pe finetune karna hai (agla phase).

**Source of truth for all values below:** https://github.com/krafton-ai/Raon-OpenTTS (official Model Zoo table) + https://arxiv.org/abs/2605.20830 (paper).

---

## VERIFIED VALUES — CONFIRMED from actual `1b.yaml` config file (user-provided, direct source)

Ye poori config **user ne khud `1b.yaml` se copy karke di hai** — ye ab authoritative source hai, GitHub README se bhi zyada exact:

```yaml
model:
  name: 1B
  tokenizer: custom                              # NOT "char" — uses a fixed vocab.txt file path
  tokenizer_path: src/f5_tts/data/vocab.txt      # their internal path — WE WILL OVERRIDE this to point at OUR VOCAB_PATH (Hindi+English phonemizer vocab), see "Important Note" below
  backbone: DiT
  arch:
    dim: 1408
    depth: 28
    heads: 24
    ff_mult: 4
    text_dim: 512
    text_mask_padding: True
    qk_norm: null                    # same as F5-TTS default — no change needed
    conv_layers: 4                   # same as F5-TTS default — no change needed
    pe_attn_head: null               # same as F5-TTS default — no change needed
    attn_mask_enabled: False
    checkpoint_activations: False    # can be flipped to True later for VRAM savings, per earlier discussion
    logit_softcapping: null          # NEW field, not present in our current EchoForge_v1_Base.yaml — add it
    post_norm: False                 # NEW field, not present in our current config — add it
    norm_type: rmsnorm               # NEW field, not present in our current config — add it
  mel_spec:
    target_sample_rate: 16000
    n_mel_channels: 80
    hop_length: 256
    win_length: 1024                 # same as F5-TTS default
    n_fft: 1024                      # same as F5-TTS default
    mel_spec_type: sbhifigan16k      # SPECIFIC custom name — do NOT write generic "hifigan", must be exactly this string
  vocoder:
    is_local: True                   # differs from F5-TTS's False — expects a local checkpoint file, not auto-download from HF
    local_path: null                 # value is null in the config itself — actual path likely set via CLI arg or env var at runtime; NOT YET CONFIRMED where/how this gets populated
  checkpoint:
    huggingface_repo: "KRAFTON/Raon-OpenTTS-1B"
    total_params: "1048M (~1.05B)"
  inference:
    nfe_steps: 32
```

Additional confirmed training-reference config (not needed for our finetune, included for context only):
```yaml
optim:
  learning_rate: 1e-4
  num_warmup_updates: 50000
  max_updates: 1000000
  batch_size_per_gpu: 14000
  batch_size_type: frame
```

---

## STILL UNVERIFIED — DO NOT GUESS, LEAVE BLANK AND ASK

1. **`vocoder.local_path` — actual runtime value.** The config shows `null`, meaning this is populated some other way (CLI flag, env var, or a code-level default elsewhere in the repo). 
   → **Where to find it:** Check `src/f5_tts/infer/infer_cli.py` or wherever the vocoder is loaded in the Raon-OpenTTS repo, specifically for how `local_path` gets set when `is_local: True`. The README's vocoder-download step saves to `pretrained_models/generator.ckpt` — but whether the code auto-detects this path or needs an explicit flag is not confirmed. Bring back this specific file/code section and I will provide the exact value.

2. **`datasets.hf_repo` naming discrepancy.** The GitHub README's badges/links point to `KRAFTON/Raon-OpenTTS-Pool` (capital O, hyphenated), but the actual `1b.yaml` config the user provided says `KRAFTON/RAON-TTS-Pool` (all-caps RAON, no "Open"). This is NOT relevant to our finetuning (we use our own Hindi dataset, not their pool), but flagging it in case any code path references this string — **do not silently pick one; if this string is needed anywhere in our integration, ask.**

**IMPORTANT: This step (base-model architecture switch) does NOT require the training-data pipeline (`datasets:` section, `hf_repo`, etc.) at all — we are not training on their data. Only the `model:` section (architecture) and `vocoder:` section (for inference/generation) are relevant here.**

---

## Kya files touch karni hain — is codebase mein

### File 1: Naya config file banao
`src/echoforge_tts/configs/RaonOpenTTS_1B.yaml` naam se ek naya config banao (purane `EchoForge_v1_Base.yaml` ko as-is rehne do, delete mat karo — reference/rollback ke liye). Structure `EchoForge_v1_Base.yaml` jaisa hi rakho, sirf upar diye "VERIFIED VALUES" daalo jahan-jahan confirm hain, aur jo "COULD NOT VERIFY" hain unko `# TODO: unconfirmed, see prompt notes` comment ke saath current-F5-TTS-default pe hi chhod do (change mat karo jab tak confirm na ho).

### File 2: Checkpoint-download path
Jahan bhi model checkpoint HuggingFace se download hota hai (`infer_gradio.py` mein jaisa `hf://SWivid/F5-TTS/...` pattern tha), wahan naya path add karo `hf://KRAFTON/Raon-OpenTTS-1B/...` ke liye. **Exact filename (jaise `model_last.pt` ya `.safetensors` naam) README mein `model_last.pt` mention hua hai** — ye confirm hai, use karo.

### File 3: Vocoder integration — NAYA COMPONENT
Abhi codebase Vocos/BigVGAN use karta hai (`mel_spec_type: vocos | bigvgan`). Raon-OpenTTS ko **HiFi-GAN** chahiye, aur config-file confirm karti hai iska exact internal naam **`sbhifigan16k`** hai (NOT generic "hifigan" — is exact string ka use karna hai jahan bhi `mel_spec_type` set ho raha hai). Ye abhi codebase mein exist nahi karta:
- Vocoder-loading code mein ek naya option add karo `mel_spec_type: sbhifigan16k`
- `vocoder.is_local: True` hai (F5-TTS ke `False` se ulta) — matlab ye HuggingFace se auto-download expect nahi karta, local checkpoint-file chahiye
- Download-instruction README se confirm hai: `speechbrain/tts-hifigan-libritts-16kHz` se `generator.ckpt` chahiye, jo `pretrained_models/` folder mein rakha jaata hai unke setup mein
- `vocoder.local_path` config mein `null` hai — **exact runtime-path kahan se aata hai ye abhi unverified hai** (upar "STILL UNVERIFIED" section point 1 dekho) — is field ko hardcode mat karo jab tak confirm na ho, filhaal ek clearly-marked placeholder path use karo (jaise `pretrained_models/generator.ckpt`, README ke download-instruction se match karta huya) aur comment mein likho ki ye assumption hai, confirmed nahi
- **Ye ek naya vocoder-wrapper likhna padega** — abhi ka codebase directly HiFi-GAN support nahi karta, isliye ye sabse bada naya-code wala hissa hai is switch mein

### File 4: Sample-rate cascade-check
`target_sample_rate: 16000` set karne se poori pipeline mein jahan bhi `24000` hardcoded hai (audio-loading, mel-extraction, dataset-prep), sab jagah verify/update karo. Grep karo `24000` ke liye poore codebase mein aur har jagah check karo ki ye Raon-OpenTTS path ke liye 16000 use ho raha hai — lekin **purana F5-TTS/EchoForge_v1_Base config abhi bhi 24000 use karega**, dono config independently sahi rehne chahiye (ek-doosre ko break na karein).

### File 5: Vocab-handling — IMPORTANT NOTE, koi confusion na ho
Confirmed config batati hai unka `tokenizer: custom` hai (NOT "char" jaisa maine pehle socha tha) — matlab wo bhi ek fixed vocab.txt-file-path use karte hain (`tokenizer_path: src/f5_tts/data/vocab.txt`), bilkul jaisa humara apna `VOCAB_PATH` system hai. Ye actually **fayda ki baat hai** — humein bas humare config mein `tokenizer: custom` aur `tokenizer_path` ko humare apne `VOCAB_PATH` (Hindi+English phonemizer vocab) ki taraf point karna hai.

Unka original vocab-size 5,512 hai (unka English character-set, `src/f5_tts/data/vocab.txt` unke repo mein). **Humein ye vocab use NAHI karni** — humara Hindi+English phonemizer-vocab (jo humne Phase 1-2 mein bana chuka hai) hi use hoga. Iska matlab:
- Checkpoint load karte waqt, model ka embedding-layer **humare vocab-size ke hisaab se resize/extend** karna padega (jaisa humne "Phase 2: Vocab Extension" mein already discuss kiya tha — `expand_model_embeddings()` jaisa concept, lekin ab 1B checkpoint ke upar)
- Ye is naye-step ka sabse technically-sensitive hissa hai — checkpoint ke text-embedding weights sirf unke 5,512-vocab ke liye trained hain, humare Hindi-tokens ke liye naye/random embeddings honge (jaisa humne "model growth" discussion mein cover kiya tha)

---

## Verification Checklist (Replit Agent khud confirm kare)

1. Naya config-file (`RaonOpenTTS_1B.yaml`) banaya gaya, purana `EchoForge_v1_Base.yaml` untouched raha
2. Jo values maine "VERIFIED" mein di hain wahi exact daali gayi hain — koi rounding/approximation nahi
3. Jo "COULD NOT VERIFY" hain, unke liye seedha `1b.yaml` config-file (upar diya GitHub link) fetch karne ki koshish ki gayi — agar successful, to real values use hui; agar fail, to explicitly TODO-comment ke saath chhoda gaya aur mujhe report kiya gaya kaunsi values missing hain
4. HiFi-GAN vocoder-wrapper naya likha gaya (kyunki pehle exist nahi karta tha), aur ye purane Vocos/BigVGAN wrapper ko break nahi karta
5. Sample-rate 16000 sirf Raon-OpenTTS path mein apply hua, purana F5-TTS 24000-path unaffected raha
6. Checkpoint-download path (`hf://KRAFTON/Raon-OpenTTS-1B/model_last.pt`) sahi se wire hua
7. Vocab-extension logic (Phase 2 se) naye 1B-checkpoint ke embedding-layer ke saath compatible hai — ye khaas taur pe test karna zaroori hai kyunki shape-mismatch yahan sabse zyada likely hai

---

## Explicitly Out of Scope for this step

- Actual Hindi-finetuning (agla phase hai, is step mein sirf architecture-switch karna hai)
- Scratch-training kisi bhi tarah se — hum sirf pretrained-checkpoint use kar rahe hain
- `runtime/triton_trtllm/` — pehle jaisa hi, is scope se bahar
