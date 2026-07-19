# EchoForge AI — Fix Missed File: `infer/infer_cli.py`

## Context
Poori-tarah-Raon-1B-only-migration ke pichhle-round mein **`infer/infer_cli.py` accidentally-miss ho gayi** — is file mein abhi bhi poora-purana-330M/E2TTS-logic hai, jabki baaki-saari-files (`api.py`, `finetune_gradio.py`, `finetune_cli.py`, `infer_gradio.py`, `speech_edit.py`, `socket_server.py`) already-successfully update ho chuki hain (verified). Ye same-pattern hai jo `api.py` mein tha — isliye same-tarike se fix karna hai.

**Confirmed via direct code-inspection — exact lines jo fix karni hain:**

---

## Kya karna hai

### 1. Help-text update karo (line ~55)
```python
help="The model name: EchoForge_v1_Base | EchoForge_Base | E2TTS_Base | etc.",
```
Isko badal ke sirf `RaonOpenTTS_1B` ka reference rakho:
```python
help="The model name: RaonOpenTTS_1B",
```

### 2. Default-model badlo (line ~187)
```python
model = args.model or config.get("model", "EchoForge_v1_Base")
```
→
```python
model = args.model or config.get("model", "RaonOpenTTS_1B")
```

### 3. Poora model-selection/checkpoint-download-block replace karo (lines ~274-291)

Current (purana):
```python
repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "EchoForge_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "EchoForge_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "EchoForge_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
elif ckpt_file.startswith("hf://"):
    ckpt_file = str(cached_path(ckpt_file))
```

Ye poora-block replace karo — `api.py` mein jo pattern already-apply ho chuka hai (verified: `repo_name="KRAFTON/Raon-OpenTTS-1B"`, `ckpt_step=520000`, `ckpt_type="pt"`), **wahi-consistent-values** yahan bhi use karo:

```python
repo_name, ckpt_step, ckpt_type = "KRAFTON/Raon-OpenTTS-1B", 520000, "pt"

assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://{repo_name}/model_{ckpt_step}.{ckpt_type}"))
elif ckpt_file.startswith("hf://"):
    ckpt_file = str(cached_path(ckpt_file))
```

**Zaroori-note:** Purana-URL-format tha `hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}` (jisme `{model}` bhi path-mein-tha, jaise `F5TTS_v1_Base/model_1250000.safetensors`). Naya-Raon-checkpoint ka actual-file-structure **flat hai** (humne khud verify kiya tha download karte-waqt — `model_520000.pt` seedha repo-root mein hai, koi sub-folder nahi). Isliye naya-URL-format `hf://{repo_name}/model_{ckpt_step}.{ckpt_type}` hona chahiye, **`{model}`-path-segment hatana hai** — is farak ko zaroor-dhyan-mein-rakhna, warna phir 404 aayega jaisa humne pehle-round mein dekha tha.

### 4. `vocoder_name`/`mel_spec_type` default verify karo (line ~215 aur uske upar-import)
```python
vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
```
`mel_spec_type` `utils_infer.py` se import ho raha hai — confirm karo ye variable ab Raon-1B ke `sbhifigan16k` ki taraf point karta hai ya purane-`vocos`-default ki taraf. Agar `utils_infer.py` mein ye already-globally-updated hai (jo pichhle-round mein hone ki possibility hai), to yahan extra-kuch nahi karna. Agar nahi, to yahan explicit-default `"sbhifigan16k"` set karo taaki CLI bina-explicit-flag-diye bhi sahi-vocoder-use kare.

### 5. Docstring/prog-description update karo (line ~34-36)
```python
description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
```
→
```python
description="Commandline interface for EchoForge (Raon-OpenTTS-1B) TTS with Advanced Batch Processing.",
```

---

## Verification Checklist (Replit Agent khud confirm kare)

1. `grep -n "SWivid\|E2TTS\|E2-TTS\|EchoForge_v1_Base\|EchoForge_Base" src/echoforge_tts/infer/infer_cli.py` — zero-results aane chahiye
2. `python -m echoforge_tts.infer.infer_cli --help` chalake confirm karo help-text sirf `RaonOpenTTS_1B` batata hai
3. Bina kisi `--model`/`--ckpt_file` flag ke CLI chalane-par (default-config se), checkpoint-download-URL `hf://KRAFTON/Raon-OpenTTS-1B/model_520000.pt` bane — koi `SWivid`/`F5TTS_v1_Base`-path na bane
4. Poore-codebase-mein final-grep (triton ke alawa) — ab **sirf harmless-comments** (jaise `modules.py`, `finetune_gradio.py`, `utils_infer.py`, `RaonOpenTTS_1B.yaml` mein jo purane-round mein already-verified-harmless the) bachne-chahiye, koi functional-code-path nahi

---

## Explicitly Out of Scope
- `runtime/triton_trtllm/` — hamesha-ki-tarah scope se bahar
- Baaki-saari-files (`api.py`, `finetune_gradio.py`, etc.) — already-verified-correct pichhle-round mein, inhe touch mat karo
