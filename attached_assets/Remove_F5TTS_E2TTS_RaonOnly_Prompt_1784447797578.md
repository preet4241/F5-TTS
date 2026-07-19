# EchoForge AI — Complete Removal of F5-TTS/SWivid + E2TTS, Raon-1B-Only

## Goal
EchoForge ab **poori tarah Raon-OpenTTS-1B-based hona chahiye**. Koi bhi purana F5-TTS/SWivid (330M `EchoForge_v1_Base`, `EchoForge_Base`) ya E2TTS reference — chahe checkpoint-download-URL ho, config-file ho, UI-dropdown-choice ho, ya training-flow ho — **sab hataana hai**. Sirf `RaonOpenTTS_1B` bachna chahiye default aur sirf-option ke taur pe.

**Temporary/current state (important context for Replit Agent):** Abhi checkpoint seedha `KRAFTON/Raon-OpenTTS-1B` (HuggingFace) se download hoga — ye external-source hai. **Baad mein** ye checkpoint apne naam se re-upload hoga (naya HuggingFace-repo, EchoForge ke apne account se) — is fix mein sirf ek jagah (config-level) `KRAFTON/Raon-OpenTTS-1B` ka reference hardcode karo taaki baad mein sirf-ek-jagah badal ke naya-repo point karna aasan ho, poore-codebase mein bikhra hua na ho.

---

## Confirmed scope — ye files touch karni hain (verified via codebase-grep, guess nahi)

### File 1: `api.py` — MODIFY (delete nahi, ye ek reusable public-API-class hai)
- `class F5TTS.__init__()` mein `model="EchoForge_v1_Base"` default ko `model="RaonOpenTTS_1B"` karo
- Line ~65: `repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"` — ye hardcoded-default hai jo purane-model ko assume karta hai. Isko `RaonOpenTTS_1B`-specific values se replace karo (`repo_name="KRAFTON/Raon-OpenTTS-1B"`, `ckpt_step=520000`, `ckpt_type="pt"` — jaisa humne verify kiya, actual-filename `model_520000.pt` hai)
- `elif model == "EchoForge_Base":` aur `elif model == "E2TTS_Base":` conditional-branches (lines ~68-76) — **poori tarah hatao**, koi replacement-branch nahi chahiye kyunki ab sirf ek-hi model-path hai
- **Class ka naam `F5TTS` hi rakhna hai ya `EchoForge` karna hai — ye decide karna hai:** class-naam `F5TTS` hai jo ab misleading hai (product ka naam EchoForge hai, F5-TTS nahi). Class ka naam `EchoForge` karo, lekin agar koi backward-compatibility-concern ho (jaise koi existing-code `from echoforge_tts.api import F5TTS` use kar raha ho), ek alias rakho: `F5TTS = EchoForge` (deprecated-alias, taaki purana-import-path na tooте)

### File 2: `train/finetune_gradio.py` — MODIFY
- Line ~1119-1123: `EchoForge_v1_Base`/`EchoForge_Base`/`E2TTS_Base` ke liye checkpoint-download conditional-branches — sirf `RaonOpenTTS_1B` ka branch rakho (`hf://KRAFTON/Raon-OpenTTS-1B/model_520000.pt`), baaki hatao
- Sabhi UI-dropdown `choices=["EchoForge_v1_Base", "EchoForge_Base", "E2TTS_Base"]` (jitni bhi jagah, teen occurrences already confirmed: lines ~1522, ~1600, ~1832) — inhe `choices=["RaonOpenTTS_1B"]` karo, aur `value=` bhi `"RaonOpenTTS_1B"` karo jahan set hai
- Poore-training-flow ko verify karo — batch-size-calculation (`calculate_train()`, jo humne Phase-6 mein fix kiya tha), vocab-extension-logic, sab **model-agnostic-code hai already** (kisi specific-model-naam pe hardcoded-dependency nahi thi) — ye sab **as-is chalta rahega**, sirf model-selection-choices badal rahe hain

### File 3: `train/finetune_cli.py` — MODIFY
- Same pattern jaisa File 2: `choices=["EchoForge_v1_Base", "EchoForge_Base", "E2TTS_Base"]` (line ~30) ko `choices=["RaonOpenTTS_1B"]` karo
- Lines ~101-137: checkpoint-download conditional-branches, sirf `RaonOpenTTS_1B`-branch rakho

### File 4: `infer/infer_gradio.py` — MODIFY (already partially Raon-1B-ready hai)
- `load_echoforge()` function (jo `EchoForge_v1_Base`/330M load karta hai) — **poori tarah hatao**, uske saare call-sites bhi hatao
- `load_e2tts()` function — **poori tarah hatao**, uske saare call-sites bhi hatao
- `USING_SPACES`-related E2TTS-conditional-logic — hatao agar sirf E2TTS ke liye tha
- Isi file mein pehle se maujood `load_raon_1b()` — **iska naam consider karo change karne ka** (jaise `load_echoforge_model()` ya bas `load_model()` jaisa generic-naam), kyunki ab "Raon" specifically-naam-lena zaroori nahi hai jab wahi ab default/sirf-model hai. **Ye Replit Agent ka call hai** — agar rename karte hain to sab call-sites bhi consistently update karne honge
- UI-model-selector-dropdown (agar hai is file mein bhi, jo Gradio-tabs mein model choose karne deta ho) — same tarah sirf `RaonOpenTTS_1B` option rakho

### File 5: `infer/speech_edit.py` — MODIFY
- Grep se confirm hua isme bhi `SWivid`-reference hai — dhoondo aur `KRAFTON/Raon-OpenTTS-1B` se replace karo, jaisa baaki-files mein kiya

### File 6: `socket_server.py` — MODIFY
- Line ~227: `default=str(hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors"))` — isko `repo_id="KRAFTON/Raon-OpenTTS-1B", filename="model_520000.pt"` se replace karo

### File 7: Config files — DELETE
- `configs/EchoForge_Base.yaml` — **delete karo** (purana-330M-alternate-config)
- `configs/EchoForge_v1_Base.yaml` — **delete karo** (purana-330M-primary-config, jise humne rollback-ke-liye pehle rakha tha — ab confirm-decision ke baad hataana hai)
- `configs/E2TTS_Base.yaml` — **delete karo**
- `configs/E2TTS_Small.yaml` — **delete karo**
- `configs/EchoForge_Small.yaml` — **verify karo ye kis-architecture pe based hai** (agar F5-TTS-330M-family ka chhota-version hai, isse bhi delete karo; agar independently-useful hai future ke liye, Replit Agent decide kare aur user ko flag kare apne-decision ke saath)
- `configs/RaonOpenTTS_1B.yaml` — **yahi akela config bachega**, untouched rakho

### File 8: `model/backbones/unett.py` — DELETE
- Confirmed via grep: sirf `E2TTS_Base.yaml`/`E2TTS_Small.yaml` ke `backbone: UNetT` se reference hota hai, aur `model/__init__.py` + `infer/infer_gradio.py` mein import hota hai
- File delete karo, aur `model/__init__.py` + `infer/infer_gradio.py` se iska import-statement bhi hatao

### File 9: `model/backbones/mmdit.py` — DECISION NEEDED (Replit Agent flag kare, delete na kare bina-confirm)
- Grep se pata chala ye **kisi bhi config se directly reference nahi hota** (orphan-code hai already, koi active-model isko use nahi karta) — lekin `model/__init__.py`, `model/modules.py`, aur `scripts/count_params_gflops.py` mein import hota hai
- **Ye E2TTS ka sibling-architecture nahi hai** (MM-DiT, Stable-Diffusion-3-style, alag-research-experiment tha original-repo mein) — F5-TTS/E2TTS dono se independent hai
- **Replit Agent ise delete NA kare abhi** — sirf report kare ki "ye file kisi active-config se linked nahi hai, currently orphan/unused hai, delete karna safe lagta hai lekin scope se bahar hai kyunki na F5-TTS na E2TTS se directly related hai — user confirm kare"

### File 10: `model/backbones/dit.py` — MINOR TEXT-ONLY CHANGE
- Line ~194 ka comment `# "layernorm" (default, matches all existing EchoForge / E2TTS configs)` — sirf comment hai, koi functional-code nahi. Update karo taaki E2TTS ka mention na rahe (jaise `# "layernorm" (default)`), kyunki E2TTS-configs ab exist nahi karenge

---

## Verification Checklist (Replit Agent khud confirm kare)

1. Poore `src/echoforge_tts/` mein (triton_trtllm ke alawa) grep karo `SWivid`, `E2TTS`, `E2-TTS`, `EchoForge_v1_Base`, `EchoForge_Base` ke liye — koi occurrence nahi bachni chahiye (sirf `RaonOpenTTS_1B` aur naya-`KRAFTON/Raon-OpenTTS-1B`-reference bachein)
2. `configs/` folder mein sirf `RaonOpenTTS_1B.yaml` bache (plus `EchoForge_Small.yaml` ka decision jo upar flag kiya gaya)
3. `model/backbones/` mein `unett.py` delete ho, `dit.py` aur `mmdit.py` bachein (mmdit orphan-flag ke saath)
4. `api.py` ka `F5TTS`/`EchoForge` class import+construct karke verify karo — default-model-load bina kisi purane-330M-reference ke Raon-1B ki taraf jaaye
5. `finetune_gradio.py` Gradio-UI khud chalake dekho — model-dropdown mein sirf "RaonOpenTTS_1B" dikhe, koi purana-option na dikhe
6. Poore codebase mein ek final grep: `grep -rn "1250000\|F5TTS_v1_Base\|F5TTS_Base" src/echoforge_tts/` (triton ke alawa) — zero-results aane chahiye

---

## Explicitly Out of Scope
- `runtime/triton_trtllm/` — pehle jaisa hi, is poore scope se bahar (already multiple-baar confirm ho chuka hai, wahan purana-F5-TTS-reference rehna expected/accepted hai)
- Actual checkpoint ko naye-HuggingFace-repo pe re-upload karna — ye future-step hai, abhi sirf code mein `KRAFTON/Raon-OpenTTS-1B` hi hardcoded rahega (single-clean-location pe, taaki baad mein aasani se swap ho sake)
- Dataset/Phase-3 — is fix se koi lena-dena nahi
