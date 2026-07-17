# EchoForge AI — Tokenizer System Overhaul (Chinese Removal + Hindi/English Phonemizer)

## Goal
Poore tokenizer system se Chinese-specific logic (`rjieba`, `pypinyin`) ko completely hata kar, uski jagah ek Hindi + English phonemizer-based system laana hai. Ye change poore project mein jahan-jahan tokenizer touch hota hai wahan consistently reflect hona chahiye — training, inference, eval, aur Gradio UI sab jagah.

---

## File 1: `model/utils.py`

**Problem:** Ye file ka `convert_char_to_pinyin()` function poori tarah Chinese-centric hai — `rjieba` se text segment karta hai aur `pypinyin` se Chinese characters ko tone-marked pinyin mein convert karta hai. Ek `is_chinese()` helper bhi hai jo sirf CJK Unicode range check karta hai. Ye poore system ka entry point hai jahan raw text, model ke training/inference se pehle, phonetic/character units mein convert hota hai. Hindi (Devanagari) ke liye is function mein koi handling nahi hai — Hindi text bina kisi normalization ke raw characters ki tarah nikal jaata hai.

**Kya karna hai:**
- `rjieba` aur `pypinyin` imports poori tarah hata do
- `is_chinese()` helper hata do (ya replace kar do naye language-detection helpers se)
- `convert_char_to_pinyin()` function ko rename aur rebuild karo ek naye function mein jo:
  - English text ke liye ek English G2P/phonemizer use kare (jaise `phonemizer` library backend `espeak-ng` ke sath, ya `g2p_en`)
  - Hindi (Devanagari) text ke liye ek Hindi phonemizer use kare (jaise `indic_nlp_library`, ya `phonemizer` ka `espeak-ng` Hindi voice — espeak-ng dono languages support karta hai ek hi library se, is se dependency simplify ho sakti hai)
  - Mixed Hindi-English (Hinglish) text ko handle kare — har word/segment ko uski script ke hisaab se sahi phonemizer mein route kare
  - Ek naya `is_devanagari(c)` helper add karo jo Unicode range `\u0900`–`\u097F` check kare, `is_chinese()` ke replacement ke taur pe
- Function ka return format same rakho jitna ho sake (list of char/phoneme lists) taaki downstream code (backbone models) mein breaking changes minimum hon
- `get_tokenizer()` function ke docstring/comments update karo jahan "pinyin" ka reference hai, taaki naya tokenizer mode (jaise `"phonemizer"` ya jo bhi naam do) explain ho

**Dependency changes:** `requirements.txt` / `pyproject.toml` mein `rjieba` aur `pypinyin` hata kar `phonemizer` (+ system-level `espeak-ng` dependency note) ya jo bhi library choose karo add karo.

---

## File 2: `train/datasets/prepare_csv_wavs.py`

**Problem:** Ye file (jo custom Hindi dataset prep ke liye entry point hai) `convert_char_to_pinyin` import karke apne text-processing batch pe call karta hai.

**Kya karna hai:** Naye renamed function ko import/call karo. Function signature same rahega to sirf import statement aur call-site update honge.

---

## File 3: `train/datasets/prepare_emilia.py` aur `prepare_wenetspeech4tts.py`

**Problem:** Dono files `convert_char_to_pinyin` use karte hain apne respective dataset-prep pipelines mein. `prepare_wenetspeech4tts.py` khud hi Mandarin-specific dataset (WenetSpeech) ke liye hai.

**Kya karna hai:**
- `prepare_csv_wavs.py` ko naye function ke sath update karo (jaisa File 2 mein)
- `prepare_wenetspeech4tts.py` aur `prepare_emilia.py` (agar Emilia ka ZH split use nahi karna) — decide karo ki inhe delete karna hai ya as-is chhod dena hai unused state mein. Agar Chinese dataset support poori tarah hata rahe ho, ye do files ab kaam ke nahi rahenge — inhe project se remove karna cleaner hoga taaki purana dead code na rahe.

---

## File 4: `train/finetune_gradio.py`

**Problem:** Do jagah `convert_char_to_pinyin([text], polyphone=True)` call hota hai — dataset-sample preview aur training-data prep ke dauraan.

**Kya karna hai:** Dono call-sites ko naye function ke sath update karo. Agar naye function ka signature `polyphone` parameter nahi rakhta (kyunki polyphone Chinese-specific concept hai — ek character ke multiple pronunciations), to ye parameter bhi is file se hata do.

Isi file mein `transcribe(file_segment, language)` bhi call hota hai audio-slicing ke dauraan — Whisper transcription flow hai, ye already theek hai jaisa humne confirm kiya, isko touch mat karo. Sirf sanity-check karo ki `language` variable jo yahan pass ho raha hai UI se sahi tarike se "hi" ya "en" values leke aa raha hai (dropdown/selector se), koi hardcoded ya broken value nahi.

---

## File 5: `infer/utils_infer.py`

**Problem:** `convert_char_to_pinyin(text_list)` call hota hai inference text preprocessing mein (`infer_batch_process()` ke andar) — ye woh jagah hai jahan har generation request ka text tokenize hota hai final audio banane se pehle.

**Kya karna hai:** Call-site ko naye function ke naam/signature ke sath update karo. Ye sabse critical call-site hai kyunki ye har single inference request pe chalta hai — is jagah particular dhyan do ki naya phonemizer function fast ho (agar `espeak-ng` subprocess-based hai to har call pe naya process spawn na ho, agar possible ho to phonemizer object ko ek baar initialize karke reuse karo, jaise `asr_pipe` global pattern already is file mein use ho raha hai).

Isi file mein Whisper `transcribe()` aur `initialize_asr_pipeline()` bhi hain — humne confirm kiya ye already Hindi+English dono support karte hain (`whisper-large-v3-turbo` multilingual model hai), inhe replace mat karo. Bas verify karo ki `transcribe(ref_audio, language)` jahan-jahan call hota hai, wahan `language` explicitly "hi" ya "en" pass ho raha hai jab user ne wo select kiya ho — na ki hamesha `None` (auto-detect) pe chhoda ja raha ho, jo mixed Hinglish reference audio pe galat language detect kar sakta hai.

---

## File 6: `infer/speech_edit.py`

**Problem:** `convert_char_to_pinyin(text_list)` call hota hai speech-editing ke text preprocessing mein — same pattern jaisa `utils_infer.py` mein hai.

**Kya karna hai:** Call-site update karo naye function ke sath, same tarike se jaisa File 5 mein.

---

## File 7: `eval/utils_eval.py`

**Problem A (tokenizer):** `convert_char_to_pinyin(text, polyphone=polyphone)` call hota hai evaluation text-prep mein.

**Problem B (WER language support):** `load_asr_model(lang)` aur `run_asr_wer()` functions mein sirf do languages hardcoded hain — `"zh"` (FunASR Paraformer) aur `"en"` (faster-whisper). Koi bhi aur language pass karne pe `NotImplementedError` raise hota hai. Hindi ("hi") ke liye koi branch nahi hai. `punctuation_all` bhi Chinese punctuation (`zhon.hanzi`) + English punctuation combine karta hai — Devanagari punctuation/normalization ka koi handling nahi.

**Kya karna hai:**
- Tokenizer call-site (Problem A) ko naye function ke sath update karo
- `load_asr_model()` mein ek naya `elif lang == "hi":` branch add karo jo `faster-whisper` ya `openai/whisper-large-v3-turbo` ko Hindi mode mein load kare (English wale `elif` branch jaisa hi pattern, bas `language="hi"` pass karke)
- `run_asr_wer()` mein corresponding `elif lang == "hi":` branch add karo transcription call ke liye
- **Decide karo:** `"zh"` branch (FunASR/Paraformer) ko poori tarah hatana hai ya rehne dena hai as an unused/legacy option. Agar Chinese support completely remove kar rahe ho project se, is branch ko bhi hata do aur `zhconv`, `funasr` jaisi Chinese-specific eval dependencies bhi requirements se hata do.
- Text-normalization step (jahan `truth`/`hypo` compare hone se pehle process hote hain) mein ek Hindi-specific branch add karo — Chinese ke liye jo character-by-character splitting hoti hai (`" ".join([x for x in truth])`), Hindi ke liye ye zaroori nahi (Hindi already space-separated words hoti hai jaisa English), isliye Hindi ko English wale normalization path (lowercase — Devanagari mein case nahi hota to ye skip ho sakta hai, bas punctuation-strip aur whitespace-normalize) ke jaisa treat karo, naya branch add karke.

---

## File 8: Config files — `configs/*.yaml` (EchoForge_Base.yaml, EchoForge_Small.yaml, EchoForge_v1_Base.yaml, EchoForge_v1_Small.yaml, E2TTS_Base.yaml, E2TTS_Small.yaml)

**Problem:** Har config file mein `tokenizer: pinyin` hardcoded default hai.

**Kya karna hai:** Sabhi config files mein `tokenizer: pinyin` ko naye tokenizer mode ke naam se replace karo (jo naam File 1 mein `get_tokenizer()` ke andar naye mode ke liye choose karo, jaise `tokenizer: phonemizer`). Comment bhi update karo jahan "pinyin" ka mention hai.

---

## File 9: Vocab files — `data/` directory (jahan bhi `_pinyin/vocab.txt` naming pattern hai)

**Problem:** Dataset-specific vocab folders ka naming pattern `{dataset_name}_pinyin/` hai (jaise `get_tokenizer()` mein path-construction se pata chalta hai). Ye purane Chinese pinyin-tokenizer convention se aaya naming hai.

**Kya karna hai:** Ye code-change nahi hai, lekin naya vocab jab Hindi dataset ke liye banega (Phase 2 — vocab extension step, jo hum baad mein karenge), uska folder naming convention nayi tokenizer scheme ke hisaab se hona chahiye (jaise `{dataset_name}_phonemizer/`). Abhi is file-map mein sirf note kar rahe hain — actual vocab generation alag step hai jo tokenizer code update hone ke baad hoga.

---

## Cross-cutting Verification Checklist (sab files update hone ke baad)

1. Poore `src/echoforge_tts/` mein dobara grep karo `rjieba`, `pypinyin`, `is_chinese` ke liye — kahin bhi reference nahi bachna chahiye except `runtime/triton_trtllm/` folder (wo explicitly untouched/justified hai, usse touch mat karna)
2. `convert_char_to_pinyin` naam poore codebase mein kahin bhi reh na jaaye — naye function-naam se consistently replace hona chahiye
3. Ek chhota manual test: same function ko English text, Hindi (Devanagari) text, aur mixed Hinglish text (ek hi sentence mein dono) pe chalao — teeno cases mein crash nahi hona chahiye aur output reasonable phoneme/char list dena chahiye
4. `requirements.txt`/`pyproject.toml` mein purani Chinese-specific libraries (`rjieba`, `pypinyin`, aur agar eval se `zhon`/`funasr`/`zhconv` bhi hata rahe ho to wo bhi) clean ho
5. `runtime/triton_trtllm/` folder ko is change se bilkul touch mat karna — wo pehle se justified reason se untouched rakha gaya hai, uska apna separate `convert_char_to_pinyin` copy hai jo is scope se bahar hai

---

## Explicitly Out of Scope (is update mein mat chhedo)

- Whisper ASR setup (`initialize_asr_pipeline`, `transcribe()` in `infer/utils_infer.py`) — already Hindi+English dono support karta hai (`whisper-large-v3-turbo` multilingual hai), replace nahi karna, sirf verify karna ki language param sahi pass ho raha hai
- `runtime/triton_trtllm/` — separate deployment path, untouched rakhna hai
- Vocab.txt actual generation/extension — ye tokenizer code update hone ke baad ka alag step hai
