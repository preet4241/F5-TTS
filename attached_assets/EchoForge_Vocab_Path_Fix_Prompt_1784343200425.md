# EchoForge AI — Fixed Vocab Path + Auto-Create Checkpoint Logic

## Goal
Vocab.txt file ke liye ek **single, fixed, hardcoded path** set karna hai jo har jagah (data preparation, training, inference/use) se same rahe — kahin bhi manually path pass karne ki zaroorat na ho. Agar vocab.txt us fixed path pe exist nahi karti, to code khud usko sahi jagah create kar de (bina crash kiye), aur agar purani/galat vocab.txt kahin aur mile to usko flag karke bataye.

---

## Current State (already verified in codebase — is se match karna hai)

**Existing vocab.txt files jo abhi project mein hain (dono purane Chinese-pinyin era ke hain, Hindi phonemizer se ban hi nahi ye):**
1. `data/Emilia_ZH_EN_pinyin/vocab.txt` — 2545 lines, purana pinyin-based vocab, `prepare_csv_wavs.py` mein `PRETRAINED_VOCAB_PATH` ke naam se hardcoded reference hai
2. `src/echoforge_tts/infer/examples/vocab.txt` — 2545 lines (same purani content), `infer/utils_infer.py` line ~249 mein hardcoded reference hai as fallback vocab for inference

**Do jagah confirmed bugs jo is fix se solve honge:**
- `prepare_csv_wavs.py` mein `is_finetune=True` (jo default hai jab tak `--pretrain` flag na diya jaaye) hone par naya vocab banata hi nahi — seedha purani pinyin `vocab.txt` copy kar deta hai. Isse Hindi phonemes kabhi vocab mein aate hi nahi.
- `save_prepped_dataset()` output folder naam seedha user-diye `out_dir` se banta hai — koi automatic `_phonemizer` suffix nahi lagta. Lekin `get_tokenizer()` (in `model/utils.py`) path expect karta hai strictly `{dataset_name}_phonemizer/vocab.txt` format mein. Ye mismatch hai — agar user ne khud sahi naam se folder nahi banaya to training-time crash hoga file-not-found ke sath.

---

## Required Change: Ek Single Fixed Path

Ek constant define karo (jaise `model/utils.py` mein, ya ek naya chhota `config/paths.py` file mein — jo bhi cleaner lage) jo poore project mein Hindi+English phonemizer vocab.txt ke liye **ek hi source-of-truth path** ho. Jaise:

```
data/echoforge_hindi_en_phonemizer/vocab.txt
```

(Exact naam Replit Agent decide kar sakta hai jo project convention se best match kare, lekin important ye hai ki **ek hi jagah define ho aur har file usi constant ko import kare** — path kahin bhi dobara hardcode na ho.)

Is constant ko in teeno jagah use karna hai:

### File 1: `model/utils.py` — `get_tokenizer()`
Abhi ye function `{dataset_name}_{tokenizer}` pattern se dynamically path banata hai. Isko naye fixed-path constant ki taraf point karo — `dataset_name` parameter ab bhi rahega (backward compatibility ke liye, ya documentation/logging ke liye), lekin actual file-lookup fixed constant se ho.

### File 2: `train/datasets/prepare_csv_wavs.py`
- `PRETRAINED_VOCAB_PATH` variable ko hata kar naye fixed-path constant se replace karo
- **`is_finetune=True` wala fix bhi yahin karna hai:** ab chahe finetune ho ya naya pretrain, dono cases mein naye Hindi+English text se hi vocab banni chahiye — purani pinyin vocab.txt ko blindly copy karne wala `shutil.copy2()` wala path poori tarah hata do. Naya phonemizer-based vocab hi hamesha generate hona chahiye (jaisa already `else` branch mein logic hai — bas usko default/only path bana do)
- Output vocab hamesha fixed constant path pe hi likhi jaaye — `out_dir` user jo bhi de, vocab.txt specifically fixed path pe save ho (raw.arrow aur duration.json bhale hi user ke diye out_dir mein rahen, lekin vocab.txt sirf fixed central location pe)

### File 3: `infer/utils_infer.py`
Line ~249 ka hardcoded `infer/examples/vocab.txt` fallback bhi naye fixed-path constant se replace karo, taaki inference bhi wahi single vocab file use kare jo training ne banayi thi — do alag vocab files (training ki aur inference ki) kabhi mismatch na ho sakein.

---

## New Logic Required: Auto-Create Checkpoint

Data-preparation step shuru hone se pehle (ya `get_tokenizer()` call hone se pehle, jahan bhi sabse pehle vocab chahiye hoti hai), ek checkpoint-check function add karo jo:

1. **Check kare** ki fixed path pe vocab.txt already exist karti hai ya nahi
2. **Agar exist karti hai** → kuch mat karo, aage badh jao (use as-is)
3. **Agar exist nahi karti** → us fixed path ka poora folder-structure khud create karo (parent directories bana ke), phir naye text-sample se phonemizer-based vocab generate karke wahi save karo, phir aage badho
4. Ye check idempotent hona chahiye — baar-baar run karne pe dobara-dobara vocab na bane agar already sahi jagah maujood hai

## New Logic Required: Old/Misplaced Vocab Detection

Ek chhota one-time scan/utility bhi add karo (script ya function, jo bhi natural jagah lage — jaise ek `scripts/find_stale_vocab.py` ya `prepare_csv_wavs.py` ke start mein hi ek check) jo:

1. Poore project mein kahin bhi `vocab.txt` naam ki files dhoonde (fixed path ke alawa)
2. Jo bhi mile, unko **flag/print** kare — file ka exact path bataye console pe (jaise `print(f"⚠️ Old/misplaced vocab.txt found at: {path}")`)
3. Un files ko **delete** kare (fixed-path wali ke alawa sab)
4. Ye scan pehli baar data-prep chalane pe automatically ho jaaye, taaki purani `data/Emilia_ZH_EN_pinyin/vocab.txt` aur `infer/examples/vocab.txt` (jo already identify ho chuki hain is codebase mein) khud detect ho kar clean ho jaayein aur console pe report bhi ho jaayein

---

## Verification Checklist (Replit Agent khud confirm kare)

1. Poore codebase mein grep karke confirm karo ki `vocab.txt` ka koi bhi hardcoded path (`PRETRAINED_VOCAB_PATH`, `infer/examples/vocab.txt` reference, ya `get_tokenizer()` ka dynamic pattern) sirf naye central constant ko refer kar raha ho — koi doosri jagah path dobara na likha ho
2. `is_finetune=True` case mein bhi ab naya phonemizer-vocab hi banta hai, purani pinyin file copy nahi hoti — is logic ko explicitly verify karo
3. Fresh run (jahan fixed-path pe koi vocab.txt nahi hai) pe auto-create checkpoint sahi se folder bana kar vocab generate kare, crash na ho
4. Dobara run (jahan fixed-path pe vocab.txt already hai) pe dobara vocab regenerate na ho, bas existing file use ho
5. Stale-vocab scan chalane pe `data/Emilia_ZH_EN_pinyin/vocab.txt` aur `src/echoforge_tts/infer/examples/vocab.txt` dono detect, report, aur delete hon (in dono ka exact path humne already is codebase mein confirm kiya hai)
6. `runtime/triton_trtllm/` folder ko is change se touch mat karna — wo already scope se bahar hai (separate reason se, pehle explain ho chuka)
