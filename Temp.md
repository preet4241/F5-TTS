## EchoForge Internal Architecture Update Plan — Best + Extra Options

### **Phase 0 — Tokenizer strategy decide karna**
**✅ Best:** `char` tokenizer — already exists, zero naya code, config switch se ho jaata hai. File khud recommend karti hai (Section 7, point 2) chhote dataset pe pehle isse try karne ko.
**➕ Extra options:**
- Proper G2P library (`indic_nlp_library` / `epitran`) — better prosody/matra handling, par zyada dev time
- `espeak-ng` Hindi voice ka phoneme output use karna — G2P se bhi zyada precise, lekin external dependency add hoti hai
- Hybrid: char tokenizer se start, baad mein G2P layer upar se add karna bina backbone dobara touch kiye

---

### **Phase 1 — Tokenizer/G2P Layer (`model/utils.py`)**
**✅ Best:** `is_chinese()` jaisa ek `is_devanagari()` check add karo (Unicode range `\u0900–\u097F`), taaki Hindi text `convert_char_to_pinyin()` mein galti se skip/corrupt na ho.
**➕ Extra options:**
- Sirf detection nahi, ek **language-tag routing system** bana sakte ho — text automatically detect kare ki Hindi hai ya English hai ya mixed (Hinglish), aur har part ko sahi tokenizer path mein bheje (ye Section 6.1 ka "code-switch detection" idea hai)
- Multi-script combined vocab (Devanagari + Latin ek hi `vocab.txt` mein) — taaki ek hi model Hindi + English + Hinglish teeno handle kare bina alag model banaye

---

### **Phase 2 — Vocab Extension (`expand_model_embeddings()`)**
**✅ Best:** Existing function use karo as-is — apne Hindi dataset ke text pe chalao, purane tokens preserve honge, naye Devanagari tokens random-init honge. Ye kaam karta hai, bas optimal nahi.
**➕ Extra options:**
- Random init ki jagah **nearest-neighbor init** (Section 4.19 suggestion) — phonetically/visually similar existing token ka embedding copy karke naya token start karo, convergence tez hoga
- Agar future mein multiple Indian languages plan hai (Hindi + Marathi + etc.), ek **shared Devanagari-family vocab** ek hi baar design karo abhi taaki baar-baar vocab expand na karna pade

---

### **Phase 3 — Dataset Prep**
**✅ Best:** `prepare_csv_wavs.py` — already tera custom-CSV format ke liye bana hai, IIT Madras data seedha isme fit hoga.
**➕ Extra options:**
- Data-quality filtering thoda strengthen karo Hindi ke liye specifically — jaise repetition-detection (jo abhi Chinese "aaaa..." pattern ke liye tuned hai) ko Devanagari repetition patterns ke liye bhi test karo
- Agar synthetic data bhi mix karna hai (jo humne pehle discuss kiya tha — supplement only), ek separate `prepare_synthetic.py` bana sakte ho jo real data se clearly tagged/separated rahe (taaki baad mein quality-debug easy ho)

---

### **Phase 4 — EPSS Timesteps Recompute**
**✅ Best:** File jo bola — apne Hindi validation set pe ek-baar grid search karke naye optimal timesteps nikaalo (Section 4.4).
**➕ Extra options:**
- Isi step pe agar time hai to **higher-order ODE solver** (RK4) try kar sakte ho Euler ki jagah (Section 4.3 suggestion) — kam steps mein better accuracy
- Long-term: step-distillation research (Section 6.2) — 1-4 step inference tak jaana, lekin ye separate bada project hai, abhi ki priority nahi

---

### **Phase 5 — Fine-tuning Run**
**✅ Best:** `finetune_cli.py` se direct CLI launch (Gradio UI ke bajaye) — kam moving parts, `shell=True` subprocess risk bhi bypass ho jaata hai kyuki tu khud CLI se control karega.
**➕ Extra options:**
- LoRA/PEFT-based fine-tuning (Section 6.3) — poora model touch karne ke bajaye chhota adapter train karo. Colab free-tier ke liye **zyada practical** hai VRAM ke hisaab se, aur future mein multiple voices ko alag chhoti LoRA files mein rakh sakte ho (ek hi base model shared)
- Full fine-tune agar VRAM allow kare — thoda behtar quality de sakta hai LoRA se, par risk zyada (OOM)

---

### **Phase 6 — Batch-size Sanity**
**✅ Best:** `calculate_train()` ka suggestion sirf starting point maano, khud manually monitor karke adjust karo (jaisa file khud warn karti hai).
**➕ Extra options:**
- GPU-specific override table bana sakte ho (T4 vs A100 alag hardcoded values) taaki har baar manually na sochna pade
- `bitsandbytes` 8-bit AdamW use karo (already trainer.py mein option hai) — VRAM aur bachega Colab pe

---

### **Phase 7 — Evaluation**
**✅ Best:** `run_asr_wer()` mein `elif lang == "hi"` branch add karo (Whisper multilingual already Hindi support karta hai) — chhota, low-effort fix.
**➕ Extra options:**
- A/B listening harness (Section 6.5) — base F5-TTS vs tera fine-tuned Hindi model side-by-side compare karne ka simple Gradio tool
- UTMOS score ko sirf relative trend (before/after epochs) ke liye graph karo, absolute number pe bharosa mat karo

---

### **Phase 8 — Concurrency/Production Safety**
**✅ Best:** Ye phase tab karo jab bot live multi-user serve karega, abhi Colab-solo stage mein zaroori nahi.
**➕ Extra options:**
- `lru_cache` cache-invalidation add karna model hot-reload ke case ke liye
- Global thread-unsafe caches (`asr_pipe`, `_ref_audio_cache`) ko per-request scope mein move karna
- Audio watermarking (Section 6.6) agar product commercial ban raha hai — responsible-AI disclosure ke liye
