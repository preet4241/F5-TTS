# EchoForge AI — Batch-Size Heuristic Safety Fix

## Goal
`train/finetune_gradio.py` ke `calculate_train()` function mein jo batch-size auto-calculation formula hai, usme ek confirmed bug hai jo **negative ya zero batch size** produce kar sakta hai chhote/kam-memory GPUs (jaise Colab T4) par. Isko safe, predictable, aur GPU-aware banana hai — bina kisi silent/accidental crash ke.

---

## Confirmed Problem (verified via simulation)

File: `train/finetune_gradio.py`, function `calculate_train()`

Current formula (jab `batch_size_type == "frame"`):
```
batch_size_per_gpu = max(int(38400 * (avg_gpu_memory - 5) / 75), int(max_sample_length))
```

**Ye formula kisi specific GPU (probably A100-class, ~40-80GB) ke liye tune kiya gaya tha** — koi comment/documentation nahi batata exact origin, magic numbers (`38400`, `5`, `75`) hardcoded hain.

**Simulation se confirm hua** (real numbers, guess nahi) — ye hypothetical edge-cases hain jo dikhate hain formula ka math kahan toot sakta hai, single-session normal-use mein zyada probability nahi hai inki, lekin protection cheap hai:
| Scenario | avg_gpu_memory | Raw batch_size_per_gpu |
|---|---|---|
| Edge-case: unexpectedly low reported memory | 4 GB | **-512 (negative!)** |
| Colab T4, typical free memory | 10 GB | 2560 |
| Colab T4, full free memory | 15 GB | 5120 |
| A100 40GB | 40 GB | 17920 |
| A100 80GB | 80 GB | 38400 |

**Context:** Ye ek single, dedicated Colab session pe chalega — koi shared/multi-notebook memory-pressure scenario nahi hai. Isliye is fix ka goal **aggressive multi-tenant fallback design karna nahi hai**, sirf ek **cheap insurance/guard-rail** add karna hai taaki agar kabhi `avg_gpu_memory` kam report ho (jaise CUDA context startup overhead ki wajah se, jo T4 pe khud hi 1-2GB tak le sakta hai), formula kabhi negative/zero batch size na de. Abhi `max(negative, max_sample_length)` ki wajah se ye kabhi-kabhi accidentally bach jaata hai, **lekin ye guaranteed protection nahi hai — sirf coincidence hai, design-level safety nahi.**

---

## Required Fix — Safety-First Approach

### 1. Hard floor/clamp on GPU memory calculation
`avg_gpu_memory - 5` wala subtraction kabhi bhi ek safe minimum se neeche nahi jaana chahiye. Ek explicit floor add karo taaki ye kabhi negative ya zero ke close na jaaye — jaise minimum effective memory value clamp karo (exact number Replit Agent decide kare based on existing formula's intent, lekin **kabhi negative allowed na ho**).

### 2. Light guard-rail for small GPU case (light-weight, over-engineer mat karo)
Single-Colab-session context hai, isliye poora multi-tier fallback-system design karne ki zaroorat nahi. Bas ek simple threshold-check kaafi hai: agar `avg_gpu_memory` kisi bahut kam value (jaise ~6GB se neeche — jo genuinely broken/unexpected state hoga T4 jaisa GPU ke liye) pe aaye, to formula pe blindly trust karne ke bajaye ek **chhota safe fixed number** use karo us edge-case ke liye. Ye sirf ek safety-net hai, koi elaborate GPU-tier classification system nahi banani.

### 3. Never return zero or negative — explicit final safety check
Formula chalane ke baad, return karne se **pehle** ek final explicit check add karo: agar calculated `batch_size_per_gpu` kisi bhi tarike se `1` se kam nikle, to usko ek **safe minimum value** (jaise `1` ya chhota fixed default) pe force karo. Ye ek last-line-of-defense hai — chahe upar ka formula/floor kisi wajah se fail ho jaaye, ye check final guarantee deta hai ki kabhi bhi invalid batch size training ko na jaaye.

### 4. User-facing warning (silent mat rehne do)
Agar calculated batch size formula se bahut chhota nikalta hai (jo indicate karta hai GPU memory tight hai), ek **clear warning message** return/display karo Gradio UI mein — user ko batao "GPU memory kam detect hui, conservative batch size use ho raha hai — agar training slow lage ya OOM aaye to manually adjust karo." Silently kisi bhi number pe chala dena galat hai; user ko pata hona chahiye jab system ek edge-case/fallback path le raha ho.

### 5. Same fix `batch_size_type == "sample"` branch mein bhi
Formula sirf `"frame"` type ke liye check hui hai (jahan bug confirm hua), lekin `"sample"` type wala branch (`batch_size_per_gpu = int(200 / (total_duration / total_samples))`) bhi similarly verify karo — agar `total_duration/total_samples` bahut zyada ho (lambi average audio clips), ye bhi chhota/zero batch size de sakta hai. Same safety-floor logic wahan bhi apply karo.

---

## Explicitly NOT in scope for this fix
- Poora `calculate_train()` function redesign karna nahi hai — sirf safety-clamping add karni hai, existing formula ka intent/behavior bade GPUs ke liye same rahe
- Actual optimal batch-size discovery (real training run ke through) — ye alag step hai, tab hoga jab actual dataset training ho rahi hogi
- `duration.json`-dependent logic (jo dataset se aata hai) — is fix ka scope sirf GPU-memory-based calculation tak hai

---

## Verification Checklist (Replit Agent khud confirm kare)

1. Simulate karo same scenarios (4GB, 10GB, 15GB, 40GB, 80GB avg_gpu_memory) jaise humne test kiya — confirm karo ab koi bhi scenario negative ya zero batch size nahi deta
2. `"sample"` batch_size_type branch bhi similarly test karo edge-case values ke sath (bahut lambi average duration)
3. Warning message sahi se trigger ho jab fallback/conservative path liya jaaye
4. Bade GPU (jaise 40GB+) ka behavior existing formula jaisa hi rahe (regression na ho) — sirf chhote GPU ka case naya safe path le
