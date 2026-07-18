# EchoForge AI — Fix Naming-Collision + Add Missing DiT Architecture Support

## Goal — Do alag fixes, dono is prompt mein cover karne hain

**Fix 1:** `expand_model_embeddings` naam do jagah use ho raha hai (`train/finetune_gradio.py` aur `infer/utils_infer.py`), do alag kaam ke liye — ye rename karke resolve karna hai, koi functionality hataani nahi hai, sirf naam-clash khatam karna hai.

**Fix 2:** `RAON_1B_MODEL_CFG` (in `infer/infer_gradio.py`) mein `norm_type`, `logit_softcapping`, `post_norm` keys pass ho rahe hain DiT model ko, lekin `model/backbones/dit.py` ka `DiT.__init__()` inhe accept hi nahi karta (signature explicit hai, `**kwargs` nahi hai) — **ye abhi crash karega** jaise hi `load_raon_1b()` call hoga. Ye missing-feature hai — DiT backbone mein inka real support add karna hai.

---

## FIX 1: Naming-Collision Resolve Karo

### Confirmed current state (verified in codebase)
1. `train/finetune_gradio.py` line ~1026 — `def expand_model_embeddings(ckpt_path, new_ckpt_path, num_new_tokens=42):` — ye checkpoint-**file**-level function hai, ek `.pt`/`.safetensors` file leta hai, naya file banata hai. UI mein call hota hai line ~1125 pe.
2. `infer/utils_infer.py` line ~105 — `def expand_model_embeddings(model, new_vocab_size, init_std=0.02):` — ye in-memory **model-object**-level function hai, naya (humne Raon-1B integration ke liye banaya). `infer/infer_gradio.py` line ~194, 213 mein import+call hota hai.

### Kya karna hai
- `infer/utils_infer.py` wale (naya, model-object-level) ka **naam badlo** taaki collision khatam ho — suggest: `expand_text_embedding_layer()` (clearly batata hai ye in-memory embedding-layer resize karta hai, checkpoint-file nahi)
- Iska matching import+call-site `infer/infer_gradio.py` mein (line ~194, ~213) bhi naye naam se update karo
- `train/finetune_gradio.py` wala purana function **naam mat badlo** — wo already-working UI-feature hai (checkpoint-file preparation ke liye), usko chhedne ki zaroorat nahi, sirf doosre wale ka naam badalna kaafi hai collision resolve karne ke liye
- Dono function ke docstring ke top pe ek chhoti clarifying line add karo (jaise `"""NOTE: unrelated to train/finetune_gradio.py's checkpoint-file-level function of a similar earlier name — that one operates on saved .pt files; this one operates on an in-memory model object."""`) taaki future-developer ko turant samajh aaye ye do alag cheezein hain agar wo dono dhoondhte hue kabhi confuse ho

---

## FIX 2: DiT Backbone mein `norm_type` / `post_norm` / `logit_softcapping` ka Real Support Add Karo

### Confirmed problem (verified via code inspection)
`model/backbones/dit.py` ka `DiT.__init__()` signature explicitly ye parameters leta hai: `dim, depth, heads, dim_head, dropout, ff_mult, mel_dim, text_num_embeds, text_dim, text_mask_padding, text_embedding_average_upsampling, qk_norm, conv_layers, pe_attn_head, attn_backend, attn_mask_enabled, long_skip_connection, checkpoint_activations`. Isme `norm_type`, `logit_softcapping`, `post_norm` **hain hi nahi** — na parameter list mein, na `**kwargs` catch-all hai jo inhe silently absorb kar le.

`infer/infer_gradio.py` ka `RAON_1B_MODEL_CFG` in teeno keys ko pass karta hai model-build ke waqt — is se seedha `TypeError: unexpected keyword argument` crash hoga.

### Kya karna hai — architecture-level addition

**Ye ek naya feature add karna hai, "purana code hataana" nahi hai** — kyunki ye teeno cheezein DiT ke original F5-TTS design mein kabhi thi hi nahi. Raon-OpenTTS ne apne 1B-training ke liye khud DiT mein ye modifications kiye the — humein wahi replicate karna hai apne `dit.py` mein taaki checkpoint ke weights sahi tarike se, sahi behavior ke saath load/use hon (sirf load ho jaana kaafi nahi hai — agar normalization-type galat hai to output-quality silently kharab hogi, chahe koi crash na ho).

1. **`norm_type` parameter add karo `DiT.__init__()` mein** (default `"layernorm"` rakho taaki purane `EchoForge_v1_Base`/E2TTS configs bina change ke chalte rahein — sirf naya optional parameter hai). Value `"rmsnorm"` diye jaane par, jahan bhi `DiTBlock` ke andar normalization-layers hain (attention se pehle/baad, feedforward se pehle/baad), un layers ko `LayerNorm` ki jagah `RMSNorm` use karna chahiye. PyTorch mein `torch.nn.RMSNorm` already available hai (ya agar version-compatibility issue ho, ek chhota custom RMSNorm-class likh sakte ho — simple hai, sirf variance-normalization hai bina mean-centering ke).

2. **`post_norm` parameter add karo** (default `False`). Ye decide karta hai normalization **attention/feedforward-block se pehle** (`pre-norm`, jo F5-TTS ka current default-design hai) lagti hai ya **baad mein** (`post-norm`). `DiTBlock` ke andar forward-pass ki normalization-placement ko is flag ke hisaab se conditional banao.

3. **`logit_softcapping` parameter add karo** (default `None`). Agar value di jaaye (jaise koi float, though Raon-1B mein `null` hi hai to shayad activate nahi hoga unke case mein, lekin support hona chahiye future-use ke liye), to final output-logits par ek soft-capping function apply karo (typically `softcap * tanh(logits / softcap)` jaisa formula hota hai, jo bade transformer-models mein gradient-explosion rokne ke liye use hota hai — verify exact formula ki zaroorat pade to Gemma/Grok jaise papers ka reference le sakte ho jinhone ye technique popularize ki).

### Zaroori — backward-compatibility
Sabhi teen naye parameters ke defaults aise rakhne hain ki **purane `EchoForge_v1_Base.yaml` aur E2TTS configs bilkul waisa hi behave karein jaisa abhi karte hain** — koi regression nahi honi chahiye purane 330M path ke liye. Sirf jab explicitly `norm_type="rmsnorm"` ya `post_norm=True` ya `logit_softcapping=<value>` pass ho, tabhi naya-behavior activate ho.

---

## Verification Checklist (Replit Agent khud confirm kare)

1. Poore codebase mein `expand_model_embeddings` naam ka ab sirf ek hi occurrence bache (train/finetune_gradio.py mein) — doosri jagah naya naam ho, sab call-sites update hon
2. `load_raon_1b()` (infer_gradio.py) chalane ki simulation karo (ya kam-se-kam import+construct-call trace karo) — confirm karo ab `TypeError` nahi aata `norm_type`/`post_norm`/`logit_softcapping` ki wajah se
3. Purana `EchoForge_v1_Base` model (330M) load karke confirm karo uska output/behavior bilkul same hai jaisa in changes se pehle tha (regression-check — defaults ki wajah se ye pass hona chahiye, lekin explicitly verify karo)
4. `RMSNorm` ka implementation numerically sahi hai — agar naya custom-class likhi hai, ek chhota unit-test karo (random-tensor input do, verify output ka mean~0 nahi hoga jaisa LayerNorm mein hota hai, RMSNorm sirf variance-normalize karta hai)
5. `logit_softcapping=None` (default case) mein koi extra-computation na ho (performance-regression na aaye purane path ke liye)

---

## Explicitly Out of Scope
- Actual Hindi-finetuning — abhi bhi agla phase hai
- `runtime/triton_trtllm/` — pehle jaisa hi, scope se bahar
- `train/finetune_gradio.py` ka purana `expand_model_embeddings` (checkpoint-file-level) — iska naam ya logic nahi badalna, sirf docstring-clarification add karni hai
