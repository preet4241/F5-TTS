# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "phonemizer"):
    """
    tokenizer   - "phonemizer" use espeak-ng G2P for Hindi (Devanagari) and English text, need .txt vocab_file
                  vocab folder naming convention: {dataset_name}_phonemizer/vocab.txt
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "phonemizer", derived from phoneme symbols produced by espeak-ng for the dataset
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["phonemizer", "char"]:
        tokenizer_path = os.path.join(files("echoforge_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# Language detection helpers


def is_devanagari(c):
    """Returns True if the character is in the Devanagari Unicode block (Hindi/Sanskrit)."""
    return "\u0900" <= c <= "\u097F"


# Module-level phonemizer backends — lazily initialized and reused across calls
# to avoid spawning a new espeak-ng subprocess on every inference request.
_phonemizer_en = None
_phonemizer_hi = None


def _get_phonemizer_backend(lang: str):
    """Return a cached EspeakBackend for the requested language ('en' or 'hi')."""
    global _phonemizer_en, _phonemizer_hi
    from phonemizer.backend import EspeakBackend  # imported lazily so module loads without espeak-ng

    if lang == "en" and _phonemizer_en is None:
        _phonemizer_en = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
    if lang == "hi" and _phonemizer_hi is None:
        _phonemizer_hi = EspeakBackend("hi", preserve_punctuation=True, with_stress=False)
    return _phonemizer_en if lang == "en" else _phonemizer_hi


def _segment_by_script(text: str):
    """Split a string into (segment, lang) pairs based on character script.

    Devanagari characters map to 'hi', everything else maps to 'en'.
    Consecutive characters of the same script are grouped together.
    """
    segments = []
    current_chars: list[str] = []
    current_lang = None

    for c in text:
        lang = "hi" if is_devanagari(c) else "en"
        if lang != current_lang:
            if current_chars:
                segments.append(("".join(current_chars), current_lang))
            current_chars = [c]
            current_lang = lang
        else:
            current_chars.append(c)

    if current_chars:
        segments.append(("".join(current_chars), current_lang))

    return segments


def convert_text_to_phonemes(text_list):
    """Convert a list of texts to phoneme/character sequences.

    Handles English, Hindi (Devanagari), and mixed Hinglish text.
    Uses espeak-ng via the ``phonemizer`` library for both languages.
    Backends are initialized once at module level and reused across calls.

    Returns a list of character lists — same format as the former
    convert_char_to_pinyin() so all downstream tokenizer code is unchanged.
    Falls back to raw characters gracefully if espeak-ng is unavailable.
    """
    custom_trans = str.maketrans(
        {";": ",", "\u201c": '"', "\u201d": '"', "\u2018": "\'", "\u2019": "\'"}
    )

    final_text_list = []
    for text in text_list:
        text = text.translate(custom_trans)
        segments = _segment_by_script(text)
        char_list: list[str] = []

        for seg_text, lang in segments:
            if not seg_text:
                continue
            try:
                backend = _get_phonemizer_backend(lang)
                phonemized = backend.phonemize([seg_text], njobs=1)[0]
                # Add a space boundary between adjacent segments when needed
                if char_list and char_list[-1] != " " and phonemized and phonemized[0] != " ":
                    char_list.append(" ")
                char_list.extend(list(phonemized))
            except Exception:
                # Graceful fallback: emit raw characters without crashing
                char_list.extend(list(seg_text))

        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
