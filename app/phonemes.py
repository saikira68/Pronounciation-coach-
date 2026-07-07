"""Grapheme-to-phoneme (reference) and phoneme-sequence alignment helpers.

The reference pronunciation for each recognised word is produced with espeak-ng
(via `phonemizer`), which emits IPA. The learner's *actual* phonemes come from a
wav2vec2 acoustic phoneme recogniser that is also trained on espeak IPA, so both
sides live in the same symbol inventory and can be compared directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

_backend: EspeakBackend | None = None


def _get_backend() -> EspeakBackend:
    global _backend
    if _backend is None:
        _backend = EspeakBackend(
            language="en-us",
            preserve_punctuation=False,
            with_stress=False,
        )
    return _backend


# Diacritics / stress / length marks we ignore when comparing phonemes so that
# minor realisation differences are not punished as "mistakes".
_STRIP = str.maketrans({c: None for c in "ˈˌːˑ̃ʰʷʲˠˤ̥̬͡ ‿"})


def normalize_phoneme_string(s: str) -> str:
    return s.translate(_STRIP).strip()


# espeak represents diphthongs inconsistently: joined mid-word ("day" -> "eɪ")
# but split word-initially ("I" -> "a ɪ"). The acoustic recogniser always emits
# them joined, so we merge adjacent vowel pairs into a single symbol on both sides.
_DIPHTHONGS = {
    "aɪ", "eɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "eə", "ʊə", "ɛə", "ɔə", "ʊ̯",
}


def merge_diphthongs(toks: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and (toks[i] + toks[i + 1]) in _DIPHTHONGS:
            out.append(toks[i] + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def tokenize_ipa(s: str) -> List[str]:
    """Split an IPA string into individual phoneme symbols.

    espeak emits phonemes separated by spaces when asked; if not, we fall back to
    a best-effort per-character split (combining diacritics stay with their base).
    """
    s = s.strip()
    if not s:
        return []
    if " " in s:
        toks = [normalize_phoneme_string(t) for t in s.split(" ")]
        return [t for t in toks if t]
    # best effort: treat each base char as a phoneme
    return [c for c in normalize_phoneme_string(s)]


@lru_cache(maxsize=4096)
def word_to_phonemes(word: str) -> Tuple[str, ...]:
    """Canonical IPA phonemes for a single English word."""
    clean = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not clean:
        return tuple()
    out = _get_backend().phonemize(
        [clean],
        separator=Separator(phone=" ", word="", syllable=""),
        strip=True,
    )
    # espeak splits some diphthongs word-initially ("I" -> "a ɪ"); merge them so
    # the symbol inventory matches the acoustic recogniser (which emits "aɪ").
    return tuple(merge_diphthongs(tokenize_ipa(out[0])))


def words_to_phonemes(words: List[str]) -> List[List[str]]:
    """Canonical phonemes for each word (see `word_to_phonemes`)."""
    return [list(word_to_phonemes(w)) for w in words]


# Phonemes that are near-identical acoustically or reflect natural, non-erroneous
# variation (vowel reduction, t-flapping, rhotic/tense-lax pairs). Treating these
# as equivalent avoids flagging natural connected speech as "mistakes".
_EQUIV_CLASSES = [
    {"ə", "ɐ", "ʌ", "ɜ", "ɚ", "ɝ"},  # schwa / reduced central vowels
    {"i", "ɪ"},
    {"u", "ʊ"},
    {"ɔ", "ɒ", "ɑ", "ɔː"},
    {"t", "ɾ", "d"},  # tap/flap allophones
    {"ɹ", "r", "ɻ"},
    {"e", "ɛ"},
    {"o", "oʊ", "əʊ"},
]
_EQUIV: dict = {}
for _cls in _EQUIV_CLASSES:
    for _p in _cls:
        _EQUIV.setdefault(_p, set()).update(_cls)


def phonemes_equivalent(a: str, b: str) -> bool:
    if a == b:
        return True
    return b in _EQUIV.get(a, ())


@dataclass
class IdxOp:
    op: str  # "match" | "sub" | "del" | "ins"
    ref_idx: int | None  # index into ref
    hyp_idx: int | None  # index into hyp


def align_indices(ref: List[str], hyp: List[str]) -> List[IdxOp]:
    """Same NW alignment as `align_phonemes`, but returns index references so the
    caller can attribute each operation back to a word / a recognised phoneme.
    Uses `phonemes_equivalent` so natural allophonic variation counts as a match."""
    n, m = len(ref), len(hyp)

    def sub_cost(a: str, b: str) -> int:
        return 0 if phonemes_equivalent(a, b) else 1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = sub_cost(ref[i - 1], hyp[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    ops: List[IdxOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub_cost(ref[i - 1], hyp[j - 1]):
            same = phonemes_equivalent(ref[i - 1], hyp[j - 1])
            ops.append(IdxOp("match" if same else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(IdxOp("del", i - 1, None))
            i -= 1
        else:
            ops.append(IdxOp("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops
