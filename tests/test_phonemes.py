"""Unit tests for the phoneme/alignment layer (no heavy models required)."""

from app.phonemes import (
    align_indices,
    merge_diphthongs,
    phonemes_equivalent,
    word_to_phonemes,
)


def test_diphthongs_merge_word_initial():
    # espeak splits "I" into "a ɪ"; it must be merged to one symbol.
    assert word_to_phonemes("I") == ("aɪ",)
    assert merge_diphthongs(["a", "ɪ"]) == ["aɪ"]
    assert merge_diphthongs(["d", "e", "ɪ"]) == ["d", "eɪ"]


def test_g2p_basic_words():
    assert word_to_phonemes("through") == ("θ", "ɹ", "u")
    assert word_to_phonemes("") == tuple()


def test_equivalence_classes():
    assert phonemes_equivalent("ə", "ʌ")  # schwa family
    assert phonemes_equivalent("i", "ɪ")
    assert phonemes_equivalent("t", "ɾ")  # flap
    assert not phonemes_equivalent("p", "b")


def test_alignment_counts_matches_and_subs():
    ref = ["k", "æ", "t"]
    hyp = ["k", "æ", "t"]
    ops = align_indices(ref, hyp)
    assert all(o.op == "match" for o in ops)

    ops = align_indices(["k", "æ", "t"], ["k", "æ", "d"])  # t/d are equivalent
    assert all(o.op == "match" for o in ops)

    ops = align_indices(["k", "æ", "t"], ["k", "ɪ", "t"])  # æ vs ɪ is a real sub
    assert sum(1 for o in ops if o.op == "sub") == 1


def test_alignment_handles_deletion():
    ops = align_indices(["d", "ɹ", "i", "m"], ["d", "ɹ", "i"])
    assert any(o.op == "del" for o in ops)
