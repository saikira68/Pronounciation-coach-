"""Pronunciation scoring pipeline.

Pipeline
--------
1. Decode + resample the upload to 16 kHz mono (ffmpeg via librosa).
2. Transcribe with faster-whisper -> words with start/end timestamps.
3. Recognise the *actual* phonemes the speaker produced with a wav2vec2 CTC
   phoneme model, keeping frame-level timing so phonemes can be bucketed per word.
4. For every word, compare the canonical (espeak) phonemes against the recognised
   phonemes via edit-distance alignment -> per-word accuracy + labelled issues.
5. Aggregate into an overall 0-100 pronunciation score plus fluency signal.

Everything runs in-process; no audio is written to disk or sent to a third party.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .phonemes import align_indices, words_to_phonemes

SAMPLE_RATE = 16000

# Lazily-initialised heavy models (loaded once, on first request).
_whisper = None
_w2v_model = None
_w2v_processor = None
_w2v_blank_id = 0


def _load_models():
    global _whisper, _w2v_model, _w2v_processor, _w2v_blank_id
    if _whisper is None:
        from faster_whisper import WhisperModel
        import os

        _whisper = WhisperModel(
            os.environ.get("WHISPER_MODEL", "base.en"),
            device="cpu",
            compute_type="int8",
        )
    if _w2v_model is None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import os

        name = os.environ.get("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
        _w2v_processor = Wav2Vec2Processor.from_pretrained(name)
        model = Wav2Vec2ForCTC.from_pretrained(name)
        model.eval()
        _w2v_blank_id = model.config.pad_token_id or 0
        # Dynamic int8 quantization of the Linear layers roughly halves CPU
        # inference time (this model dominates latency) with negligible accuracy
        # loss. Disable with QUANTIZE=0.
        if os.environ.get("QUANTIZE", "1") == "1":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        _w2v_model = model
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return _whisper, _w2v_model, _w2v_processor


def warm_up() -> None:
    """Load models eagerly (called at startup) so the first request is fast."""
    _load_models()


def load_audio(data: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 @ 16 kHz."""
    import soundfile as sf
    import librosa

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        # Fallback for formats soundfile can't open (mp3/m4a/webm) -> librosa/audioread/ffmpeg.
        audio, sr = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        return audio.astype("float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype("float32")


def audio_duration(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE


@dataclass
class FramePhoneme:
    symbol: str
    start: float
    end: float
    prob: float


def _recognize_phonemes(audio: np.ndarray) -> List[FramePhoneme]:
    """Frame-level phoneme recognition with timestamps via CTC argmax collapse."""
    import torch

    _, model, processor = _load_models()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]  # [T, V]
    probs = torch.softmax(logits, dim=-1)
    ids = torch.argmax(logits, dim=-1)
    max_probs = probs[torch.arange(len(ids)), ids]

    n_frames = logits.shape[0]
    seconds_per_frame = audio_duration(audio) / max(n_frames, 1)
    vocab = processor.tokenizer.convert_ids_to_tokens(list(range(logits.shape[1])))

    out: List[FramePhoneme] = []
    prev = _w2v_blank_id
    run_probs: List[float] = []
    run_start = 0
    for t in range(n_frames):
        cur = int(ids[t])
        if cur != prev:
            if prev != _w2v_blank_id:
                sym = vocab[prev]
                out.append(
                    FramePhoneme(
                        symbol=sym,
                        start=run_start * seconds_per_frame,
                        end=t * seconds_per_frame,
                        prob=float(np.mean(run_probs)) if run_probs else 0.0,
                    )
                )
            run_start = t
            run_probs = []
        run_probs.append(float(max_probs[t]))
        prev = cur
    if prev != _w2v_blank_id:
        out.append(
            FramePhoneme(
                symbol=vocab[prev],
                start=run_start * seconds_per_frame,
                end=n_frames * seconds_per_frame,
                prob=float(np.mean(run_probs)) if run_probs else 0.0,
            )
        )
    # Normalise symbols (strip espeak stress/length markers).
    from .phonemes import normalize_phoneme_string

    cleaned: List[FramePhoneme] = []
    for fp in out:
        s = normalize_phoneme_string(fp.symbol)
        if s:
            cleaned.append(FramePhoneme(s, fp.start, fp.end, fp.prob))
    return cleaned


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    score: float
    label: str  # "good" | "mispronounced" | "unclear" | "missing"
    expected_phonemes: List[str]
    heard_phonemes: List[str]
    issues: List[str] = field(default_factory=list)


@dataclass
class AssessResult:
    overall_score: int
    transcript: str
    duration: float
    words: List[WordResult]
    summary: Dict[str, int]
    fluency_score: int

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _label_for(score: float, confident: bool, has_ref: bool) -> str:
    if not has_ref:
        return "good"
    if not confident:
        return "unclear"
    if score >= 80:
        return "good"
    if score >= 55:
        return "mispronounced"
    return "mispronounced"


def assess(data: bytes) -> AssessResult:
    audio = load_audio(data)
    duration = audio_duration(audio)
    whisper, _, _ = _load_models()

    segments, _info = whisper.transcribe(
        audio, language="en", word_timestamps=True, vad_filter=True
    )

    words_meta = []
    transcript_parts: List[str] = []
    for seg in segments:
        transcript_parts.append(seg.text.strip())
        for w in seg.words or []:
            words_meta.append(
                {
                    "word": w.word.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                    "prob": float(getattr(w, "probability", 1.0) or 1.0),
                }
            )
    transcript = " ".join(t for t in transcript_parts if t).strip()

    frames = _recognize_phonemes(audio)
    heard_syms = [f.symbol for f in frames]

    # Context-aware canonical phonemes for every recognised word.
    word_phon = words_to_phonemes([wm["word"] for wm in words_meta])

    # Build the canonical phoneme sequence for the whole utterance while
    # remembering which word each phoneme belongs to. Scoring is then a single
    # global alignment, which is robust to word-timestamp drift (the earlier
    # per-word time-bucketing approach mislabelled correctly-spoken words).
    scored_idx = [i for i in range(len(words_meta)) if word_phon[i]]
    canon_syms: List[str] = []
    canon_word: List[int] = []  # word index (into words_meta) per canonical phoneme
    for i in scored_idx:
        for ph in word_phon[i]:
            canon_syms.append(ph)
            canon_word.append(i)

    # per-word accumulators
    matches: Dict[int, int] = {i: 0 for i in scored_idx}
    subs: Dict[int, list] = {i: [] for i in scored_idx}
    dels: Dict[int, list] = {i: [] for i in scored_idx}
    heard_by_word: Dict[int, list] = {i: [] for i in scored_idx}
    prob_by_word: Dict[int, list] = {i: [] for i in scored_idx}

    ops = align_indices(canon_syms, heard_syms) if canon_syms else []
    last_word: Optional[int] = scored_idx[0] if scored_idx else None
    for op in ops:
        if op.ref_idx is not None:
            w = canon_word[op.ref_idx]
            last_word = w
            if op.op == "match":
                matches[w] += 1
                heard_by_word[w].append(heard_syms[op.hyp_idx])
                prob_by_word[w].append(frames[op.hyp_idx].prob)
            elif op.op == "sub":
                subs[w].append((canon_syms[op.ref_idx], heard_syms[op.hyp_idx]))
                heard_by_word[w].append(heard_syms[op.hyp_idx])
                prob_by_word[w].append(frames[op.hyp_idx].prob)
            elif op.op == "del":
                dels[w].append(canon_syms[op.ref_idx])
        else:  # insertion: extra sound, attribute to current word
            if last_word is not None:
                heard_by_word[last_word].append(heard_syms[op.hyp_idx])
                prob_by_word[last_word].append(frames[op.hyp_idx].prob)

    results: List[WordResult] = []
    total_weight = 0.0
    weighted_score = 0.0
    summary = {"good": 0, "mispronounced": 0, "unclear": 0, "missing": 0}

    for i in scored_idx:
        wm = words_meta[i]
        expected = list(word_phon[i])
        denom = max(len(expected), 1)
        acc = max(0.0, min(100.0, 100.0 * matches[i] / denom))
        mean_prob = float(np.mean(prob_by_word[i])) if prob_by_word[i] else 0.0
        heard = heard_by_word[i]

        confident = bool(heard) and mean_prob >= 0.30 and wm["prob"] >= 0.25
        if not heard:
            label = "unclear"
        else:
            label = _label_for(acc, confident, True)

        if label == "good":
            summary["good"] += 1
        elif label == "unclear":
            summary["unclear"] += 1
        else:
            summary["mispronounced"] += 1

        issues: List[str] = []
        for ref_ph, hyp_ph in subs[i][:4]:
            issues.append(f"Said /{hyp_ph}/ where /{ref_ph}/ was expected.")
        for ref_ph in dels[i][:3]:
            issues.append(f"Dropped the /{ref_ph}/ sound.")
        if label == "unclear" and not issues:
            issues.append("Segment was acoustically unclear or too quiet.")

        results.append(
            WordResult(
                word=wm["word"], start=wm["start"], end=wm["end"],
                score=round(acc, 1), label=label,
                expected_phonemes=expected, heard_phonemes=heard, issues=issues,
            )
        )
        weighted_score += acc * len(expected)
        total_weight += len(expected)

    overall = int(round(weighted_score / total_weight)) if total_weight else 0

    # Simple fluency proxy: speaking rate vs. a natural band (2-4 words/sec).
    n_words = len([w for w in results])
    wps = n_words / duration if duration else 0
    if 1.8 <= wps <= 3.6:
        fluency = 100
    elif wps < 1.8:
        fluency = int(max(40, 100 - (1.8 - wps) * 40))
    else:
        fluency = int(max(40, 100 - (wps - 3.6) * 30))

    return AssessResult(
        overall_score=overall,
        transcript=transcript,
        duration=round(duration, 2),
        words=results,
        summary=summary,
        fluency_score=fluency,
    )
