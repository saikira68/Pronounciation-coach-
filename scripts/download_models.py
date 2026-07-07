"""Pre-download the STT and phoneme models at build time so runtime is fast."""

import os

from faster_whisper import WhisperModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

WHISPER = os.environ.get("WHISPER_MODEL", "base.en")
PHONEME = os.environ.get("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")

if __name__ == "__main__":
    print(f"Downloading Whisper model: {WHISPER}")
    WhisperModel(WHISPER, device="cpu", compute_type="int8")
    print(f"Downloading phoneme model: {PHONEME}")
    Wav2Vec2Processor.from_pretrained(PHONEME)
    Wav2Vec2ForCTC.from_pretrained(PHONEME)
    print("Done.")
