FROM python:3.11-slim

# System deps: espeak-ng (canonical phonemes for the reference pronunciation),
# ffmpeg + libsndfile (audio decoding/resampling).
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libespeak-ng1 ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (Hugging Face Spaces convention: uid 1000).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 \
    OMP_NUM_THREADS=4 \
    PORT=7860

USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts

# Bake the models into the image so the first request is fast and the Space works
# offline / without hitting the Hub at runtime.
RUN python scripts/download_models.py

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
