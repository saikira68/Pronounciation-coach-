# Livo Pronunciation Coach — System Architecture

A web app where a user uploads 30–45 seconds of English speech and receives an overall
pronunciation score plus per-word highlighting of the specific sounds that were off.

## 1. Components & data flow

```
 Browser (static SPA)                     Server (FastAPI, single Docker image)
 ─────────────────────                    ─────────────────────────────────────
 • drag/drop upload                        POST /api/assess (multipart audio)
 • decodeAudioData() ──── duration ok? ──▶  ├─ decode + resample → 16 kHz mono  (ffmpeg/libsndfile)
 • consent gate (DPDP)                      ├─ enforce 30–45 s (server-side truth)
 • render score + colour-                   ├─ faster-whisper  → words + timestamps
   coded transcript  ◀──── JSON result ───  ├─ wav2vec2 CTC    → phonemes actually spoken
                                            ├─ espeak-ng (G2P) → canonical phonemes / word
                                            ├─ global phoneme alignment → per-word score
                                            └─ aggregate → overall + fluency + issues
                                            (audio buffers dropped; nothing persisted)
```

Everything is one stateless container. The frontend is plain HTML/CSS/JS served by the same
FastAPI process — no build step, so it "just works" in a browser. Deployed to **Hugging Face
Spaces** (Docker runtime, free public URL).

## 2. Models & APIs — and why over the alternatives

| Stage | Choice | Why, vs. alternatives |
| ----- | ------ | --------------------- |
| Speech-to-text | **faster-whisper `base.en`** (CTranslate2) | Robust, punctuated transcripts with word timestamps; int8 CPU inference is ~4× faster than reference Whisper, so no GPU is needed. `tiny` was less accurate on connected speech; `small`+ was slower than the UX warranted. |
| Phoneme recognition | **`facebook/wav2vec2-lv-60-espeak-cv-ft`** | Emits **espeak/IPA phonemes directly**, so it shares an inventory with the reference G2P and needs no phoneme mapping table. It recognises the sounds *actually produced*, independent of the "correct" word — which is exactly what pronunciation scoring needs. |
| Reference G2P | **espeak-ng** via `phonemizer` | Deterministic, offline, no per-call cost, same IPA space as the acoustic model. A CMUdict lookup was considered but uses ARPAbet and misses OOV words; espeak handles any token. |
| Scoring | In-house alignment (no ML) | Transparent, debuggable, tunable. |

**Why not a managed API (Azure Pronunciation Assessment, Speechace, etc.)?** They are excellent
and would be my first pick for a paid production build, but they add cost, a third-party data
processor (a DPDP complication), and vendor lock-in. The assessment asked for a deliberate stack;
a self-hosted pipeline keeps **all audio on our own server**, which makes the compliance story far
cleaner and costs nothing to run.

## 3. How scoring & highlighting work

1. **Transcribe** to get the words the learner said (and rough timestamps).
2. **Recognise phonemes** acoustically over the whole clip (CTC argmax + blank-collapse with
   frame-level timing).
3. **Canonicalise**: convert each recognised word to its expected IPA phonemes with espeak.
4. **Align**: one global Needleman–Wunsch alignment between the full canonical sequence and the
   full recognised sequence. A single global alignment (rather than per-word time-slicing) is used
   because Whisper word timestamps drift by ~50–100 ms, which was mislabelling correctly-spoken
   words; alignment absorbs that drift.
5. **Per-word accuracy** = matched phonemes / expected phonemes, mapped back to words via the
   canonical word boundaries. Substitutions and deletions become human-readable issues
   ("Said /d/ where /ð/ was expected", "Dropped the /m/ sound").
6. **Overall score** = phoneme-count-weighted mean of per-word accuracy. A **fluency** proxy from
   speaking rate is reported separately.

**Highlighting rule:** `good ≥ 80`, else `mispronounced`; low acoustic confidence or no detected
phonemes → `unclear`. To avoid punishing natural speech, phoneme **equivalence classes** (schwa
family, t-flapping, tense/lax `i~ɪ`, rhotics, diphthong merging) count as matches — so connected-
speech reductions aren't reported as errors.

## 4. DPDP compliance (Digital Personal Data Protection Act, 2023)

The uploaded voice recording is **personal data**, so the app is designed data-minimal by default:

- **Consent (§6):** processing is gated behind an explicit, unbundled consent checkbox stating the
  specific purpose (compute a pronunciation score). No consent → the Analyze button stays disabled.
- **Purpose limitation & minimisation (§5, §6):** audio is used *only* to produce the score. No
  profiling, no training, no secondary use.
- **Storage & retention:** audio is held **in memory only** for the duration of the request and the
  buffers are dropped immediately after the response. It is **never written to disk, logged, or
  cached**, so retention is effectively zero — satisfying storage-limitation and erasure duties by
  construction. The result JSON contains no raw audio.
- **Data residency:** all inference is self-hosted in one container; audio is **not sent to any
  third-party processor**. For a production India deployment the same image can run in an
  India region (e.g. HF Spaces / a cloud VM in `ap-south-1`) to keep data in-country.
- **Deletion / data-principal rights:** because nothing is retained there is nothing to delete after
  a request; closing the tab clears the browser-side copy. A production build would add a documented
  erasure endpoint and a Consent Manager only if any storage were introduced.
- **Transparency (§5):** an in-app privacy notice states purpose, retention, and that no third party
  receives the audio.

**Production hardening I'd add:** serve over HTTPS/TLS (the host already provides this), a published
privacy policy + Grievance Officer contact, rate limiting, and — only if features required storing
audio — encryption at rest, a fixed retention window with auto-purge, and a consent-withdrawal /
deletion API.

## 5. Trade-offs made

- **Self-hosted OSS over a managed pronunciation API:** cheaper, better privacy posture, fully
  controllable — at the cost of some raw accuracy and more engineering. A deliberate fit for the
  brief and the DPDP emphasis.
- **CPU-only, `base.en` model:** ~8–12 s per 45 s clip, which keeps hosting free and simple; a GPU
  or a distilled model would cut latency.
- **Free-speech (unscripted) scoring:** more realistic than "read this sentence", but the reference
  is the *recognised* transcript, so a confidently mis-recognised word can shift its own reference.
- **Known limitations:** homographs (espeak picks one pronunciation, e.g. "live" /laɪv/ vs /lɪv/) and
  heavy connected-speech reduction of function words can still produce occasional false flags;
  utterance-final unreleased stops are sometimes marked as dropped.

## 6. What I'd build next with another week

- **Reference-text mode:** let learners read a known passage → forced alignment for far more precise,
  reliable per-phoneme scoring alongside the free-speech mode.
- **Prosody & fluency:** pitch/energy/pause modelling and filler-word detection, not just word rate.
- **Actionable coaching:** per-phoneme tips, minimal-pair drills, and TTS playback of the correct
  pronunciation for each flagged word.
- **Homograph disambiguation** via POS tagging, and a GPU/quantised model for sub-second latency.
- **Accounts + progress tracking** (which would then require the full DPDP storage controls above:
  encryption at rest, retention windows, and a deletion/withdrawal API).
