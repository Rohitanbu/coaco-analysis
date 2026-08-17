# Build Prompt: Backend API + Sound Generation (Cocoa Ripeness Analyser)

> Paste into Cursor / Antigravity. This wires the `templatemo_614_quantix_saas/analyser.html` frontend (already built) to a real backend: SVM prediction + an actual playable audio file generated from the acoustic signal, replacing `MOCK_RESULT` in `templatemo-quantix-analyser.js`.

---

## Context

- Frontend already exists and expects: a prediction label + confidence, a ranked list of "other possibilities," extracted features (Amplitude, RMS, Power, Frequency), a waveform to render on canvas, and playable audio.
- Signal feature math (Amplitude/RMS/Power) and the TDMS parsing quirks (positional columns, BOM, day-first datetime) were already scoped in the earlier signal-processing module — reuse that logic here rather than re-deriving it.
- SVM model design (features, training process) was scoped in the project's SDLC report — this prompt assumes a trained model file already exists or will be trained via a separate offline script; this prompt covers **serving** it, not training it from scratch. If no trained model exists yet, stub prediction with a clearly-labeled placeholder classifier (e.g., `DummyClassifier` returning random-but-plausible probabilities) so the API contract can be built and tested end-to-end before the real model is ready — but the code must make it obvious where the swap-in point is.

## Step 1 — API Contract (define this first, before writing endpoint code)

**`POST /api/predict`** — multipart form data: `thermal_image` (jpg), `acoustic_file` (tdms)

Response JSON (single response, backend always returns the full probability distribution — the frontend, not the backend, decides whether to show the "Other Possibilities" panel based on the 85% threshold already implemented client-side):

```json
{
  "run_id": "uuid-string",
  "prediction": { "label": "Ripe", "confidence": 0.82 },
  "probabilities": [
    { "label": "Ripe", "confidence": 0.82 },
    { "label": "Unripe", "confidence": 0.10 },
    { "label": "Overripe", "confidence": 0.08 }
  ],
  "features": {
    "amplitude": 1.874,
    "rms": 0.0576,
    "power": 0.00331,
    "frequency": null
  },
  "waveform": {
    "sample_rate_hz": 4820.3,
    "samples": [/* downsampled array, ~500-1000 points, for canvas rendering */]
  },
  "audio_url": "/api/audio/<run_id>.wav"
}
```

Notes on fields:
- `probabilities` is always sorted descending by confidence and always includes every class the model was trained on — this keeps the threshold-display logic entirely in the frontend (already implemented) rather than duplicating it server-side.
- `features.frequency` — flag this explicitly to the person: the original signal-processing scope (Amplitude/RMS/Power) never defined a "Frequency" metric, and computing one legitimately requires FFT/frequency-domain analysis, which was explicitly marked out of scope. Either (a) drop `frequency` from the UI, or (b) confirm with the project owner that a specific frequency-domain metric should be added to scope, and define exactly which one (dominant frequency? zero-crossing rate converted to Hz?) before implementing. Do not silently fabricate a number for this field.
- `waveform.samples` is a **downsampled** array for the canvas chart (send ~500–1000 points regardless of the original file's row count, using e.g. every Nth sample or min/max-per-bucket downsampling) — do not send tens of thousands of raw points to the browser.
- `audio_url` points to a separate endpoint (Step 3) rather than embedding base64 audio in this JSON payload, keeping the prediction response small and cacheable independently of the audio file.

**`GET /api/audio/<run_id>.wav`** — streams the generated WAV file (Step 3) with `Content-Type: audio/wav`.

## Step 2 — Prediction Endpoint

Build `/api/predict` in `app.py` (or a blueprint, if the existing Flask app is already organized that way):

1. Validate both uploads (reuse the existing positional-column, BOM-safe TDMS parsing logic and the JPG validation already scoped).
2. Run the existing feature extraction (`utils/signal_processing.py`) to get Amplitude/RMS/Power.
3. Load the thermal image, extract its auxiliary feature(s) as scoped in the SDLC report (Section 6.2).
4. Assemble the fused feature vector and pass it to the loaded SVM model (or the labeled stub, per Context above) to get `predict_proba` output.
5. Map class indices to labels, sort descending, build the `probabilities` array.
6. Downsample the raw voltage array for the `waveform.samples` field.
7. Trigger audio generation (Step 3), store the resulting file, and build `audio_url`.
8. Persist the run to the existing SQLite `analysis_runs` table (extend the schema with a `predicted_label` and `confidence` column if not already present).
9. Return the JSON contract from Step 1.

## Step 3 — Sound Generation Module (`utils/audio_generator.py`)

**Goal:** convert the raw voltage array from the acoustic file into an actual playable `.wav` file — this is the missing piece behind the frontend's Web Audio API playback, which currently has nothing real to play.

Requirements:
1. **Derive the true sample rate from the data, don't assume one.** The TDMS/CSV `Time` column has real timestamps — compute the time delta between consecutive samples and take the median (more robust to jitter/outliers than the mean) to get `dt`, then `sample_rate_hz = 1 / dt`. This is the `sample_rate_hz` value returned in the API contract.
2. **Normalize amplitude for playback.** Raw voltage values (small floats, possibly with DC offset) need to be scaled to fit 16-bit PCM range. After DC-offset removal (reuse existing logic), normalize so the peak absolute value maps to just under the int16 max (e.g., scale to ±32000, leaving headroom to avoid clipping), then cast to `int16`.
3. **Handle unusual sample rates gracefully.** If the derived sample rate is very low (e.g., under ~2000 Hz) or very high (e.g., over 96000 Hz), still write the WAV at the true derived rate — do not silently resample to a "nicer" number like 44100 Hz, since that would distort playback speed/pitch relative to the real recording. Only resample if explicitly asked for a "normalized to standard rate" mode later; log the actual rate used either way so it's visible for debugging.
4. **Write the WAV file** using `scipy.io.wavfile.write` (or the standard library `wave` module if avoiding the SciPy dependency here is preferred — pick one and be consistent with what's already imported elsewhere in the project) to `static/audio/<run_id>.wav`.
5. Return the file path (and/or sample rate actually used) so the calling endpoint can build `audio_url`.

**Acceptance check:** generate a WAV from a real sample TDMS file, play it back locally (or inspect via `soundfile`/`scipy.io.wavfile.read`) and confirm: the file is a valid WAV, its duration roughly matches `(number of samples) / sample_rate_hz`, and there's no obvious clipping (no samples pinned at ±32767 unless the source signal was already clipped).

## Step 4 — Wire the Frontend

- Replace `MOCK_RESULT` in `templatemo-quantix-analyser.js` with an actual `fetch('/api/predict', { method: 'POST', body: formData })` call, matching the JSON contract in Step 1 field-for-field.
- Point the existing play/pause control's audio element or Web Audio API source at `audio_url` instead of any mock/synthesized tone.
- Feed `waveform.samples` into the existing canvas renderer in place of the mock array — confirm the renderer still looks correct with real (downsampled) data rather than the clean mock shape.
- Re-verify the two confidence branches (< 85% and ≥ 85%) still behave correctly now that `probabilities` is coming from a live API response instead of a hardcoded mock value.

## Step 5 — Testing

- Unit test `audio_generator.py`'s sample-rate derivation against a TDMS file with known, hand-checked timestamps.
- Integration test: POST a real sample pair to `/api/predict`, confirm the response matches the Step 1 contract shape, confirm `audio_url` resolves to a playable file.
- Manual browser test: full upload → predict → waveform renders → audio plays → confidence panel shows/hides correctly at both sides of the 85% threshold with real model output.

---

**Instruction to the agent:** Do Step 1 first and just show me the finalized contract for confirmation before writing any endpoint or audio code — in particular, tell me your decision on the `frequency` field (drop it, or propose a specific definition) rather than guessing.
