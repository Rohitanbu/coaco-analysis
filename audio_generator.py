"""Convert raw voltage arrays to playable WAV files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

INT16_HEADROOM = 32000


def normalize_for_playback(voltage: np.ndarray) -> np.ndarray:
    """Remove DC offset and scale peak to int16 headroom."""
    x = voltage.astype(np.float64)
    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return np.zeros(len(x), dtype=np.int16)
    scaled = x / peak * INT16_HEADROOM
    clipped = np.clip(scaled, -32767, 32767)
    return clipped.astype(np.int16)


def write_wav(voltage: np.ndarray, sample_rate_hz: float, output_path: Path) -> Path:
    """Write a mono 16-bit PCM WAV at the true derived sample rate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = normalize_for_playback(voltage)
    rate = int(round(sample_rate_hz))
    if rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate_hz}")

    if rate < 2000 or rate > 96000:
        logger.warning("Unusual sample rate %.2f Hz — writing WAV without resampling", rate)

    wavfile.write(str(output_path), rate, pcm)
    logger.info("Wrote WAV %s (%d samples @ %d Hz)", output_path.name, len(pcm), rate)
    return output_path
