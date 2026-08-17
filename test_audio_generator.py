"""Tests for WAV generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from utils.audio_generator import write_wav

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "Acoustics_Extracted/Acoustics/CPB/CPB19-1 (3).csv"


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not available locally")
def test_wav_duration_matches_sample_count(tmp_path: Path):
    from utils.signal_processing import parse_acoustic_file

    signal = parse_acoustic_file(SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name)
    out = tmp_path / "test.wav"
    write_wav(signal.voltage, signal.sample_rate_hz, out)

    rate, pcm = wavfile.read(out)
    expected_duration = len(signal.voltage) / signal.sample_rate_hz
    actual_duration = len(pcm) / rate
    assert abs(expected_duration - actual_duration) < 0.05
    assert pcm.dtype == np.int16
    assert np.max(np.abs(pcm)) <= 32767


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not available locally")
def test_wav_not_fully_clipped(tmp_path: Path):
    from utils.signal_processing import parse_acoustic_file

    signal = parse_acoustic_file(SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name)
    out = tmp_path / "clip.wav"
    write_wav(signal.voltage, signal.sample_rate_hz, out)
    _, pcm = wavfile.read(out)
    pinned = np.sum((pcm == 32767) | (pcm == -32767))
    assert pinned / len(pcm) < 0.01
