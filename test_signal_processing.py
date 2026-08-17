"""Tests for signal processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.signal_processing import (
    extract_features,
    parse_acoustic_file,
    parse_day_first_datetime,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "Acoustics_Extracted/Acoustics/CPB/CPB19-1 (3).csv"
SAMPLE_TDMS = ROOT / "Acoustics_Extracted/Acoustics/CPB/CPB19-1 (3).tdms"


def test_parse_day_first_datetime():
    ts = parse_day_first_datetime("18/12/2025 10:18:46.205000")
    assert ts > 0


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample CSV not available locally")
def test_csv_sample_rate_near_12800():
    data = SAMPLE_CSV.read_bytes()
    signal = parse_acoustic_file(data, SAMPLE_CSV.name)
    assert len(signal.voltage) > 1000
    assert 12000 <= signal.sample_rate_hz <= 13500


@pytest.mark.skipif(not SAMPLE_TDMS.exists(), reason="sample TDMS not available locally")
def test_tdms_sample_rate_from_wf_increment():
    data = SAMPLE_TDMS.read_bytes()
    signal = parse_acoustic_file(data, SAMPLE_TDMS.name)
    assert len(signal.voltage) > 1000
    assert 12000 <= signal.sample_rate_hz <= 13500


def test_extract_features_positive():
    x = np.sin(np.linspace(0, 20 * np.pi, 5000)) * 0.05 + 0.01
    feats = extract_features(x)
    assert feats.amplitude > 0
    assert feats.rms > 0
    assert feats.power > 0
