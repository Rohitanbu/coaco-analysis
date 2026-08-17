"""Integration tests for /api/predict."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "Acoustics_Extracted/Acoustics/CPB/CPB19-1 (3).csv"
SAMPLE_TDMS = ROOT / "Acoustics_Extracted/Acoustics/CPB/CPB19-1 (3).tdms"
SAMPLE_JPG = ROOT / "Acoustics_Extracted/Thermal Images/CPB/CPB67F.jpg"


@pytest.fixture(scope="module")
def client():
    from app import app

    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(
    not (SAMPLE_CSV.exists() and SAMPLE_JPG.exists()),
    reason="sample thermal + CSV not available locally",
)
def test_predict_dual_upload_contract(client: TestClient):
    files = {
        "thermal_image": (SAMPLE_JPG.name, SAMPLE_JPG.read_bytes(), "image/jpeg"),
        "acoustic_file": (SAMPLE_CSV.name, SAMPLE_CSV.read_bytes(), "text/csv"),
    }
    res = client.post("/api/predict", files=files)
    assert res.status_code == 200, res.text
    data = res.json()

    assert "run_id" in data
    assert "prediction" in data and "label" in data["prediction"]
    assert "probabilities" in data and len(data["probabilities"]) >= 2
    assert data["probabilities"][0]["confidence"] >= data["probabilities"][1]["confidence"]
    assert set(data["features"]) >= {"amplitude", "rms", "power", "frequency"}
    assert data["features"]["frequency"] is None
    assert "waveform" in data and len(data["waveform"]["samples"]) >= 100
    assert data["audio_url"].startswith("/api/audio/")

    audio = client.get(data["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")


@pytest.mark.skipif(
    not (SAMPLE_TDMS.exists() and SAMPLE_JPG.exists()),
    reason="sample thermal + TDMS not available locally",
)
def test_predict_tdms_upload(client: TestClient):
    files = {
        "thermal_image": (SAMPLE_JPG.name, SAMPLE_JPG.read_bytes(), "image/jpeg"),
        "acoustic_file": (SAMPLE_TDMS.name, SAMPLE_TDMS.read_bytes(), "application/octet-stream"),
    }
    res = client.post("/api/predict", files=files)
    assert res.status_code == 200, res.text
