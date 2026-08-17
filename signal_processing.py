"""Acoustic signal parsing and feature extraction for cocoa pod analysis."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

try:
    from nptdms import TdmsFile
except ImportError:  # pragma: no cover - optional at import time
    TdmsFile = None  # type: ignore[misc, assignment]

# Day-first timestamps: 18/12/2025 10:18:46.205000
_TIME_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$"
)


@dataclass(frozen=True)
class AcousticSignal:
    """Parsed acoustic recording."""

    voltage: np.ndarray
    sample_rate_hz: float
    timestamps_s: np.ndarray | None = None


@dataclass(frozen=True)
class SignalFeatures:
    amplitude: float
    rms: float
    power: float


def parse_day_first_datetime(value: str) -> float:
    """Parse day-first datetime string to seconds since epoch."""
    value = value.strip().lstrip("\ufeff")
    match = _TIME_RE.match(value)
    if not match:
        raise ValueError(f"Unrecognised timestamp format: {value!r}")

    day, month, year, hour, minute, second, frac = match.groups()
    micro = int((frac or "0").ljust(6, "0")[:6])
    dt = datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
        micro,
    )
    return dt.timestamp()


def _median_sample_rate(timestamps_s: np.ndarray) -> float:
    if len(timestamps_s) < 2:
        raise ValueError("Need at least two timestamps to derive sample rate")
    dts = np.diff(timestamps_s)
    dts = dts[dts > 0]
    if len(dts) == 0:
        raise ValueError("Non-positive time deltas in acoustic file")
    dt = float(np.median(dts))
    return 1.0 / dt


def parse_csv_acoustic(data: bytes) -> AcousticSignal:
    """Parse exported CSV with positional columns: time (col 0), voltage (col 1)."""
    text = data.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # header row

    timestamps: list[float] = []
    voltage: list[float] = []

    for row in reader:
        if len(row) < 2:
            continue
        try:
            timestamps.append(parse_day_first_datetime(row[0]))
            voltage.append(float(row[1]))
        except (ValueError, IndexError):
            continue

    if not voltage:
        raise ValueError("No voltage samples found in CSV acoustic file")

    ts = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(voltage, dtype=np.float64)
    sample_rate_hz = _median_sample_rate(ts)
    return AcousticSignal(voltage=v, sample_rate_hz=sample_rate_hz, timestamps_s=ts)


def parse_tdms_acoustic(data: bytes) -> AcousticSignal:
    """Parse TDMS acoustic file using wf_increment when available."""
    if TdmsFile is None:
        raise RuntimeError(
            "TDMS support requires nptdms (pip install nptdms). "
            "Upload CSV export instead."
        )

    tdms = TdmsFile.read(io.BytesIO(data))
    channel = None
    for group in tdms.groups():
        for ch in group.channels():
            if len(ch) > 0:
                channel = ch
                break
        if channel is not None:
            break

    if channel is None:
        raise ValueError("No data channels found in TDMS file")

    voltage = np.asarray(channel[:], dtype=np.float64)
    wf_increment = channel.properties.get("wf_increment")
    if wf_increment and float(wf_increment) > 0:
        sample_rate_hz = 1.0 / float(wf_increment)
    else:
        sample_rate_hz = 12800.0

    return AcousticSignal(
        voltage=voltage,
        sample_rate_hz=float(sample_rate_hz),
        timestamps_s=None,
    )


def parse_acoustic_file(data: bytes, filename: str) -> AcousticSignal:
    """Parse acoustic upload (.tdms or .csv)."""
    name = filename.lower()
    if name.endswith(".tdms"):
        return parse_tdms_acoustic(data)
    if name.endswith(".csv"):
        return parse_csv_acoustic(data)
    raise ValueError("Acoustic file must be .tdms or .csv")


def extract_features(voltage: np.ndarray) -> SignalFeatures:
    """Compute Amplitude (peak), RMS, and Power after DC removal."""
    x = voltage.astype(np.float64)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x**2)))
    amplitude = float(np.max(np.abs(x)))
    power = float(np.mean(x**2))
    return SignalFeatures(amplitude=amplitude, rms=rms, power=power)


def downsample_waveform(voltage: np.ndarray, target_points: int = 800) -> list[float]:
    """Min/max bucket downsampling for canvas rendering (~500–1000 points)."""
    x = np.asarray(voltage, dtype=np.float64)
    n = len(x)
    if n == 0:
        return []
    if n <= target_points:
        return x.tolist()

    bucket_size = n / target_points
    out: list[float] = []
    for i in range(target_points):
        start = int(i * bucket_size)
        end = max(start + 1, int((i + 1) * bucket_size))
        segment = x[start:end]
        if i % 2 == 0:
            out.append(float(segment.max()))
        else:
            out.append(float(segment.min()))
    return out
