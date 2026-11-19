"""Heuristic identification of common narrowband modes from demodulated audio."""

from __future__ import annotations

import numpy as np
from scipy import signal

KIWI_SAMPLE_RATE = 12_000
MIN_SAMPLE_COUNT = 4 * KIWI_SAMPLE_RATE
MAX_SAMPLE_COUNT = 20 * KIWI_SAMPLE_RATE


def _unknown_result(duration: float, rms: float, reason: str) -> dict:
    return {
        "classification": "unknown",
        "best_match": None,
        "candidates": [],
        "evidence": [reason],
        "features": {
            "duration_seconds": round(duration, 1),
            "rms": round(rms, 4),
        },
        "notice": "Audio-only heuristic; no payload decoding is performed.",
    }


def analyze_pcm(samples: np.ndarray, sample_rate: int = KIWI_SAMPLE_RATE, freq_khz: float | None = None) -> dict:
    """Rank likely audio-mode families from signed 16-bit KiwiSDR PCM samples."""
    if sample_rate != KIWI_SAMPLE_RATE:
        raise ValueError(f"Sample rate must be {KIWI_SAMPLE_RATE} Hz")
    if samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError("Audio must be a one-dimensional int16 PCM array")
    if not MIN_SAMPLE_COUNT <= samples.size <= MAX_SAMPLE_COUNT:
        raise ValueError("Audio must be between 4 and 20 seconds long")

    audio = samples.astype(np.float32) / 32768.0
    duration = samples.size / sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.003:
        return _unknown_result(duration, rms, "Audio level is too low for reliable mode analysis.")

    freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(4096, samples.size))
    audio_band = (freqs >= 250) & (freqs <= 3_200)
    band_power = psd[audio_band]
    band_freqs = freqs[audio_band]
    if band_power.size == 0 or np.sum(band_power) <= 0:
        return _unknown_result(duration, rms, "No usable energy was found in the demodulated audio band.")

    cumulative = np.cumsum(band_power)
    cumulative /= cumulative[-1]
    low = float(band_freqs[np.searchsorted(cumulative, 0.05)])
    high = float(band_freqs[min(np.searchsorted(cumulative, 0.95), band_freqs.size - 1)])
    bandwidth = high - low

    _, _, spectrum = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=512,
        noverlap=384,
        nfft=2048,
        boundary=None,
    )
    spectrum_freqs = np.fft.rfftfreq(2048, 1 / sample_rate)
    audio_bins = (spectrum_freqs >= 300) & (spectrum_freqs <= 3_000)
    track = spectrum_freqs[audio_bins][np.argmax(np.abs(spectrum[audio_bins]), axis=0)]

    # Find the two persistent audio tones that characterize common 2-FSK modes.
    histogram_edges = np.arange(300, 3_050, 25)
    counts, edges = np.histogram(track, bins=histogram_edges)
    peaks, _ = signal.find_peaks(counts, distance=3, prominence=max(2, counts.max() * 0.05))
    peak_bins = sorted(peaks, key=lambda index: counts[index], reverse=True)

    candidates: list[dict] = []
    evidence: list[str] = [
        f"Dominant audio energy spans approximately {low:.0f}-{high:.0f} Hz.",
        f"Measured audio level is {rms:.3f} RMS.",
    ]

    if len(peak_bins) >= 2:
        centers = [float((edges[index] + edges[index + 1]) / 2) for index in peak_bins[:2]]
        centers.sort()
        shift = centers[1] - centers[0]
        labels = np.argmin(np.abs(track[:, None] - np.asarray(centers)[None, :]), axis=1)
        distances = np.min(np.abs(track[:, None] - np.asarray(centers)[None, :]), axis=1)
        assigned = distances <= 40
        occupancy = float(np.mean(assigned))
        transitions = int(np.count_nonzero((labels[1:] != labels[:-1]) & assigned[1:] & assigned[:-1]))

        if 80 <= shift <= 300 and occupancy >= 0.55 and transitions >= 8:
            is_170_shift = abs(shift - 170) <= 30
            name = f"RTTY-like 2-FSK (~{shift:.0f} Hz shift)" if is_170_shift else f"2-FSK data (~{shift:.0f} Hz shift)"
            confidence = min(0.82, 0.48 + min(occupancy, 0.95) * 0.18 + min(transitions / 80, 1) * 0.12)
            mode_id = "rtty_2fsk_170" if is_170_shift else "2fsk"
            candidates.append(
                {
                    "mode_id": mode_id,
                    "name": name,
                    "category": "digital FSK",
                    "confidence": round(confidence, 2),
                    "description": (
                        "A two-tone frequency-shift-keyed signal. The ~170 Hz shift is consistent with common "
                        "amateur RTTY; baud rate and character encoding are not decoded."
                        if is_170_shift
                        else "A two-tone frequency-shift-keyed signal; its protocol and baud rate are not identified."
                    ),
                    "evidence": [
                        f"Two persistent tones near {centers[0]:.0f} and {centers[1]:.0f} Hz.",
                        f"Tone separation is approximately {shift:.0f} Hz with repeated switching.",
                    ],
                }
            )
            evidence.extend(candidates[0]["evidence"])

            if freq_khz is not None and any(abs(freq_khz - navtex_freq) <= 1 for navtex_freq in (490, 518)):
                candidates.append(
                    {
                        "mode_id": "sitor_b_navtex_candidate",
                        "name": "SITOR-B / NAVTEX candidate",
                        "category": "maritime digital FSK",
                        "confidence": round(min(confidence, 0.68), 2),
                        "description": (
                            "SITOR-B is a maritime text protocol used for NAVTEX broadcasts. "
                            "The tuned frequency is supporting context, not protocol confirmation."
                        ),
                        "evidence": [f"Receiver is tuned near the NAVTEX {freq_khz:.0f} kHz channel."],
                    }
                )
                candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)

    if not candidates and bandwidth >= 900:
        candidates.append(
            {
                "mode_id": "wideband_audio",
                "name": "Wideband audio / possible voice",
                "category": "voice or noise",
                "confidence": 0.45,
                "description": "Broad audio energy without a supported narrowband digital-mode fingerprint.",
                "evidence": [f"Audio occupies a relatively broad {bandwidth:.0f} Hz span."],
            }
        )
        evidence.extend(candidates[0]["evidence"])

    best_match = candidates[0] if candidates else None
    return {
        "classification": best_match["category"] if best_match else "unknown",
        "best_match": best_match,
        "candidates": candidates,
        "evidence": evidence if best_match else ["No supported mode fingerprint was strong enough to rank."],
        "features": {
            "duration_seconds": round(duration, 1),
            "rms": round(rms, 4),
            "audio_bandwidth_hz": round(bandwidth),
            "dominant_tone_hz": round(float(np.median(track))),
        },
        "notice": "Audio-only heuristic; no payload decoding is performed.",
    }
