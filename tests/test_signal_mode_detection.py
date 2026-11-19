import numpy as np
import pytest

from utils.signal_mode_detection import KIWI_SAMPLE_RATE, analyze_pcm


def _rtty_audio(duration=8, baud=45.45, mark_hz=1_000, shift_hz=170):
    sample_count = int(duration * KIWI_SAMPLE_RATE)
    sample_times = np.arange(sample_count) / KIWI_SAMPLE_RATE
    bit_index = (sample_times * baud).astype(int)
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=bit_index.max() + 1)
    frequencies = np.where(bits[bit_index], mark_hz, mark_hz + shift_hz)
    phase = 2 * np.pi * np.cumsum(frequencies) / KIWI_SAMPLE_RATE
    return np.asarray(12_000 * np.sin(phase), dtype=np.int16)


def test_analyze_pcm_identifies_rtty_like_170_hz_fsk():
    result = analyze_pcm(_rtty_audio())

    assert result["best_match"]["mode_id"] == "rtty_2fsk_170"
    assert result["best_match"]["confidence"] >= 0.5
    assert "baud rate" in result["best_match"]["description"]
    assert any("separation" in item for item in result["best_match"]["evidence"])
    assert "no payload decoding" in result["notice"]


def test_analyze_pcm_adds_navtex_candidate_on_navtex_channel():
    result = analyze_pcm(_rtty_audio(), freq_khz=518)

    assert any(candidate["mode_id"] == "sitor_b_navtex_candidate" for candidate in result["candidates"])


def test_analyze_pcm_does_not_identify_silence():
    result = analyze_pcm(np.zeros(4 * KIWI_SAMPLE_RATE, dtype=np.int16))

    assert result["classification"] == "unknown"
    assert result["best_match"] is None


def test_analyze_pcm_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError, match="Sample rate"):
        analyze_pcm(np.ones(4 * KIWI_SAMPLE_RATE, dtype=np.int16), sample_rate=8_000)
