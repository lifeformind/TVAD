import numpy as np
import pytest
from modes.talkback.endpointing import NullTurnDetector


def test_null_detector_reports_complete():
    det = NullTurnDetector()
    # contract: returns a float prob in [0,1]; Null always "complete"
    p = det.endpoint_prob(np.zeros(8000, dtype=np.float32), sample_rate=16000)
    assert 0.0 <= p <= 1.0
    assert p == 1.0


def test_smart_turn_smoke():
    try:
        from modes.talkback.endpointing import SmartTurnDetector
        det = SmartTurnDetector()
    except Exception:
        pytest.skip("pipecat/onnxruntime/model unavailable")
    p = det.endpoint_prob(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)
    assert 0.0 <= p <= 1.0
