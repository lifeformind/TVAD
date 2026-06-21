import numpy as np
from modes.talkback.endpointing import NullTurnDetector


def test_null_detector_reports_complete():
    det = NullTurnDetector()
    # contract: returns a float prob in [0,1]; Null always "complete"
    p = det.endpoint_prob(np.zeros(8000, dtype=np.float32), sample_rate=16000)
    assert 0.0 <= p <= 1.0
    assert p == 1.0
