# tests/director/test_handoff_contract.py
"""DirectorHandoff/DirectorResult are the renamed handoff contract (binding
interface, owned by Plan 02; asserted here so Plan 03 is self-contained)."""

import numpy as np


def test_director_handoff_has_holdout_embedding_field():
    from modes.talkback.handoff import DirectorHandoff
    emb = np.ones(192, dtype=np.float32)
    h = DirectorHandoff(
        mic="mic", primary_embedding=emb, holdout_embedding=emb,
        first_segment="seg", config={}, vad="vad", embedder="emb",
    )
    assert h.holdout_embedding is emb
    assert h.primary_embedding is emb
    assert h.mic == "mic"


def test_director_result_carries_reason_turns_duration():
    from modes.talkback.handoff import DirectorResult
    r = DirectorResult(reason="silence_timeout", turns=3, total_duration_s=12.5)
    assert r.reason == "silence_timeout"
    assert r.turns == 3
    assert r.total_duration_s == 12.5
