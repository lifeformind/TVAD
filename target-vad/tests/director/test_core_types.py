from modes.director.state import State
from modes.director.config import DirectorConfig
from modes.director.context import Context, new_context
from modes.talkback.conversation import ConversationManager


def test_state_has_five_members():
    assert {s.name for s in State} == {
        "IDLE", "LISTENING", "THINKING", "SPEAKING", "EVALUATING"}


def test_config_defaults_match_spec():
    c = DirectorConfig()
    assert c.silence_timeout_s == 30.0
    assert c.hard_timeout_s == 300.0
    assert c.nudge_lead_s == 5.0
    assert 0.0 < c.nudge_lead_s < c.silence_timeout_s
    assert c.endpoint_threshold == 0.5
    assert c.verify_window_ms == 700.0
    assert c.min_speech_ms == 120.0
    assert c.speaker_threshold == 0.20
    assert c.conf_floor == 0.5
    assert c.duck_level == 0.35


def test_new_context_starts_a_session_clock():
    cfg = DirectorConfig()
    conv = ConversationManager(system_prompt="sys")
    ctx = new_context(cfg, conv, now=100.0, proximity_rms=0.02)
    assert ctx.now == 100.0
    assert ctx.started_at == 100.0
    assert ctx.last_speech_at == 100.0
    assert ctx.gen_id == 0
    assert ctx.nudged_cycle is False
    assert ctx.ducked is False
    assert ctx.interrupted_stack == []
    assert ctx.pending_steer is None
    assert ctx.proximity_rms == 0.02
    assert ctx.conversation is conv
    assert ctx.cfg is cfg
