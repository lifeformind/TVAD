from modes.director import events as E
from modes.director.events import PresenceStatus
from modes.director.config import DirectorConfig
from modes.director.context import new_context
from modes.talkback.conversation import ConversationManager


def test_owner_presence_event_carries_status_and_time():
    ev = E.OwnerPresenceEvent(status=PresenceStatus.PRESENT, now=12.5)
    assert ev.status is PresenceStatus.PRESENT
    assert ev.now == 12.5


def test_presence_status_members():
    assert {s.name for s in PresenceStatus} == {"PRESENT", "ABSENT", "UNAVAILABLE"}


def test_config_floor_control_defaults():
    cfg = DirectorConfig()
    assert cfg.owner_absent_grace_s == 3.0
    assert cfg.active_talk_guard_s == 3.0


def test_context_starts_unavailable_at_session_now():
    ctx = new_context(DirectorConfig(), ConversationManager(system_prompt="x"),
                      now=7.0, proximity_rms=0.0)
    assert ctx.presence_status is PresenceStatus.UNAVAILABLE
    assert ctx.presence_since == 7.0
