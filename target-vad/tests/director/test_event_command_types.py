import pytest

from modes.director import events as E
from modes.director import commands as C


def test_events_construct_and_are_frozen():
    seg = E.SegmentEndpointed(duration_ms=900.0, rms=0.4, is_target=True,
                              endpoint_prob=0.8)
    assert seg.is_target is True
    with pytest.raises(Exception):
        seg.is_target = False  # frozen


def test_commands_construct_and_are_frozen():
    cmd = C.StartGeneration(gen_id=3, messages=[{"role": "user", "content": "hi"}],
                            steer=None)
    assert cmd.gen_id == 3
    with pytest.raises(Exception):
        cmd.gen_id = 4  # frozen


def test_tick_and_reply_carry_their_payloads():
    assert E.Tick(now=12.5).now == 12.5
    rc = E.ReplyComplete(gen_id=2, assistant_text="done")
    assert rc.gen_id == 2 and rc.assistant_text == "done"
