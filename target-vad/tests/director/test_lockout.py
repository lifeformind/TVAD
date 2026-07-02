from modes.director.lockout import Lockout, LockoutAction
from modes.director.safety_net import SafetyVerdict


def _fail():
    return SafetyVerdict(score=0.0, smoother_ok=False, window_rms=0.1)


def _pass():
    return SafetyVerdict(score=0.9, smoother_ok=True, window_rms=0.1)


def test_first_miss_is_warn_not_eject():
    lk = Lockout()
    assert lk.on_verdict(_fail(), rms_ok=True) is LockoutAction.WARN


def test_two_misses_but_rms_ok_does_not_eject():
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=True)
    assert lk.on_verdict(_fail(), rms_ok=True) is LockoutAction.WARN


def test_two_misses_and_failed_proximity_ejects():
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=False)
    assert lk.on_verdict(_fail(), rms_ok=False) is LockoutAction.EJECT


def test_a_pass_resets_the_miss_streak():
    lk = Lockout()
    lk.on_verdict(_fail(), rms_ok=False)
    assert lk.on_verdict(_pass(), rms_ok=True) is LockoutAction.NONE
    assert lk.on_verdict(_fail(), rms_ok=False) is LockoutAction.WARN


def test_idle_after_quiet_window_post_eject():
    lk = Lockout(idle_after_s=5.0)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.on_verdict(_fail(), rms_ok=False)   # EJECT
    lk.note_ejected_at(now=100.0)
    # near-field active -> not idle, AND it resets the quiet clock to 104.9
    assert lk.on_idle_tick(now=104.9, near_field_rms_active=True) is None
    # only 0.1s of silence since that reset -> not yet
    assert lk.on_idle_tick(now=105.0, near_field_rms_active=False) is None
    # >= idle_after_s of continuous silence from the reset -> IDLE
    assert lk.on_idle_tick(now=110.0, near_field_rms_active=False) is LockoutAction.IDLE


def test_no_permanent_lockout_idle_rearms_on_activity():
    lk = Lockout(idle_after_s=5.0)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.on_verdict(_fail(), rms_ok=False)
    lk.note_ejected_at(now=100.0)
    # Activity at 103 resets the quiet clock; without that reset, 107-100=7s
    # would already be IDLE. With it, only 4s of silence has elapsed -> not yet.
    assert lk.on_idle_tick(now=103.0, near_field_rms_active=True) is None
    assert lk.on_idle_tick(now=107.0, near_field_rms_active=False) is None
    # ...and once a continuous idle_after_s of silence passes from the reset -> IDLE.
    assert lk.on_idle_tick(now=108.5, near_field_rms_active=False) is LockoutAction.IDLE
