import numpy as np
from modes.director.events import PresenceStatus
from modes.director.vision.classify import classify_presence, cosine, PresenceDebouncer

REF = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BIG_CENTER = (256, 96, 128, 168)   # ~0.093 area-frac, centered in 640x360


def test_owner_present():
    s = classify_presence(np.array([0.9, 0.1, 0.0]), BIG_CENTER, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.PRESENT


def test_stranger_absent():
    s = classify_presence(np.array([0.0, 1.0, 0.0]), BIG_CENTER, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_no_face_absent():
    s = classify_presence(None, None, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_owner_too_small_absent():
    tiny = (310, 170, 20, 26)      # ~0.0023 area-frac < 0.015
    s = classify_presence(np.array([1.0, 0.0, 0.0]), tiny, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_owner_offcenter_absent():
    edge = (0, 0, 128, 168)        # center x-frac ~0.1 < zone start 0.2
    s = classify_presence(np.array([1.0, 0.0, 0.0]), edge, 640, 360, REF,
                          identity_threshold=0.40, min_area_frac=0.015)
    assert s is PresenceStatus.ABSENT


def test_cosine_basic():
    assert cosine(np.array([1, 0]), np.array([1, 0])) == 1.0
    assert abs(cosine(np.array([1, 0]), np.array([0, 1]))) < 1e-9


def test_cosine_zero_vector():
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == 0.0


def test_debouncer_hysteresis():
    deb = PresenceDebouncer(present_after_s=1.0, absent_after_s=2.0)
    assert deb.update(True, 0.0) == "absent"     # starts absent
    assert deb.update(True, 0.5) == "absent"     # < present_after
    assert deb.update(True, 1.1) == "present"    # >= present_after
    assert deb.update(False, 2.0) == "present"   # < absent_after
    assert deb.update(False, 4.0) == "absent"    # >= absent_after


def test_debouncer_reset():
    deb = PresenceDebouncer(present_after_s=1.0, absent_after_s=2.0)
    # Drive to present
    assert deb.update(True, 0.0) == "absent"
    assert deb.update(True, 1.1) == "present"
    # Reset to absent
    deb.reset()
    # After reset, absent detection returns "absent"
    assert deb.update(False, 1.2) == "absent"
    # A single present update right after reset does not immediately return "present"
    assert deb.update(True, 1.2) == "absent"    # must re-accrue from reset point
    # Re-accruing to present takes present_after_s from reset
    assert deb.update(True, 2.2) == "present"
