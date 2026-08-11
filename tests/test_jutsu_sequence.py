"""Tests for src/jutsu_sequence.py - JutsuSequenceEngine.

No webcam, no model, no torch required: these feed simulated prediction
strings directly into the engine, exactly as scripts/run_naruto_image_classifier.py
would feed it real classifier output frame by frame. Runnable standalone
(`python tests/test_jutsu_sequence.py`) or via pytest.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.jutsu_sequence import JutsuSequenceEngine, JutsuTracker, JUTSU_SEQUENCES


def feed(engine, predictions, mouth_o_at_end=False, now=None):
    """Feed a list of prediction strings one at a time. If mouth_o_at_end,
    the final update() call also carries mouth_o=True (simulating the
    Fireball jutsu's non-hand-sign last step)."""
    completed_overall = []
    for i, pred in enumerate(predictions):
        is_last = (i == len(predictions) - 1)
        completed = engine.update(pred, mouth_o=(mouth_o_at_end and is_last), now=now)
        completed_overall.extend(completed)
    return completed_overall


def test_rasengan_basic():
    engine = JutsuSequenceEngine()
    completed = feed(engine, ["tiger", "bowl"])
    assert completed == ["rasengan"], completed
    print("test_rasengan_basic: PASSED")


def test_chidori_basic():
    engine = JutsuSequenceEngine()
    completed = feed(engine, ["ox", "hare", "monkey", "bowl"])
    assert completed == ["chidori"], completed
    print("test_chidori_basic: PASSED")


def test_fireball_basic():
    engine = JutsuSequenceEngine()
    completed = feed(
        engine,
        ["horse", "tiger", "snake", "ram", "monkey", "boar", "horse", "tiger"],
        mouth_o_at_end=False,
    )
    assert completed == [], "fireball should not complete before mouth_o"
    # mouth_o is a separate signal, not a hand prediction - feed it alongside
    # an irrelevant/unknown hand prediction, as the real loop would.
    completed = engine.update("unknown", mouth_o=True)
    assert completed == ["fireball"], completed
    print("test_fireball_basic: PASSED")


def test_wrong_gestures_dont_reset_chidori():
    engine = JutsuSequenceEngine()
    completed = feed(engine, ["ox", "tiger", "horse", "hare", "monkey", "bowl"])
    # Chidori must complete. Note "tiger" then later "bowl" also happen to
    # satisfy Rasengan's own independent sequence (tiger -> bowl) - that's
    # correct under the parallel-independent-tracking design (see
    # test_three_jutsu_tracked_independently_in_parallel), not a bug, so
    # this assertion checks Chidori's own completion rather than asserting
    # completed == ["chidori"] and accidentally depending on Rasengan NOT
    # also completing.
    assert "chidori" in completed, completed
    print("test_wrong_gestures_dont_reset_chidori: PASSED")


def test_single_frame_recognition_is_enough():
    engine = JutsuSequenceEngine()
    # each prediction occurs exactly once, no repeats/holding needed
    completed = feed(engine, ["ox", "hare", "monkey", "bowl"])
    assert completed == ["chidori"], completed
    print("test_single_frame_recognition_is_enough: PASSED")


def test_repeated_previous_gesture_ignored():
    engine = JutsuSequenceEngine()
    tracker = engine.trackers["chidori"]
    # ox -> hare -> ox (repeat of an already-accepted step) -> tiger (irrelevant) -> monkey -> bowl
    engine.update("ox")
    engine.update("hare")
    assert tracker.get_progress().current_step == "monkey", \
        f"expected to be waiting for monkey after ox,hare - got {tracker.get_progress().current_step}"
    engine.update("ox")     # repeat of step 0 - must NOT re-accept or move anything
    assert tracker.get_progress().current_step == "monkey", \
        "repeating an already-accepted gesture must not affect progress"
    engine.update("tiger")  # irrelevant to chidori - ignored (this also happens to be
                             # rasengan's own step 0, which is fine/expected and unrelated
                             # to what this test checks)
    assert tracker.get_progress().current_step == "monkey"
    completed = engine.update("monkey")
    assert "chidori" not in completed, "not complete yet, still need bowl"
    assert tracker.get_progress().current_step == "bowl"
    completed = engine.update("bowl")
    assert "chidori" in completed, completed
    print("test_repeated_previous_gesture_ignored: PASSED")


def test_resets_after_completion():
    engine = JutsuSequenceEngine()
    feed(engine, ["tiger", "bowl"])
    progress = engine.get_progress("rasengan")
    assert progress.accepted == [False, False], "should be back to a clean slate after completing"
    assert all(not a for a in progress.accepted)
    print("test_resets_after_completion: PASSED")


def test_cooldown_blocks_immediate_retrigger():
    engine = JutsuSequenceEngine(cooldown_seconds=3.0)
    t0 = 1000.0
    completed = feed(engine, ["tiger", "bowl"], now=t0)
    assert completed == ["rasengan"]
    # Try again immediately (same timestamp) - must NOT retrigger during cooldown
    completed = feed(engine, ["tiger", "bowl"], now=t0 + 0.01)
    assert completed == [], f"should be blocked by cooldown, got {completed}"
    # After the cooldown window has passed, it should work again
    completed = feed(engine, ["tiger", "bowl"], now=t0 + 3.5)
    assert completed == ["rasengan"], f"should retrigger after cooldown elapses, got {completed}"
    print("test_cooldown_blocks_immediate_retrigger: PASSED")


def test_bowl_does_not_need_to_be_simultaneous():
    # "tiger -> [user moves hand, other predictions happen] -> bowl" must still work,
    # since irrelevant intermediate predictions are simply ignored.
    engine = JutsuSequenceEngine()
    completed = feed(engine, ["tiger", "unknown", "ok_irrelevant", "unknown", "bowl"])
    assert completed == ["rasengan"], completed
    print("test_bowl_does_not_need_to_be_simultaneous: PASSED")


def test_three_jutsu_tracked_independently_in_parallel():
    # "tiger" is step 1 of rasengan AND step 2 of fireball (after "horse").
    # Performing horse, then tiger should advance fireball's tracker to step 2 accepted,
    # while ALSO advancing rasengan's tracker to step 1 accepted (both see "tiger" match
    # their own current expected step) - this is intentional per the independent-tracking design.
    engine = JutsuSequenceEngine()
    engine.update("horse")
    fireball_progress = engine.get_progress("fireball")
    rasengan_progress = engine.get_progress("rasengan")
    assert fireball_progress.current_step == "tiger", fireball_progress
    assert rasengan_progress.current_step == "tiger", "rasengan step 0 is tiger, unaffected by horse"

    completed = engine.update("tiger")
    assert completed == [], "neither jutsu complete yet"
    fireball_progress = engine.get_progress("fireball")
    rasengan_progress = engine.get_progress("rasengan")
    assert fireball_progress.accepted[:2] == [True, True], fireball_progress
    assert rasengan_progress.accepted[0] is True, "rasengan's tiger step should ALSO have advanced"
    assert rasengan_progress.current_step == "bowl"

    completed = engine.update("bowl")
    assert completed == ["rasengan"], completed
    # fireball should be unaffected by rasengan completing/resetting
    fireball_progress = engine.get_progress("fireball")
    assert fireball_progress.accepted[:2] == [True, True], "fireball's progress must be untouched by rasengan resetting"
    print("test_three_jutsu_tracked_independently_in_parallel: PASSED")


def test_no_stable_frame_requirement():
    # A brand-new engine; a single correct prediction must accept the step
    # immediately, with no repetition/holding needed (this is really the same
    # guarantee as test_single_frame_recognition_is_enough, checked at the
    # tracker level instead of via full-sequence completion).
    tracker = JutsuTracker("rasengan", JUTSU_SEQUENCES["rasengan"])
    assert tracker.get_progress().current_step == "tiger"
    tracker.update("tiger", mouth_o=False)
    assert tracker.get_progress().current_step == "bowl", "one frame should be enough to accept a step"
    print("test_no_stable_frame_requirement: PASSED")


def test_get_current_step_api():
    engine = JutsuSequenceEngine()
    assert engine.get_current_step("chidori") == "ox"
    engine.update("ox")
    assert engine.get_current_step("chidori") == "hare"
    print("test_get_current_step_api: PASSED")


def test_manual_reset():
    engine = JutsuSequenceEngine()
    engine.update("ox")
    engine.update("hare")
    assert engine.get_current_step("chidori") == "monkey"
    engine.reset("chidori")
    assert engine.get_current_step("chidori") == "ox", "manual reset should return to step 0"
    print("test_manual_reset: PASSED")


ALL_TESTS = [
    test_rasengan_basic,
    test_chidori_basic,
    test_fireball_basic,
    test_wrong_gestures_dont_reset_chidori,
    test_single_frame_recognition_is_enough,
    test_repeated_previous_gesture_ignored,
    test_resets_after_completion,
    test_cooldown_blocks_immediate_retrigger,
    test_bowl_does_not_need_to_be_simultaneous,
    test_three_jutsu_tracked_independently_in_parallel,
    test_no_stable_frame_requirement,
    test_get_current_step_api,
    test_manual_reset,
]


if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"{test.__name__}: FAILED - {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} tests passed")
    if failures:
        sys.exit(1)
