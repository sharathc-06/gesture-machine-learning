"""Tests for src/jutsu_sequence.py - JutsuSequenceEngine.

No webcam, no model, no torch required: these feed simulated prediction
strings directly into the engine, exactly as scripts/run_naruto_image_classifier.py
would feed it real classifier output frame by frame. Runnable standalone
(`python tests/test_jutsu_sequence.py`) or via pytest.

NOTE ON THE SELECTION REWORK: JutsuSequenceEngine now requires an explicit
engine.select(jutsu_name) before update() does anything (previously it fed
every tracker every prediction in parallel with no selection concept at
all). Every test below that exercises a single jutsu now calls select()
first; test_shared_gesture_does_not_advance_unselected_jutsu and
test_switching_jutsu_works replace the old
test_three_jutsu_tracked_independently_in_parallel, which specifically
tested the old parallel-everything-advances behavior that selection was
introduced to remove.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.jutsu_sequence import JutsuSequenceEngine, JutsuTracker, JUTSU_SEQUENCES


def feed(engine, predictions, mouth_o_at_end=False, now=None):
    """Feed a list of prediction strings one at a time to whichever jutsu is
    currently selected on `engine`. If mouth_o_at_end, the final update()
    call also carries mouth_o=True (simulating Fireball's non-hand-sign last
    step)."""
    completed_overall = []
    for i, pred in enumerate(predictions):
        is_last = (i == len(predictions) - 1)
        completed = engine.update(pred, mouth_o=(mouth_o_at_end and is_last), now=now)
        completed_overall.extend(completed)
    return completed_overall


def test_rasengan_basic():
    engine = JutsuSequenceEngine()
    engine.select("rasengan")
    completed = feed(engine, ["tiger", "bowl"])
    assert completed == ["rasengan"], completed
    print("test_rasengan_basic: PASSED")


def test_chidori_basic():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    completed = feed(engine, ["ox", "hare", "monkey", "bowl"])
    assert completed == ["chidori"], completed
    print("test_chidori_basic: PASSED")


def test_fireball_basic():
    engine = JutsuSequenceEngine()
    engine.select("fireball")
    completed = feed(
        engine,
        ["horse", "tiger", "snake", "ram", "monkey", "boar", "horse", "tiger"],
        mouth_o_at_end=False,
    )
    assert completed == [], "fireball should not complete before mouth_o"
    completed = engine.update("unknown", mouth_o=True)
    assert completed == ["fireball"], completed
    print("test_fireball_basic: PASSED")


def test_wrong_gestures_dont_reset_chidori():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    completed = feed(engine, ["ox", "tiger", "horse", "hare", "monkey", "bowl"])
    # With chidori selected, "tiger"/"horse" (irrelevant to chidori, and not
    # fed to rasengan/fireball at all since they're not selected) are simply
    # ignored - unlike the old parallel-tracking version of this test,
    # nothing else can complete alongside chidori here.
    assert completed == ["chidori"], completed
    print("test_wrong_gestures_dont_reset_chidori: PASSED")


def test_single_frame_recognition_is_enough():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    completed = feed(engine, ["ox", "hare", "monkey", "bowl"])
    assert completed == ["chidori"], completed
    print("test_single_frame_recognition_is_enough: PASSED")


def test_repeated_previous_gesture_ignored():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    tracker = engine.trackers["chidori"]
    engine.update("ox")
    engine.update("hare")
    assert tracker.get_progress().current_step == "monkey", \
        f"expected to be waiting for monkey after ox,hare - got {tracker.get_progress().current_step}"
    engine.update("ox")     # repeat of step 0 - must NOT re-accept or move anything
    assert tracker.get_progress().current_step == "monkey"
    engine.update("tiger")  # irrelevant - ignored
    assert tracker.get_progress().current_step == "monkey"
    completed = engine.update("monkey")
    assert completed == [], "not complete yet, still need bowl"
    assert tracker.get_progress().current_step == "bowl"
    completed = engine.update("bowl")
    assert completed == ["chidori"]
    print("test_repeated_previous_gesture_ignored: PASSED")


def test_resets_after_completion():
    engine = JutsuSequenceEngine()
    engine.select("rasengan")
    feed(engine, ["tiger", "bowl"])
    progress = engine.get_progress("rasengan")
    assert progress.accepted == [False, False], "should be back to a clean slate after completing"
    print("test_resets_after_completion: PASSED")


def test_cooldown_blocks_immediate_retrigger():
    engine = JutsuSequenceEngine(cooldown_seconds=3.0)
    engine.select("rasengan")
    t0 = 1000.0
    completed = feed(engine, ["tiger", "bowl"], now=t0)
    assert completed == ["rasengan"]
    completed = feed(engine, ["tiger", "bowl"], now=t0 + 0.01)
    assert completed == [], f"should be blocked by cooldown, got {completed}"
    completed = feed(engine, ["tiger", "bowl"], now=t0 + 3.5)
    assert completed == ["rasengan"], f"should retrigger after cooldown elapses, got {completed}"
    print("test_cooldown_blocks_immediate_retrigger: PASSED")


def test_bowl_does_not_need_to_be_simultaneous():
    engine = JutsuSequenceEngine()
    engine.select("rasengan")
    completed = feed(engine, ["tiger", "unknown", "ok_irrelevant", "unknown", "bowl"])
    assert completed == ["rasengan"], completed
    print("test_bowl_does_not_need_to_be_simultaneous: PASSED")


def test_no_stable_frame_requirement():
    tracker = JutsuTracker("rasengan", JUTSU_SEQUENCES["rasengan"])
    assert tracker.get_progress().current_step == "tiger"
    tracker.update("tiger", mouth_o=False)
    assert tracker.get_progress().current_step == "bowl", "one frame should be enough to accept a step"
    print("test_no_stable_frame_requirement: PASSED")


def test_get_current_step_api():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    assert engine.get_current_step("chidori") == "ox"
    engine.update("ox")
    assert engine.get_current_step("chidori") == "hare"
    print("test_get_current_step_api: PASSED")


def test_manual_reset():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    engine.update("ox")
    engine.update("hare")
    assert engine.get_current_step("chidori") == "monkey"
    engine.reset("chidori")
    assert engine.get_current_step("chidori") == "ox", "manual reset should return to step 0"
    print("test_manual_reset: PASSED")


# ---- New tests for the selection rework (requirements 1-6) ----

def test_select_rasengan_activates_only_rasengan():
    engine = JutsuSequenceEngine()
    engine.select("rasengan")
    assert engine.active_jutsu == "rasengan"
    engine.update("tiger")
    assert engine.get_progress("rasengan").accepted[0] is True, "rasengan should have advanced"
    assert engine.get_progress("chidori").accepted == [False, False, False, False], "chidori must be untouched"
    assert engine.get_progress("fireball").accepted == [False] * 9, "fireball must be untouched"
    print("test_select_rasengan_activates_only_rasengan: PASSED")


def test_select_chidori_activates_only_chidori():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    assert engine.active_jutsu == "chidori"
    engine.update("ox")
    assert engine.get_progress("chidori").accepted[0] is True
    assert engine.get_progress("rasengan").accepted == [False, False]
    assert engine.get_progress("fireball").accepted == [False] * 9
    print("test_select_chidori_activates_only_chidori: PASSED")


def test_select_fireball_activates_only_fireball():
    engine = JutsuSequenceEngine()
    engine.select("fireball")
    assert engine.active_jutsu == "fireball"
    engine.update("horse")
    assert engine.get_progress("fireball").accepted[0] is True
    assert engine.get_progress("rasengan").accepted == [False, False]
    assert engine.get_progress("chidori").accepted == [False, False, False, False]
    print("test_select_fireball_activates_only_fireball: PASSED")


def test_shared_gesture_does_not_advance_unselected_jutsu():
    # Chidori and Fireball both include "monkey". With Chidori selected,
    # feeding "monkey" (chidori's 3rd step, after ox+hare) must NOT advance
    # Fireball's tracker at all, even though "monkey" is also one of
    # Fireball's steps - this is the core bug this whole phase fixes.
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    engine.update("ox")
    engine.update("hare")
    engine.update("monkey")  # chidori's 3rd step - also appears in fireball's sequence
    fireball_progress = engine.get_progress("fireball")
    assert fireball_progress.accepted == [False] * 9, \
        f"fireball must not advance from a prediction while chidori is selected, got {fireball_progress.accepted}"
    assert engine.get_progress("chidori").accepted == [True, True, True, False]
    print("test_shared_gesture_does_not_advance_unselected_jutsu: PASSED")


def test_switching_jutsu_works():
    engine = JutsuSequenceEngine()
    engine.select("chidori")
    engine.update("ox")
    engine.update("hare")
    assert engine.get_progress("chidori").accepted == [True, True, False, False]

    engine.select("rasengan")
    assert engine.active_jutsu == "rasengan"
    engine.update("tiger")
    assert engine.get_progress("rasengan").accepted == [True, False]
    # chidori's progress is untouched while it's not selected...
    assert engine.get_progress("chidori").accepted == [True, True, False, False]
    completed = engine.update("bowl")
    assert completed == ["rasengan"]

    # ...and switching back to chidori resumes exactly where it left off
    # (selecting does not reset progress - see JutsuSequenceEngine docstring).
    engine.select("chidori")
    assert engine.get_progress("chidori").accepted == [True, True, False, False]
    completed = engine.update("monkey")
    assert completed == []
    completed = engine.update("bowl")
    assert completed == ["chidori"]
    print("test_switching_jutsu_works: PASSED")


def test_selected_jutsu_completes_normally():
    engine = JutsuSequenceEngine()
    engine.select("fireball")
    completed = feed(
        engine, ["horse", "tiger", "snake", "ram", "monkey", "boar", "horse", "tiger"]
    )
    assert completed == []
    completed = engine.update("unknown", mouth_o=True)
    assert completed == ["fireball"], completed
    print("test_selected_jutsu_completes_normally: PASSED")


def test_update_before_any_selection_is_a_noop():
    engine = JutsuSequenceEngine()
    assert engine.active_jutsu is None
    completed = engine.update("tiger")
    assert completed == [], "update() before any select() must do nothing"
    assert engine.get_progress("rasengan").accepted == [False, False], \
        "no tracker should advance before a jutsu is selected"
    print("test_update_before_any_selection_is_a_noop: PASSED")


def test_deselect_stops_progress():
    engine = JutsuSequenceEngine()
    engine.select("rasengan")
    engine.update("tiger")
    assert engine.get_progress("rasengan").accepted[0] is True
    engine.deselect()
    assert engine.active_jutsu is None
    completed = engine.update("bowl")
    assert completed == [], "deselected engine must not progress or complete anything"
    assert engine.get_progress("rasengan").accepted == [True, False], "progress frozen, not lost, on deselect"
    print("test_deselect_stops_progress: PASSED")


def test_select_unknown_jutsu_raises():
    engine = JutsuSequenceEngine()
    try:
        engine.select("not_a_real_jutsu")
        raised = False
    except KeyError:
        raised = True
    assert raised, "selecting an unknown jutsu name should raise KeyError, not silently no-op"
    print("test_select_unknown_jutsu_raises: PASSED")


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
    test_no_stable_frame_requirement,
    test_get_current_step_api,
    test_manual_reset,
    test_select_rasengan_activates_only_rasengan,
    test_select_chidori_activates_only_chidori,
    test_select_fireball_activates_only_fireball,
    test_shared_gesture_does_not_advance_unselected_jutsu,
    test_switching_jutsu_works,
    test_selected_jutsu_completes_normally,
    test_update_before_any_selection_is_a_noop,
    test_deselect_stops_progress,
    test_select_unknown_jutsu_raises,
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
