"""Pure, serializable state machine; a missing observation never enters this function."""
import copy

from .errors import EvaluationError


def advance(previous, epoch, distance, settings):
    state = copy.deepcopy(previous) if previous else {
        "last_epoch": None, "best_epoch": None, "best_distance": None,
        "progress_reference": None, "stalled_epochs": 0,
        "target_start_epoch": None, "target_valid_epochs": 0,
        "target_confirmations": 0, "should_stop": False, "reason": "continue"}
    if state["should_stop"]:
        raise EvaluationError("This run has already stopped; use a new output directory for a new run.")
    last = state["last_epoch"]
    if last is not None and epoch <= last:
        raise EvaluationError("Observation epochs must increase strictly.")
    # Skipped/failed evaluations do not count as multiple observations of no progress.
    elapsed = 0 if last is None else min(epoch - last, settings["every_n_epochs"])
    improved_best = state["best_distance"] is None or distance < state["best_distance"]
    if improved_best:
        state.update(best_epoch=epoch, best_distance=distance)
    reference = state["progress_reference"]
    if reference is None or reference - distance >= settings["min_delta"] - 1e-12:
        state.update(progress_reference=distance, stalled_epochs=0)
    elif epoch > settings["min_epochs"]:
        state["stalled_epochs"] += min(elapsed, epoch - settings["min_epochs"])
    if epoch >= settings["min_epochs"]:
        if distance > settings["target_reset_tolerance"]:
            state.update(target_start_epoch=None, target_valid_epochs=0, target_confirmations=0)
        elif distance <= settings["target_tolerance"]:
            if state["target_start_epoch"] is None:
                state["target_start_epoch"] = epoch
            else:
                state["target_valid_epochs"] += elapsed
            state["target_confirmations"] += 1
        else:
            # Hysteresis keeps the observation window, but requires renewed confirmation.
            if state["target_start_epoch"] is not None:
                state["target_valid_epochs"] += elapsed
            state["target_confirmations"] = 0
        if state["target_start_epoch"] is not None and state["target_valid_epochs"] >= settings["post_target_epochs"] and state["target_confirmations"] >= settings["target_confirmations"]:
            state.update(should_stop=True, reason="target_reached_and_confirmed")
        else:
            if state["target_start_epoch"] is not None and state["target_valid_epochs"] >= settings["post_target_epochs"]:
                # An unconfirmed target window expires; ordinary plateau logic resumes.
                state.update(target_start_epoch=None, target_valid_epochs=0, target_confirmations=0)
            if state["target_start_epoch"] is None and state["stalled_epochs"] >= settings["plateau_patience_epochs"]:
                state.update(should_stop=True, reason="plateau")
    state["last_epoch"] = epoch
    state["improved_best"] = improved_best
    return state
