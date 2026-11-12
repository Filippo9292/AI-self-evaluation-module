# Early-stopping state machine

The policy receives only completed checkpoint observations. It uses the normalized quartile distance defined in [protocol.md](protocol.md).

## Best and progress

The first valid observation initializes both references. Every strictly smaller distance updates the absolute best and causes a checkpoint copy. Equal distances retain the earlier checkpoint. A separate progress reference is updated only when the distance improves by at least `min_delta` relative to that reference. Small improvements can therefore accumulate without losing the absolute best.

Before `min_epochs` no stop is emitted. Plateau accounting starts after that minimum. Each later valid observation without sufficient improvement adds at most `every_n_epochs` to the stalled counter. A missing interval is not treated as evidence of a plateau. For an ordinary schedule every five epochs, thirty patience epochs correspond to six valid intervals. Late successful retries count one interval, not all unobserved elapsed time.

## Target approach

An observation at or below `target_tolerance` starts the additional observation window after the minimum epoch. Further successful observations add valid observed epoch intervals. Consecutive in-tolerance observations increment target confirmation count. An observation above the target but within `target_reset_tolerance` retains the window and resets consecutive confirmations. Marked deterioration above the reset tolerance clears the window.

When the additional window reaches `post_target_epochs` and confirmations reach `target_confirmations`, the policy stops with `target_reached_and_confirmed`. While this window is active, plateau stopping is suspended. If a full window expires without sufficient confirmation, it is cleared and normal plateau logic resumes. The policy selects the closest checkpoint over all completed observations, not only those inside the window.

## Plateau

Without an active target window, reaching `plateau_patience_epochs` stops with `plateau`. If the ordinary training budget ends before either rule fires, the trainer should still retrieve `monitor.best()` and restore its retained checkpoint.

## Failure and recovery

Provider errors, malformed distributions, request limits and incomplete panels interrupt the observation. They do not assign a low score and do not advance the policy. The default Python API raises `EvaluationError`; the integrating trainer decides whether to suspend training or continue until a later monitored checkpoint. Do not catch the exception and invent a score.

Each epoch records an immutable checkpoint/image manifest before evaluation. Raw answers are journaled before validation. A new best is copied and hash-checked before the atomic `observation.json` commit. This file is the authoritative policy event. State/history/report files are derived conveniences; restarting reconstructs the previous decision from committed observations. Duplicate observation of the same epoch and same inputs returns the committed decision without new API calls.

Epochs must increase; changing checkpoint bytes/images under a planned epoch, changing the generation panel, inserting old epochs or changing the policy mid-run is rejected. A stopped run must use a new output directory to start again. The service may change hidden weights without changing its model identifier; local fingerprinting cannot detect that.
