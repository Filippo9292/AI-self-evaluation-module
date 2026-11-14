"""Causal hysteresis and elapsed-epoch confirmation, independent of model/domain."""
import numpy as np

def replay(events, policy):
    active=False;since=None;history=[];trace=[];stop=None;last=None
    cadence=policy['audit_every_epochs']
    for event in events:
        e=event['epoch']
        if last is not None and e<=last:raise ValueError('Epochs must increase.')
        if last is not None and e-last!=cadence:active=False;since=None;history=[]
        history.append(event)
        window=[r for r in history if e-policy['smoothing_horizon_epochs']<=r['epoch']<=e]
        full=len(window)==policy['smoothing_horizon_epochs']//cadence+1 and window[0]['epoch']==e-policy['smoothing_horizon_epochs']
        q=float(np.median([r['quality'] for r in window]));d=float(np.median([r['defect'] for r in window]))
        pp=[r['progress'] for r in window];p=None if any(v is None for v in pp) else float(np.median(pp))
        if not full:active=False
        elif active:active=q>=policy['quality_exit_ratio'] and d>=policy['defect_exit_normalized']
        else:active=q>=policy['quality_enter_ratio'] and d>=policy['defect_enter_normalized']
        no_gain=p is not None and p<=policy['progress_max'];eligible=bool(full and active and no_gain)
        if not eligible:since=None
        elif since is None:since=e
        elapsed=0 if since is None else e-since
        if stop is None and since is not None and elapsed>=policy['confirmation_epochs']:stop=e
        trace.append(dict(epoch=e,quality_smooth=q,defect_smooth=d,progress_smooth=p,mature=bool(active),no_progress=bool(no_gain),eligible=eligible,confirmation_start=since,elapsed=elapsed,first_stop=stop==e,should_stop=stop is not None,first_stop_epoch=stop))
        last=e
    return trace
