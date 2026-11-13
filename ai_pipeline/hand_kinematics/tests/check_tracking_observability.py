"""Run with:  python tests/check_tracking_observability.py"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with contextlib.redirect_stdout(io.StringIO()):
    import check_tracker_logic as L

import hand_tracker as ht

ok = True


def check(name, condition, extra=""):
    global ok
    ok &= bool(condition)
    print(("PASS " if condition else "FAIL ") + name + (f"  [{extra}]" if extra else ""))


stats = ht.TrackerStats()
stats.record(ht.STATUS_DETECTED, True, False, 1.0, state=0)
stats.record(ht.STATUS_DETECTED, True, False, 1.0, state=0)
stats.record(ht.STATUS_DETECTED, True, False, 1.0, state=1)
stats.record(ht.STATUS_HELD, False, False, 1.0, state=1)
stats.record(ht.STATUS_RECOVERED, False, False, 1.0, state=1)
stats.record(ht.STATUS_LOST, False, False, 1.0, state=1)
stats.record(ht.STATUS_LOST, False, False, 1.0, state=1)
stats.record(ht.STATUS_DETECTED, True, False, 1.0, state=0)
summary = stats.summary()

check("state counts are tracked", summary["state_counts"] == {0: 3, 1: 5, 2: 0}, str(summary["state_counts"]))
check("gesture-state changes are counted", summary["state_changes"] == 2, str(summary["state_changes"]))
check("longest stable gesture run is tracked", summary["longest_stable_frames"] == 5, str(summary["longest_stable_frames"]))
check("status changes are counted", summary["status_changes"] == 4, str(summary["status_changes"]))
check("grace-period expiry is counted once", summary["lost_events"] == 1, str(summary["lost_events"]))
check("ROI recovery is counted", summary["recovery_events"] == 1, str(summary["recovery_events"]))
check("reacquisition after loss is counted", summary["reacquired_events"] == 2, str(summary["reacquired_events"]))

script = [L.hand(0.5, 0.5)] * 3 + [L.EMPTY] * 2
tracker, _, _ = L.make(script, state=2, enable_roi_recovery=False)
L.run(tracker, 5)
tracker_summary = tracker.stats.summary()
check("process_frame records gesture state", tracker_summary["state_counts"][2] == 5, str(tracker_summary["state_counts"]))
check("process_frame records a status transition", tracker_summary["status_changes"] == 1, str(tracker_summary["status_changes"]))

startup, _, _ = L.make([L.EMPTY], state=0, enable_roi_recovery=False)
startup_out = startup.process_frame(L.FRAME)
check("startup no-hand status is LOST and observable",
      startup_out == (-1.0, -1.0, 0) and startup.status == ht.STATUS_LOST
      and not startup.tracking_valid and startup.missed_frames == 1
      and startup.get_status() == (ht.STATUS_LOST, False))

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
