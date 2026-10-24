# Viewing to programme matching

Dependency-free Python 3.10+ implementation of decisions D01 and D02 in [the decision record](D:/Projects/hse-recsys/exam/decisions.md). It produces qualifying user/broadcast interactions, not recommendations or MAP scores.

## Run the tests

From `D:\Projects\hse-recsys`:

```powershell
python -m unittest discover -s exam/tests -v
```

The tests use tiny in-memory fixtures, including real CSV parsing, and never read or modify the original dataset. They cover known answers such as 300 seconds passing a long programme's threshold, 299 seconds failing, 120 seconds passing for a four-minute bulletin, and duplicate/overlapping views contributing their union rather than their sum.

## API

```python
import csv
from datetime import datetime
from exam.matching import match_viewings

with open("exam/dataset-sweet-tv/dataset11-30.csv", encoding="utf-8", newline="") as views, \
     open("exam/dataset-sweet-tv/export_arh_11-20-final.csv", encoding="utf-8", newline="") as schedule:
    result = match_viewings(
        csv.DictReader(views),
        csv.DictReader(schedule),
        window_start=datetime(2020, 3, 9),
        window_end=datetime(2020, 5, 18),
    )
```

This is a usage example, not a completed full-dataset run. Supply all schedule rows needed to cover a requested window, combining readers if that window spans files. Supply a fresh reader on each call; an iterator is consumed once. Each call represents one explicit half-open partition. Choosing or orchestrating train/validation windows is outside this module.

Required input fields:

| Input | Fields |
| --- | --- |
| Views | `user_id`, `vsetv_id`, `start_time`, `stop_time`, `duraton` |
| Schedule | `channel_id`, `start_time`, `duration`, `tv_show_id` |

Extra fields are ignored. IDs must be exact decimal strings or Python integers; floating-point IDs are rejected. IDs are returned as canonical decimal strings. Malformed fields raise `ValueError`. Window bounds must be naive whole-second datetimes. No timezone correction is inferred.

## Matching behaviour

- Same channel, half-open intervals, and union overlap `O >= min(300, 0.5 * L)` per broadcast. L is the full scheduled duration even if the caller's window clips the broadcast. Odd durations retain their half-second threshold.
- Exact duplicate views are removed using their five parsed required values. Valid same-user/channel viewing intervals are clipped to the window and unioned before attribution, including repeated visits to one broadcast.
- Duration mismatches, nonpositive view intervals/durations and sessions over 24 hours are returned in `quarantined` and excluded. “Over 24 hours” uses either declared or wall-clock duration, so inconsistent rows cannot evade exclusion. Exactly 24 hours is allowed. The entire original interval is checked before clipping.
- Broadcasts with nonpositive duration are excluded. ID 0 never creates a positive interaction, but remains in conflict detection: simultaneous ID 0 and nonzero assignments are ambiguous.
- Where different programme IDs overlap on a channel, only the ambiguous time span is removed. Unambiguous portions can still qualify. Different channels never conflict.
- Airings with identical channel, programme ID, start and duration are deduplicated regardless of descriptive metadata. Overlapping airings with the same programme ID are not conflicting assignments; each distinct broadcast can contribute one interaction. Repeated airings are not collapsed to one programme-level row.
- The function reads its iterables, does not mutate input mappings and writes no files.

## Outputs and accounting

`result.matches` is a tuple of `BroadcastMatch` records containing exact string IDs, full broadcast start/end, original duration and the qualifying overlap seconds. Ordering is deterministic by numeric user/channel IDs, broadcast start/end and programme ID. Each record is one positive interaction; repeated programme airings remain separate.

`result.quarantined` retains the user/channel, original viewing start/end, declared duration and every applicable exclusion reason. It includes excluded long/nonpositive sessions as well as duration mismatches. No repair is applied. It is returned in memory, not persisted to disk.

`result.audit` distinguishes input-row counts, duplicate excess rows, unique excluded rows, individual reasons and the mismatch/over-24-hour intersection. The following identities hold:

```text
view_rows_read = duplicate_view_rows + unique_view_rows
unique_view_rows = excluded_view_rows + outside_window_view_rows + view_rows_used
positive_overlap_broadcasts = below_threshold_broadcasts + matched_broadcasts
```

Reason counts overlap; do not sum them to obtain total exclusions. Exclusion checks precede window clipping, so these source-quality counts include rows outside the requested window. `view_rows_used` counts valid rows intersecting the window, including those with no matching schedule or qualifying programme. `positive_overlap_broadcasts` and `matched_broadcasts` count user/broadcast pairs. `conflicting_schedule_seconds` counts ambiguous seconds per channel within the window, independently of viewing volume; it is not a count of affected users or views.

## Limits and open decisions

The function accepts iterators but is not a bounded-memory streaming implementation: it holds a schedule index, grouped viewing intervals and output records in memory. Full-dataset memory/runtime have not been benchmarked. The small tests validate matching semantics, not production-scale performance.

Timezone semantics, final refitting, algorithm choices and other open items in the decision record remain open. These automated tests are agent-run evidence and do not establish that the student has personally verified the matching rule.
