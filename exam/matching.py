"""Read-only viewing-to-broadcast matching for decisions D01/D02.

Accepts csv.DictReader iterables. Dates are naive source wall-clock times;
no timezone, train/validation orchestration or recommendation model is inferred.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
import re
from typing import Iterable, Mapping


Row = Mapping[str, str | int]
SECOND = timedelta(seconds=1)


@dataclass(frozen=True)
class BroadcastMatch:
    user_id: str
    channel_id: str
    tv_show_id: str
    broadcast_start: datetime
    broadcast_end: datetime
    duration_seconds: int
    overlap_seconds: int


@dataclass(frozen=True)
class QuarantinedViewing:
    user_id: str
    channel_id: str
    start: datetime
    end: datetime
    declared_duration_seconds: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[BroadcastMatch, ...]
    quarantined: tuple[QuarantinedViewing, ...]
    audit: dict[str, int]


@dataclass(frozen=True)
class _Broadcast:
    channel: str
    item: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class _Segment:
    start: datetime
    end: datetime
    broadcasts: tuple[_Broadcast, ...]


def _field(row: Row, name: str):
    try:
        return row[name]
    except KeyError:
        raise ValueError(f'Missing required column: {name}') from None


def _integer(row: Row, name: str, minimum: int | None = None) -> int:
    value = _field(row, name)
    if (isinstance(value, bool) or not isinstance(value, (str, int))
            or re.fullmatch(r'-?[0-9]+', str(value)) is None):
        raise ValueError(f'{name} must be an exact integer, not {value!r}')
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return number


def _date(row: Row, name: str, fmt: str) -> datetime:
    value = _field(row, name)
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a timestamp string')
    try:
        return datetime.strptime(value, fmt)
    except ValueError:
        raise ValueError(f'Invalid {name}: {value!r}; expected {fmt}') from None


def _schedule_index(rows: Iterable[Row], start: datetime, end: datetime,
                    audit: Counter) -> dict[str, tuple[list[_Segment], list[datetime]]]:
    """Sweep each channel into disjoint unambiguous segments, retaining airing identity."""
    events = defaultdict(list)
    seen = set()
    for row in rows:
        audit['schedule_rows_read'] += 1
        channel = str(_integer(row, 'channel_id', 1))
        item = str(_integer(row, 'tv_show_id', 0))
        begin = _date(row, 'start_time', '%d.%m.%Y %H:%M:%S')
        duration = _integer(row, 'duration')
        key = (channel, item, begin, duration)
        if key in seen:
            audit['duplicate_airings'] += 1
            continue
        seen.add(key)
        if duration <= 0:
            audit['nonpositive_broadcast_rows'] += 1
            continue
        broadcast = _Broadcast(channel, item, begin, begin + timedelta(seconds=duration))
        left, right = max(start, broadcast.start), min(end, broadcast.end)
        if left >= right:
            audit['outside_window_broadcast_rows'] += 1
            continue
        if item == '0':
            audit['unlabelled_broadcast_rows'] += 1
        # Keep ID 0 in the sweep: overlapping unknown and known assignments conflict.
        events[channel].extend(((left, 1, broadcast), (right, -1, broadcast)))

    index = {}
    for channel, changes in events.items():
        active = set()
        item_counts = Counter()
        segments = []
        previous = None
        for time, simultaneous in groupby(sorted(changes, key=lambda x: x[0]), key=lambda x: x[0]):
            if previous is not None and previous < time:
                if len(item_counts) > 1:
                    audit['conflicting_schedule_seconds'] += (time - previous) // SECOND
                elif item_counts and '0' not in item_counts:
                    segments.append(_Segment(previous, time, tuple(active)))
            # All changes at one boundary are applied together (half-open semantics).
            for _, delta, broadcast in simultaneous:
                if delta == 1:
                    active.add(broadcast)
                else:
                    active.remove(broadcast)
                item_counts[broadcast.item] += delta
                if item_counts[broadcast.item] == 0:
                    del item_counts[broadcast.item]
            previous = time
        index[channel] = (segments, [segment.end for segment in segments])
    return index


def match_viewings(view_rows: Iterable[Row], schedule_rows: Iterable[Row], *,
                   window_start: datetime, window_end: datetime) -> MatchResult:
    """Return qualifying user/broadcast matches within one explicit time partition.

    Required view columns: user_id, vsetv_id, start_time, stop_time, duraton.
    Required schedule columns: channel_id, start_time, duration, tv_show_id.
    Other columns are ignored. Inputs are never mutated and no files are written.
    Malformed IDs/dates raise ValueError; duration anomalies are returned separately.
    Bounds must be naive whole-second datetimes, with start strictly before end.
    """
    for boundary in (window_start, window_end):
        if (not isinstance(boundary, datetime) or boundary.tzinfo is not None
                or boundary.microsecond != 0):
            raise ValueError('Window bounds must be naive whole-second datetimes')
    if window_start >= window_end:
        raise ValueError('Window start must be before window end')

    names = ('view_rows_read', 'duplicate_view_rows', 'unique_view_rows',
             'duration_mismatch_rows', 'over_24h_view_rows', 'nonpositive_view_rows',
             'mismatch_and_over_24h_rows', 'excluded_view_rows',
             'outside_window_view_rows', 'view_rows_used', 'merged_view_intervals',
             'schedule_rows_read', 'duplicate_airings', 'nonpositive_broadcast_rows',
             'outside_window_broadcast_rows', 'unlabelled_broadcast_rows',
             'conflicting_schedule_seconds', 'positive_overlap_broadcasts',
             'below_threshold_broadcasts', 'matched_broadcasts', 'matched_users')
    audit = Counter(dict.fromkeys(names, 0))
    index = _schedule_index(schedule_rows, window_start, window_end, audit)
    groups = defaultdict(list)
    for row in view_rows:
        audit['view_rows_read'] += 1
        user = str(_integer(row, 'user_id', 1))
        channel = str(_integer(row, 'vsetv_id', 1))
        begin = _date(row, 'start_time', '%Y-%m-%d %H:%M:%S')
        finish = _date(row, 'stop_time', '%Y-%m-%d %H:%M:%S')
        duration = _integer(row, 'duraton')
        groups[user, channel].append((begin, finish, duration))

    matches, quarantined = [], []
    for (user, channel), rows in sorted(groups.items(), key=lambda x: tuple(map(int, x[0]))):
        merged = []
        previous = None
        for row in sorted(rows):
            if row == previous:
                audit['duplicate_view_rows'] += 1
                continue
            previous = row
            audit['unique_view_rows'] += 1
            begin, finish, duration = row
            wall_duration = (finish - begin) // SECOND
            mismatch = duration != wall_duration
            too_long = max(duration, wall_duration) > 86400
            reasons = []
            if mismatch:
                reasons.append('duration_mismatch')
                audit['duration_mismatch_rows'] += 1
            if too_long:
                reasons.append('over_24h')
                audit['over_24h_view_rows'] += 1
            if min(duration, wall_duration) <= 0:
                reasons.append('nonpositive_duration')
                audit['nonpositive_view_rows'] += 1
            if mismatch and too_long:
                audit['mismatch_and_over_24h_rows'] += 1
            if reasons:
                audit['excluded_view_rows'] += 1
                quarantined.append(QuarantinedViewing(user, channel, begin, finish,
                                                       duration, tuple(reasons)))
                continue
            left, right = max(begin, window_start), min(finish, window_end)
            if left >= right:
                audit['outside_window_view_rows'] += 1
                continue
            audit['view_rows_used'] += 1
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))

        audit['merged_view_intervals'] += len(merged)
        segments, ends = index.get(channel, ([], []))
        overlaps = Counter()
        for left, right in merged:
            position = bisect_right(ends, left)
            while position < len(segments) and segments[position].start < right:
                segment = segments[position]
                seconds = (min(right, segment.end) - max(left, segment.start)) // SECOND
                for broadcast in segment.broadcasts:
                    overlaps[broadcast] += seconds
                position += 1
        for broadcast, seconds in overlaps.items():
            audit['positive_overlap_broadcasts'] += 1
            duration = (broadcast.end - broadcast.start) // SECOND
            # Integer comparison preserves half-second thresholds for odd durations.
            if seconds * 2 < min(600, duration):
                audit['below_threshold_broadcasts'] += 1
                continue
            matches.append(BroadcastMatch(user, channel, broadcast.item, broadcast.start,
                                          broadcast.end, duration, seconds))

    matches.sort(key=lambda m: (int(m.user_id), int(m.channel_id), m.broadcast_start,
                                m.broadcast_end, int(m.tv_show_id)))
    audit['matched_broadcasts'] = len(matches)
    audit['matched_users'] = len({m.user_id for m in matches})
    return MatchResult(tuple(matches), tuple(quarantined), dict(audit))
