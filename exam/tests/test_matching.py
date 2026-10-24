"""Small hand-computed fixtures; no real data files or third-party packages."""
import copy
import csv
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import io
import unittest


BASE = datetime(2020, 5, 18)
USER = '18440717181607000261'


def view(start, end, *, user=USER, channel='7', duration=None):
    return {'user_id': user, 'vsetv_id': channel,
            'start_time': (BASE + timedelta(seconds=start)).strftime('%Y-%m-%d %H:%M:%S'),
            'stop_time': (BASE + timedelta(seconds=end)).strftime('%Y-%m-%d %H:%M:%S'),
            'duraton': str(end - start if duration is None else duration)}


def airing(start, duration, *, item='101', channel='7'):
    return {'channel_id': channel,
            'start_time': (BASE + timedelta(seconds=start)).strftime('%d.%m.%Y %H:%M:%S'),
            'duration': str(duration), 'tv_show_id': item}


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.matching'),
                             'The viewing-to-programme matcher is not implemented yet')
        self.match = importlib.import_module('exam.matching').match_viewings

    def run_match(self, views, schedule, start=-3600, end=7200):
        return self.match(views, schedule,
                          window_start=BASE + timedelta(seconds=start),
                          window_end=BASE + timedelta(seconds=end))

    def test_long_programme_accepts_300_seconds_but_not_299(self):
        result = self.run_match([view(0, 299, user='1'), view(0, 300)], [airing(0, 3600)])
        self.assertEqual([(m.user_id, m.overlap_seconds) for m in result.matches], [(USER, 300)])

    def test_short_programme_requires_half_including_odd_duration(self):
        for length, watched, expected in [(240, 119, 0), (240, 120, 1), (121, 60, 0), (121, 61, 1)]:
            with self.subTest(length=length, watched=watched):
                result = self.run_match([view(0, watched)], [airing(0, length)])
                self.assertEqual(len(result.matches), expected)

    def test_channel_must_match_and_touching_endpoints_give_no_overlap(self):
        result = self.run_match([view(-600, 0), view(600, 1200), view(0, 600, channel='8')],
                                [airing(0, 600)])
        self.assertEqual(result.matches, ())

    def test_duplicate_and_overlapping_views_count_union_seconds_without_mutation(self):
        views = [view(100, 300), view(0, 200), view(0, 200)]
        schedule = [airing(0, 600)]
        original = copy.deepcopy((views, schedule))
        result = self.run_match(views, schedule)
        self.assertEqual([m.overlap_seconds for m in result.matches], [300])
        self.assertEqual(result.audit['duplicate_view_rows'], 1)
        self.assertEqual((views, schedule), original)

    def test_disjoint_visits_accumulate_within_one_broadcast(self):
        result = self.run_match([view(450, 600), view(0, 150)], [airing(0, 600)])
        self.assertEqual([m.overlap_seconds for m in result.matches], [300])

    def test_repeated_programme_airings_remain_separate_interactions(self):
        result = self.run_match([view(0, 900)], [airing(600, 600), airing(0, 600)])
        self.assertEqual([(m.tv_show_id, m.overlap_seconds) for m in result.matches],
                         [('101', 600), ('101', 300)])

    def test_window_clipping_uses_full_broadcast_duration_and_separate_partitions(self):
        schedule = [airing(-300, 600)]
        before = self.run_match([view(-300, 299)], schedule, start=-300, end=0)
        after = self.run_match([view(-300, 299)], schedule, start=0, end=300)
        self.assertEqual([m.overlap_seconds for m in before.matches], [300])
        self.assertEqual(before.matches[0].duration_seconds, 600)
        self.assertEqual(after.matches, ())  # 299 is not half of the full 600 seconds.

    def test_quarantine_reasons_and_intersection_are_not_double_counted(self):
        views = [view(0, 600, duration=599), view(0, 90000),
                 view(0, 90000, user='2', duration=89999)]
        result = self.run_match(views, [airing(0, 3600)])
        self.assertEqual(result.matches, ())
        self.assertEqual(result.audit['duration_mismatch_rows'], 2)
        self.assertEqual(result.audit['over_24h_view_rows'], 2)
        self.assertEqual(result.audit['mismatch_and_over_24h_rows'], 1)
        self.assertEqual(result.audit['excluded_view_rows'], 3)
        self.assertEqual(len(result.quarantined), 3)
        both = next(q for q in result.quarantined if q.user_id == '2')
        self.assertEqual(set(both.reasons), {'duration_mismatch', 'over_24h'})

    def test_exactly_24_hours_is_allowed_but_nonpositive_views_are_not(self):
        result = self.run_match([view(0, 86400), view(0, 0), view(600, 0)], [airing(0, 600)])
        self.assertEqual([m.overlap_seconds for m in result.matches], [600])
        self.assertEqual(result.audit['nonpositive_view_rows'], 2)

    def test_zero_ids_and_nonpositive_broadcasts_never_create_matches(self):
        result = self.run_match([view(0, 600)],
                                [airing(0, 600, item='0'), airing(0, 0), airing(0, -1)])
        self.assertEqual(result.matches, ())
        self.assertEqual(result.audit['nonpositive_broadcast_rows'], 2)

    def test_conflicting_schedule_time_is_removed_from_both_items(self):
        # A: [0,600), B: [200,800). Conflict [200,600) removes 400 seconds.
        result = self.run_match([view(0, 800)], [airing(200, 600, item='102'), airing(0, 600)])
        self.assertEqual(result.matches, ())  # Each retains 200, below its 300-second threshold.
        self.assertEqual(result.audit['conflicting_schedule_seconds'], 400)

    def test_nested_zero_id_conflict_preserves_only_unambiguous_seconds(self):
        result = self.run_match([view(0, 600)], [airing(0, 600), airing(200, 200, item='0')])
        self.assertEqual([m.overlap_seconds for m in result.matches], [400])
        self.assertEqual(result.audit['conflicting_schedule_seconds'], 200)

    def test_same_item_overlap_is_not_conflict_and_exact_airing_duplicate_is_removed(self):
        result = self.run_match([view(0, 800)],
                                [airing(0, 600), airing(200, 600), airing(0, 600)])
        self.assertEqual([m.overlap_seconds for m in result.matches], [600, 600])
        self.assertEqual(result.audit['conflicting_schedule_seconds'], 0)
        self.assertEqual(result.audit['duplicate_airings'], 1)

    def test_csv_iterators_preserve_large_ids_and_parse_both_timestamp_formats(self):
        views = csv.DictReader(io.StringIO(
            'user_id,vsetv_id,start_time,stop_time,duraton\n'
            f'{USER},7,2020-05-18 00:00:00,2020-05-18 00:05:00,300\n'))
        schedule = csv.DictReader(io.StringIO(
            'channel_id,start_time,duration,tv_show_id\n7,18.05.2020 00:00:00,600,101\n'))
        result = self.run_match(views, schedule)
        self.assertEqual(result.matches[0].user_id, USER)
        self.assertEqual(result.matches[0].broadcast_start, BASE)
        self.assertEqual(result.matches[0].broadcast_end, BASE + timedelta(seconds=600))

    def test_bad_ids_dates_and_windows_fail_explicitly(self):
        for change in [{'user_id': float(USER)}, {'start_time': 'bad date'}, {'vsetv_id': '0'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_match([{**view(0, 600), **change}], [airing(0, 600)])
        with self.assertRaises(ValueError):
            self.run_match([], [], start=0, end=0)
        with self.assertRaises(ValueError):
            self.match([], [], window_start=BASE.replace(tzinfo=timezone.utc),
                       window_end=BASE + timedelta(seconds=600))

    def test_empty_inputs_and_outside_window_views_are_accounted_for(self):
        self.assertEqual(self.run_match([], []).matches, ())
        result = self.run_match([view(600, 900)], [airing(0, 600)], start=0, end=600)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.audit['outside_window_view_rows'], 1)


if __name__ == '__main__':
    unittest.main()
