"""Known-answer tests; any file evaluation uses temporary synthetic data only."""
import csv
from contextlib import redirect_stdout
from datetime import datetime
import importlib
import importlib.util
import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest


START = datetime(2020, 5, 18)
END = datetime(2020, 7, 27)


def interaction(user, item):
    return {'user_id': str(user), 'tv_show_id': str(item)}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.evaluation'),
                             'Popularity and MAP@5 are not implemented yet')
        self.api = importlib.import_module('exam.evaluation')

    def test_ap_known_answers(self):
        cases = [
            ('perfect', [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
            ('no hits', [6, 7], [1, 2, 3, 4, 5], 0.0),
            ('empty target', [], [1, 2, 3, 4, 5], 0.0),
            ('hits at one and three', [1, 2, 3], [1, 9, 2, 8, 7], 5 / 9),
            ('only fifth hits', [1], [9, 8, 7, 6, 1], 1 / 5),
            ('denominator capped at five', list(range(1, 11)), [1, 2, 3, 4, 5], 1.0),
            ('duplicate relevance is binary', [1, 1, 2], [1, 2, 3, 4, 5], 1.0),
        ]
        for name, relevant, prediction, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(self.api.average_precision_at_5(relevant, prediction), expected)

    def test_rank_order_changes_ap_without_changing_hits(self):
        first = self.api.average_precision_at_5([1, 2], [1, 2, 8, 9, 10])
        last = self.api.average_precision_at_5([1, 2], [8, 9, 10, 1, 2])
        self.assertEqual(first, 1.0)
        self.assertAlmostEqual(last, (1 / 4 + 2 / 5) / 2)

    def test_invalid_rankings_are_rejected_even_for_empty_targets(self):
        for ranked in [[1, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
                       [0, 1, 2, 3, 4], [1.0, 2, 3, 4, 5], '12345']:
            with self.subTest(ranked=ranked), self.assertRaises(ValueError):
                self.api.average_precision_at_5([], ranked)

    def test_map_is_macro_average_including_empty_targets(self):
        relevant = {'1': [1], '2': [1]}  # User 3 has no validation rows.
        predictions = {'1': [1, 2, 3, 4, 5], '2': [5, 4, 3, 2, 1], '3': [1, 2, 3, 4, 5]}
        self.assertAlmostEqual(
            self.api.mean_average_precision_at_5(relevant, predictions, ['1', '2', '3']), 0.4)

    def test_missing_predictions_duplicate_users_and_empty_cohort_raise(self):
        for users, predictions in [(['1', '2'], {'1': [1, 2, 3, 4, 5]}),
                                   (['1', '1'], {'1': [1, 2, 3, 4, 5]}), ([], {})]:
            with self.subTest(users=users), self.assertRaises(ValueError):
                self.api.mean_average_precision_at_5({}, predictions, users)

    def test_popularity_uses_distinct_users_and_numeric_ties(self):
        training = [interaction(1, 10)] * 4 + [interaction(2, 10), interaction(1, 2),
                    interaction(2, 2), interaction(2, 3), interaction(9000, 99)]
        model = self.api.fit_popularity(iter(training))
        self.assertEqual(model.counts, {'10': 2, '2': 2, '3': 1, '99': 1})
        self.assertEqual(model.training_users, frozenset({'1', '2', '9000'}))
        # 99 is trained but unavailable. Zero-support 4 and 5 remain candidates.
        self.assertEqual(model.top_five([10, 2, 3, 99 + 1, 5, 4]), ('2', '10', '3', '4', '5'))

    def test_zero_support_candidates_fill_five_and_small_pool_is_rejected(self):
        model = self.api.fit_popularity([])
        self.assertEqual(model.top_five([10, 2, 9, 5, 4, 3]), ('2', '3', '4', '5', '9'))
        with self.assertRaises(ValueError):
            model.top_five([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            model.top_five([0, 1, 2, 3, 4, 5])

    def test_candidate_window_keeps_crossing_airings_and_cold_items(self):
        def row(time, duration, item):
            return {'start_time': time, 'duration': str(duration), 'tv_show_id': str(item)}
        schedule = [row('17.05.2020 23:59:00', 120, 1),
                    row('17.05.2020 23:59:00', 60, 2),
                    row('27.07.2020 00:00:00', 60, 3),
                    row('26.07.2020 23:59:00', 120, 99),
                    row('18.05.2020 00:00:00', 0, 4),
                    row('18.05.2020 00:00:00', 600, 0)]
        self.assertEqual(self.api.candidate_programmes(schedule, START, END), frozenset({'1', '99'}))

    def test_evaluation_keeps_seen_items_cold_targets_and_fixed_users(self):
        training = [interaction(1, 2), interaction(2, 2), interaction(1, 10),
                    interaction(2, 10), interaction(2, 3)]
        validation = [interaction(1, 2), interaction(1, 2), interaction(1, 99),
                      interaction(2, 10), interaction(3, 3)]
        result = self.api.evaluate_popularity(training, validation, [2, 3, 4, 5, 10, 99], ['1', '2', '3', '4'])
        self.assertEqual(result.top_five, ('2', '10', '3', '4', '5'))
        stats = result.statistics
        self.assertAlmostEqual(stats['cohorts']['all_users']['map_at_5'], 1 / 3)
        self.assertAlmostEqual(stats['cohorts']['nonempty_targets']['map_at_5'], 4 / 9)
        self.assertAlmostEqual(stats['cohorts']['warm_users']['map_at_5'], 0.5)
        self.assertAlmostEqual(stats['cohorts']['cold_start_users']['map_at_5'], 1 / 6)
        self.assertEqual(stats['cohorts']['all_users']['empty_target_users'], 1)
        self.assertEqual(stats['relevant_user_item_pairs'], 4)
        self.assertEqual(stats['cold_item_relevant_pairs'], 1)  # 99 stays in the denominator.
        self.assertEqual(stats['zero_support_candidates'], 3)
        self.assertEqual([u['user_id'] for u in result.per_user], ['1', '2', '3', '4'])

    def test_undefined_auxiliary_cohort_mean_is_none(self):
        result = self.api.evaluate_popularity([interaction(1, 1)], [], [1, 2, 3, 4, 5], ['1'])
        self.assertEqual(result.statistics['cohorts']['all_users']['map_at_5'], 0.0)
        self.assertIsNone(result.statistics['cohorts']['nonempty_targets']['map_at_5'])
        self.assertIsNone(result.statistics['cohorts']['cold_start_users']['map_at_5'])

    def test_validation_item_outside_pool_raises_instead_of_dropping_target(self):
        with self.assertRaises(ValueError):
            self.api.evaluate_popularity([], [interaction(1, 99)], [1, 2, 3, 4, 5], ['1'])

    def test_large_user_ids_stay_exact(self):
        user = '18440717181607000261'
        result = self.api.evaluate_popularity([interaction(user, 1)], [interaction(user, 1)],
                                               [1, 2, 3, 4, 5], [user])
        self.assertEqual(result.per_user[0]['user_id'], user)
        self.assertEqual(result.statistics['cohorts']['warm_users']['users'], 1)


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.evaluate_popularity'),
                             'The evaluation runner is not implemented yet')
        self.runner = importlib.import_module('exam.evaluate_popularity')

    def fixture(self, root):
        source, matched, output = root / 'source', root / 'matched', root / 'output'
        source.mkdir(); matched.mkdir()
        def write(path, header, rows):
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
        write(source / 'submission.csv', ['user_id', 'tv_show_id'],
              [(str(u), '0 0 0 0 0') for u in range(1, 1261)])
        schedule_header = ['start_time', 'duration', 'tv_show_id']
        write(source / 'export_arh_11-20-final.csv', schedule_header,
              [('17.05.2020 23:59:00', 120, 2)])
        write(source / 'export_arh_21-30-final.csv', schedule_header,
              [('18.05.2020 00:00:00', 600, i) for i in [3, 4, 5, 10, 99]])
        write(matched / 'train-matches.csv', ['user_id', 'tv_show_id'],
              [(1, 2), (2, 2), (1, 10), (2, 10), (2, 3)])
        write(matched / 'validation-matches.csv', ['user_id', 'tv_show_id'],
              [(1, 2), (1, 99), (2, 10), (3, 3)])
        metadata = {'partitions': {}}
        for name, a, b in [('train', '2020-03-09T00:00:00', '2020-05-18T00:00:00'),
                            ('validation', '2020-05-18T00:00:00', '2020-07-27T00:00:00')]:
            metadata['partitions'][name] = {'window_start_inclusive': a, 'window_end_exclusive': b}
        (matched / 'data-quality.json').write_text(json.dumps(metadata), encoding='utf-8')
        return source, matched, output

    def test_runner_writes_complete_fixture_outputs_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, matched, output = self.fixture(root)
            before = {p: p.read_bytes() for folder in [source, matched] for p in folder.iterdir()}
            with redirect_stdout(StringIO()):
                self.runner.run_evaluation(source, matched, output)
            stats = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(stats['cohorts']['all_users']['users'], 1260)
            self.assertAlmostEqual(stats['cohorts']['all_users']['map_at_5'], 1 / 945)
            self.assertEqual(stats['cohorts']['all_users']['empty_target_users'], 1257)
            with (output / 'validation-predictions.csv').open(newline='', encoding='utf-8') as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(len(predictions), 1260)
            self.assertEqual(predictions[0], {'user_id': '1', 'tv_show_id': '2 10 3 4 5'})
            with (output / 'per-user.csv').open(newline='', encoding='utf-8') as f:
                scores = list(csv.DictReader(f))
            self.assertEqual(len(scores), 1260)
            self.assertEqual(scores[0]['relevant_ids'], '2 99')
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            with self.assertRaises(FileExistsError):
                self.runner.run_evaluation(source, matched, output)

    def test_runner_rejects_wrong_partition_provenance_without_creating_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = self.fixture(Path(directory))
            p = matched / 'data-quality.json'
            meta = json.loads(p.read_text())
            meta['partitions']['train']['window_end_exclusive'] = '2020-07-27T00:00:00'
            p.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.runner.run_evaluation(source, matched, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
