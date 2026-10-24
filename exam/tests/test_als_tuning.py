"""Small synthetic checks; never execute the dataset tuning experiment."""
from contextlib import redirect_stdout
import csv
from dataclasses import replace
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from exam.als import ALSConfig, build_interactions, fit_als, implicit_objective, solve_factors


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIn('confidence_weighting', ALSConfig.__dataclass_fields__, 'Confidence modes not implemented')

    def test_binary_update_ignores_repeat_count_and_matches_hand_answer(self):
        fixed = np.array([[1.], [2.]])
        for count in (1., 7.):
            actual = solve_factors(fixed, [(np.array([0]), np.array([count]))],
                                    alpha=2, regularization=1, confidence_weighting='binary')
            self.assertAlmostEqual(actual[0, 0], 3/8)
        # Missing pair still contributes 2^2; confidence is 3, not 2 or 4.

    def test_binary_objective_matches_dense_reference(self):
        config = ALSConfig(factors=1, alpha=2, regularization=1, confidence_weighting='binary')
        observations = ((np.array([0]), np.array([7.])),)
        loss = implicit_objective(np.array([[0.375]]), np.array([[1.], [2.]]), observations, config)
        self.assertAlmostEqual(loss, 6.875)

    def test_log_default_is_unchanged_and_invalid_mode_fails(self):
        data = build_interactions([{'user_id': str(u), 'tv_show_id': str(i)}
                                   for u, i in [(1, 1), (1, 2), (2, 2), (2, 3), (2, 3)]])
        config = ALSConfig(factors=2, iterations=4, alpha=3, regularization=0.2, seed=7)
        self.assertEqual(config.confidence_weighting, 'log')
        model = fit_als(data, config)
        np.testing.assert_allclose(model.loss_history,
            [13.534204443928, 6.892801912419, 1.929724823139, 1.590958539508, 1.409997181328], atol=1e-10, rtol=0)
        with self.assertRaises(ValueError):
            replace(config, confidence_weighting='unknown')


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.tune_als'), 'Tuning runner not implemented')
        self.api = importlib.import_module('exam.tune_als')

    def test_grid_has_eight_distinct_configs_including_original(self):
        grid = self.api.default_grid()
        self.assertEqual(len(grid), 8)
        self.assertEqual({(c.confidence_weighting, c.alpha, c.regularization) for c in grid},
                         {(w, a, r) for w in ('log', 'binary') for a in (1., 20.) for r in (0.1, 10.)})
        self.assertTrue(all((c.factors, c.iterations, c.seed) == (32, 15, 42) for c in grid))
        self.assertIn(ALSConfig(), grid)

    def test_selection_uses_map_not_loss_with_deterministic_ties(self):
        rows = [dict(run_id='popularity', method='popularity', map_at_5=0.7, final_objective=None),
                dict(run_id='als-02', method='implicit_als', map_at_5=0.5, final_objective=1),
                dict(run_id='als-01', method='implicit_als', map_at_5=0.5, final_objective=100)]
        choice = self.api.select_best(rows)
        self.assertEqual(choice, {'best_als': 'als-01', 'best_overall': 'popularity'})
        rows[1]['map_at_5'] = 0.8
        self.assertEqual(self.api.select_best(rows)['best_overall'], 'als-02')
        rows[1]['map_at_5'] = 0.7
        self.assertEqual(self.api.select_best(rows)['best_overall'], 'popularity')

    def test_tiny_experiment_saves_shared_cohorts_comparison_and_artifacts(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = EvaluationRunnerTests().fixture(Path(directory))
            before = {p: p.read_bytes() for folder in (source, matched) for p in folder.iterdir()}
            configs = [ALSConfig(factors=2, iterations=2, alpha=1, seed=11),
                       ALSConfig(factors=2, iterations=2, alpha=1, seed=11, confidence_weighting='binary')]
            with redirect_stdout(StringIO()):
                result = self.api.run_experiment(source, matched, output, configurations=configs)
            with (output/'comparison.csv').open(newline='', encoding='utf-8') as f:
                comparison = list(csv.DictReader(f))
            self.assertEqual(len(comparison), 3)
            self.assertEqual(result['status'], 'complete')
            manifest = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['completed_runs'], ['als-01', 'als-02'])
            self.assertEqual(len(manifest['configurations']), 2)
            baseline = float(comparison[0]['map_at_5'])
            self.assertAlmostEqual(baseline, 1/945)
            for r in comparison:
                self.assertEqual(int(r['users']), 1260)
                self.assertEqual(int(r['empty_target_users']), 1257)
                self.assertEqual(int(r['candidate_programmes']), 6)
                self.assertAlmostEqual(float(r['delta_vs_popularity']), float(r['map_at_5'])-baseline)
                saved = json.loads((output/r['run_id']/'evaluation.json').read_text(encoding='utf-8'))
                self.assertEqual(float(r['map_at_5']), saved['cohorts']['all_users']['map_at_5'])
                self.assertEqual(saved['cold_item_relevant_pairs'], 1)
                self.assertTrue((output/r['run_id']/'validation-predictions.csv').is_file())
            for name in ('als-01', 'als-02'):
                self.assertTrue((output/name/'model.npz').is_file())
                record = json.loads((output/name/'training.json').read_text(encoding='utf-8'))
                self.assertIn(record['parameters']['confidence_weighting'], ('log', 'binary'))
                self.assertTrue(all(b['objective'] <= a['objective']+1e-8
                                    for a, b in zip(record['loss_history'], record['loss_history'][1:])))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                self.api.run_experiment(source, matched, output, configurations=configs)

    def test_invalid_grid_fails_before_reading_sources(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for configs in [[], [ALSConfig(), ALSConfig()], [ALSConfig(), ALSConfig(seed=1)]]:
                with self.assertRaises(ValueError):
                    self.api.run_experiment(root/'missing', root/'missing', root/'output', configurations=configs)
                self.assertFalse((root/'output').exists())

    def test_failed_run_preserves_completed_comparison_and_marks_incomplete(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = EvaluationRunnerTests().fixture(Path(directory))
            configs = [ALSConfig(factors=2, iterations=1, alpha=1),
                       ALSConfig(factors=2, iterations=1, alpha=1, confidence_weighting='binary')]
            train = self.api.train_prepared
            def fail_second(inputs, path, config):
                if config.confidence_weighting == 'binary':
                    raise RuntimeError('Synthetic failure')
                return train(inputs, path, config)
            with patch.object(self.api, 'train_prepared', side_effect=fail_second), redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
                    self.api.run_experiment(source, matched, output, configurations=configs)
            manifest = json.loads((output/'experiment.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['completed_runs'], ['als-01'])
            self.assertEqual(manifest['current_run'], 'als-02')
            self.assertNotIn('best_overall', manifest)
            with (output/'comparison.csv').open(newline='', encoding='utf-8') as f:
                self.assertEqual([r['run_id'] for r in csv.DictReader(f)], ['popularity', 'als-01'])


if __name__ == '__main__':
    unittest.main()
