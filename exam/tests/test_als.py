"""Numerical and protocol checks on tiny synthetic inputs only."""
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

import numpy as np


def row(user, item):
    return {'user_id': str(user), 'tv_show_id': str(item)}


class ALSTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.als'), 'ALS is not implemented yet')
        self.api = importlib.import_module('exam.als')

    def test_aggregation_counts_broadcasts_and_preserves_exact_sorted_ids(self):
        large = '18440717181607000261'
        data = self.api.build_interactions([row(large, 10), row(large, 10), row(2, 2)])
        self.assertEqual(data.user_ids, ('2', large))
        self.assertEqual(data.item_ids, ('2', '10'))
        np.testing.assert_array_equal(data.by_user[1][0], [1])
        np.testing.assert_array_equal(data.by_user[1][1], [2])
        self.assertEqual(data.popularity.counts, {'2': 1, '10': 1})
        self.assertEqual(data.popularity.training_rows, 3)

    def test_scalar_update_has_hand_calculated_answer_and_missing_pair_penalty(self):
        # Y=[1,2], first item observed, c=3, lambda=1:
        # A=3*1^2 + 1*2^2 + 1=8; b=3*1=3; x=3/8.
        fixed = np.array([[1.0], [2.0]])
        obs = [(np.array([0]), np.array([1.0])), (np.array([], dtype=int), np.array([]))]
        actual = self.api.solve_factors(fixed, obs, alpha=2/np.log(2), regularization=1)
        np.testing.assert_allclose(actual, [[0.375], [0.0]], rtol=0, atol=1e-14)

    def test_two_factor_update_matches_dense_normal_equations(self):
        fixed = np.array([[1., 2.], [2., -1.], [0.5, 1.]])
        counts = np.array([2., 0., 1.])
        confidence = 1 + 3 * np.log1p(counts)
        preference = (counts > 0).astype(float)
        expected = np.linalg.solve(fixed.T @ np.diag(confidence) @ fixed + 0.2*np.eye(2),
                                   fixed.T @ np.diag(confidence) @ preference)
        actual = self.api.solve_factors(fixed, [(np.array([0, 2]), np.array([2., 1.]))],
                                        alpha=3, regularization=0.2)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)

    def test_objective_matches_hand_answer_and_dense_sum(self):
        data = self.api.build_interactions([row(1, 1), row(2, 2), row(2, 2)])
        x = np.array([[0.375], [0.]])
        y = np.array([[1.], [2.]])
        alpha = 2/np.log(2)
        config = self.api.ALSConfig(factors=1, iterations=1, alpha=alpha, regularization=1)
        counts = np.array([[1., 0.], [0., 2.]])
        expected = np.sum((1+alpha*np.log1p(counts))*((counts>0)-x@y.T)**2) + np.sum(x*x)+np.sum(y*y)
        actual = self.api.implicit_objective(x, y, data.by_user, config)
        self.assertAlmostEqual(actual, expected, places=12)
        # One user, one observed item and a second unobserved factor:
        # 3*(1-3/8)^2 + (2*3/8)^2 + (3/8)^2 + 1^2+2^2 = 6.875.
        self.assertAlmostEqual(self.api.implicit_objective(x[:1], y, data.by_user[:1], config), 6.875)

    def test_tiny_als_is_finite_decreases_objective_and_is_reproducible(self):
        data = self.api.build_interactions([row(1, 1), row(1, 2), row(2, 2), row(2, 3), row(2, 3)])
        config = self.api.ALSConfig(factors=2, iterations=4, alpha=3, regularization=0.2, seed=7)
        first = self.api.fit_als(data, config)
        second = self.api.fit_als(data, config)
        self.assertEqual(len(first.loss_history), 5)
        self.assertTrue(np.isfinite(first.loss_history).all())
        self.assertTrue(np.all(np.diff(first.loss_history) <= 1e-10))
        self.assertLess(first.loss_history[-1], first.loss_history[0])
        np.testing.assert_array_equal(first.user_factors, second.user_factors)
        np.testing.assert_array_equal(first.item_factors, second.item_factors)
        self.assertEqual(first.user_factors.shape, (2, 2))
        self.assertEqual(first.item_factors.shape, (3, 2))

    def test_ranking_repeats_ties_negative_scores_and_cold_completion(self):
        data = self.api.build_interactions([row(1, 2), row(2, 2), row(1, 10), row(1, 3)])
        model = self.api.fit_als(data, self.api.ALSConfig(factors=1, iterations=1))
        # Force known scores: item 2 and 10 tie, 3 is negative; cold 4,5 fill.
        model = replace(model, user_factors=np.ones((2, 1)), item_factors=np.array([[1.], [-2.], [1.]]))
        self.assertEqual(model.recommend('1', [2, 3, 4, 5, 10, 99]), ('2', '10', '3', '4', '5'))
        self.assertEqual(model.recommend('999', [2, 3, 4, 5, 10, 99]), ('2', '3', '10', '4', '5'))
        # Restrict availability; a trained but unavailable programme cannot appear.
        self.assertEqual(model.recommend('1', [3, 4, 5, 6, 7]), ('3', '4', '5', '6', '7'))
        tied = replace(model, item_factors=np.ones((3, 1)))
        self.assertEqual(tied.recommend('1', [2, 3, 4, 5, 10, 99]), ('2', '3', '10', '4', '5'))

    def test_empty_training_falls_back_and_too_small_pool_raises(self):
        model = self.api.fit_als(self.api.build_interactions([]), self.api.ALSConfig(iterations=1))
        self.assertEqual(model.recommend('1', [10, 2, 3, 4, 5, 6]), ('2', '3', '4', '5', '6'))
        with self.assertRaises(ValueError):
            model.recommend('1', [1, 2, 3, 4])

    def test_bad_configuration_and_ids_fail(self):
        for kwargs in [{'factors': 0}, {'iterations': 0}, {'alpha': -1}, {'regularization': 0},
                       {'alpha': float('nan')}, {'regularization': float('inf')}, {'seed': -1}, {'factors': True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.api.ALSConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.api.build_interactions([row(1, 0)])

    def test_nonfinite_factors_are_rejected(self):
        data = self.api.build_interactions([row(1, 1)])
        model = self.api.fit_als(data, self.api.ALSConfig(factors=1, iterations=1))
        model = replace(model, user_factors=np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            model.recommend('1', [1, 2, 3, 4, 5])

    def test_shared_evaluator_uses_each_users_own_ranking_and_rejects_ineligible_ids(self):
        from exam.evaluation import evaluate_rankings
        popularity = self.api.build_interactions([row(1, 1)]).popularity
        predictions = {'1': [1, 2, 3, 4, 5], '2': [5, 4, 3, 2, 1]}
        result = evaluate_rankings(popularity, [row(1, 1), row(2, 1)], [1, 2, 3, 4, 5],
                                   ['1', '2'], predictions, 'synthetic')
        self.assertAlmostEqual(result.statistics['cohorts']['all_users']['map_at_5'], 0.6)
        self.assertAlmostEqual(result.per_user[1]['ap_at_5'], 0.2)
        for bad in [{'1': [1, 2, 3, 4, 99]}, {'1': [1, 2, 3, 4, 5], '2': [1, 2, 3, 4, 5]}]:
            with self.assertRaises(ValueError):
                evaluate_rankings(popularity, [], [1, 2, 3, 4, 5], ['1'], bad, 'synthetic')


class ALSRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.train_als'), 'ALS runner is not implemented yet')
        self.runner = importlib.import_module('exam.train_als')
        self.api = importlib.import_module('exam.als')

    def test_synthetic_runner_uses_shared_protocol_and_saves_exact_model(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, matched, output = EvaluationRunnerTests().fixture(root)
            before = {p: p.read_bytes() for folder in (source, matched) for p in folder.iterdir()}
            config = self.api.ALSConfig(factors=2, iterations=2, seed=11)
            with redirect_stdout(StringIO()):
                self.runner.run_training(source, matched, output, config)
            stats = json.loads((output/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(stats['cohorts']['all_users']['users'], 1260)
            self.assertEqual(stats['cohorts']['all_users']['empty_target_users'], 1257)
            self.assertEqual(stats['relevant_user_item_pairs'], 4)
            self.assertEqual(stats['cold_item_relevant_pairs'], 1)
            self.assertEqual(stats['candidate_programmes'], 6)
            self.assertEqual(stats['training_users'], 2)
            with np.load(output/'model.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['user_ids'].tolist(), ['1', '2'])
                self.assertEqual(saved['item_ids'].tolist(), ['2', '3', '10'])
                self.assertEqual(saved['user_factors'].shape, (2, 2))
            with (output/'validation-predictions.csv').open(newline='', encoding='utf-8') as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(len(predictions), 1260)
            self.assertEqual(predictions[2]['tv_show_id'], '2 10 3 4 5')  # Cold user.
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            with self.assertRaises(FileExistsError):
                self.runner.run_training(source, matched, output, config)
            # Alter validation targets only: fitted factors must remain identical.
            with (matched/'validation-matches.csv').open('a', encoding='utf-8', newline='') as f:
                csv.writer(f).writerow([3, 99])
            other = root/'other'
            with redirect_stdout(StringIO()):
                self.runner.run_training(source, matched, other, config)
            with np.load(output/'model.npz', allow_pickle=False) as a, np.load(other/'model.npz', allow_pickle=False) as b:
                np.testing.assert_array_equal(a['user_factors'], b['user_factors'])
                np.testing.assert_array_equal(a['item_factors'], b['item_factors'])


if __name__ == '__main__':
    unittest.main()
