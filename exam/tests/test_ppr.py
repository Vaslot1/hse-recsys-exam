"""Known answers and tiny synthetic files only; never run the dataset experiment."""
from contextlib import redirect_stdout
import csv
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
from exam.als import build_interactions


def interactions(pairs):
    return build_interactions({'user_id': str(u), 'tv_show_id': str(i)} for u, i in pairs)


class PPRTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.ppr'), 'PPR not implemented')
        self.api = importlib.import_module('exam.ppr')

    def test_coview_weights_are_distinct_users_without_self_loops(self):
        data = interactions([(1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 3), (4, 4)])
        graph = self.api.build_graph(data)
        np.testing.assert_array_equal(graph.weights.toarray(),
            [[0, 2, 1, 0], [2, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_allclose(graph.transition.toarray(),
            [[0, 2/3, 1/3, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(graph.item_ids, ('1', '2', '3', '4'))
        self.assertEqual(graph.diagnostics['dangling_nodes'], 1)
        self.assertEqual(graph.diagnostics['undirected_edges'], 2)

    def test_two_node_cycle_matches_known_answer_and_preserves_mass(self):
        p = sparse.csr_matrix([[0., 1.], [1., 0.]])
        result = self.api.personalized_pagerank(p, [1., 0.], self.api.PPRConfig())
        np.testing.assert_allclose(result.scores[:, 0], [20/37, 17/37], atol=1e-10, rtol=0)
        self.assertLessEqual(result.last_delta[0], 1e-10)
        self.assertLessEqual(result.residual[0], 1e-10)
        self.assertAlmostEqual(result.scores[:, 0].sum(), 1)
        self.assertLessEqual(result.max_mass_error[0], 1e-10)
        self.assertGreaterEqual(result.scores.min(), 0)

    def test_dangling_node_redistributes_to_personalization(self):
        p = sparse.csr_matrix([[0., 1.], [0., 0.]])
        result = self.api.personalized_pagerank(p, [1., 0.], self.api.PPRConfig(damping=0.5))
        np.testing.assert_allclose(result.scores[:, 0], [2/3, 1/3], atol=1e-10, rtol=0)
        isolated = self.api.personalized_pagerank(sparse.csr_matrix((2, 2)), [1., 3.], self.api.PPRConfig())
        np.testing.assert_allclose(isolated.scores[:, 0], [0.25, 0.75])
        self.assertEqual(isolated.iterations[0], 1)

    def test_sparse_iteration_matches_independent_dense_linear_solve(self):
        p = np.array([[0., 0.75, 0.25], [1., 0., 0.], [0., 0., 0.]])
        seed = np.array([0.6, 0.1, 0.3]); damping = 0.8
        effective = p.copy(); effective[2] = seed
        expected = np.linalg.solve(np.eye(3)-damping*effective.T, (1-damping)*seed)
        result = self.api.personalized_pagerank(sparse.csr_matrix(p), seed, self.api.PPRConfig(damping=damping))
        np.testing.assert_allclose(result.scores[:, 0], expected, atol=1e-9, rtol=0)

    def test_batch_agrees_with_separate_users_including_unreachable_components(self):
        p = sparse.csr_matrix([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        seeds = np.array([[1., 0.], [0., 0.], [0., 1.]])
        config = self.api.PPRConfig()
        batch = self.api.personalized_pagerank(p, seeds, config)
        for col in range(2):
            single = self.api.personalized_pagerank(p, seeds[:, col], config)
            np.testing.assert_allclose(batch.scores[:, col], single.scores[:, 0], atol=1e-14)
            self.assertEqual(batch.iterations[col], single.iterations[0])
        self.assertEqual(batch.iterations[1], 1)
        np.testing.assert_array_equal(batch.scores[:, 1], [0., 0., 1.])

    def test_zero_damping_returns_normalized_seed(self):
        p = sparse.csr_matrix([[0., 1.], [1., 0.]])
        result = self.api.personalized_pagerank(p, [2., 3.], self.api.PPRConfig(damping=0))
        np.testing.assert_allclose(result.scores[:, 0], [0.4, 0.6])

    def test_nonconvergence_is_explicit_not_silent_fallback(self):
        with self.assertRaises(self.api.ConvergenceError):
            self.api.personalized_pagerank(sparse.csr_matrix([[0., 1.], [1., 0.]]), [1., 0.],
                                           self.api.PPRConfig(max_iterations=1, tolerance=1e-12))

    def test_invalid_transitions_seeds_weights_and_configurations_fail(self):
        for p in [[[0., 0.5], [1., 0.]], [[-1., 2.], [1., 0.]], [[0., np.nan], [1., 0.]]]:
            with self.assertRaises(ValueError):
                self.api.personalized_pagerank(sparse.csr_matrix(p), [1., 0.], self.api.PPRConfig())
        for seed in [[0., 0.], [-1., 2.], [np.inf, 1.], [1., 0., 0.]]:
            with self.assertRaises(ValueError):
                self.api.personalized_pagerank(sparse.eye(2, format='csr'), seed, self.api.PPRConfig())
        with self.assertRaises(ValueError):
            self.api.transition_matrix(sparse.csr_matrix([[0., -1.], [1., 0.]]))
        for kwargs in [{'damping': 1}, {'damping': -1}, {'damping': float('nan')},
                       {'tolerance': 0}, {'max_iterations': 0}, {'max_iterations': True}]:
            with self.assertRaises(ValueError):
                self.api.PPRConfig(**kwargs)

    def test_ranking_retains_watched_items_uses_ties_and_completes_cold_items(self):
        data = interactions([(1, 2), (2, 2), (1, 3), (1, 10)])
        graph = self.api.build_graph(data)
        ranked = self.api.rank_scores(graph, [1/3, 1/3, 1/3], data.popularity, [2, 3, 4, 5, 10, 99])
        self.assertEqual(ranked, ('2', '3', '10', '4', '5'))
        self.assertEqual(self.api.rank_scores(graph, [0.1, 0.2, 0.7], data.popularity, [3, 4, 5, 6, 7]),
                         ('3', '4', '5', '6', '7'))
        with self.assertRaises(ValueError):
            self.api.rank_scores(graph, [0.1, 0.2, 0.7], data.popularity, [1, 2, 3, 4])

    def test_empty_graph_and_large_item_ids(self):
        empty = self.api.build_graph(interactions([]))
        self.assertEqual(empty.transition.shape, (0, 0))
        self.assertEqual(self.api.rank_scores(empty, [], interactions([]).popularity, [1, 2, 3, 4, 5]),
                         ('1', '2', '3', '4', '5'))
        large = '18440717181607000261'
        graph = self.api.build_graph(interactions([(1, large), (1, 2)]))
        self.assertEqual(graph.item_ids, ('2', large))


class PPRRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exam.evaluate_ppr'), 'PPR runner not implemented')
        self.runner = importlib.import_module('exam.evaluate_ppr')
        self.api = importlib.import_module('exam.ppr')

    def test_synthetic_runner_persists_diagnostics_and_validation_cannot_change_graph(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = EvaluationRunnerTests().fixture(Path(directory))
            before = {p: p.read_bytes() for folder in (source, matched) for p in folder.iterdir()}
            with redirect_stdout(StringIO()):
                self.runner.run_evaluation(source, matched, output, self.api.PPRConfig(), batch_size=1)
            stats = json.loads((output/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(stats['cohorts']['all_users']['users'], 1260)
            self.assertEqual(stats['cohorts']['all_users']['empty_target_users'], 1257)
            self.assertEqual(stats['candidate_programmes'], 6)
            self.assertEqual(stats['cold_item_relevant_pairs'], 1)
            with (output/'validation-predictions.csv').open(newline='', encoding='utf-8') as f:
                predictions = list(csv.DictReader(f))
            self.assertEqual(predictions[2]['tv_show_id'], '2 10 3 4 5')
            with (output/'ppr-diagnostics.csv').open(newline='', encoding='utf-8') as f:
                diagnostics = list(csv.DictReader(f))
            self.assertEqual(len(diagnostics), 1260)
            self.assertEqual(diagnostics[0]['status'], 'converged')
            self.assertEqual(diagnostics[2]['status'], 'popularity_fallback')
            with np.load(output/'ppr-scores.npz', allow_pickle=False) as scores:
                self.assertEqual(scores['user_ids'].tolist(), ['1', '2'])
                self.assertEqual(scores['item_ids'].tolist(), ['2', '3', '10'])
                np.testing.assert_allclose(scores['scores'].sum(axis=1), 1, atol=1e-10)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                self.runner.run_evaluation(source, matched, output, self.api.PPRConfig())
            with (matched/'validation-matches.csv').open('a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([3, 99])
            other = Path(directory)/'other'
            with redirect_stdout(StringIO()):
                self.runner.run_evaluation(source, matched, other, self.api.PPRConfig(), batch_size=2)
            np.testing.assert_array_equal(sparse.load_npz(output/'graph.npz').toarray(),
                                           sparse.load_npz(other/'graph.npz').toarray())
            with np.load(output/'ppr-scores.npz', allow_pickle=False) as a, np.load(other/'ppr-scores.npz', allow_pickle=False) as b:
                np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-14)

    def test_runner_nonconvergence_does_not_save_partial_results(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = EvaluationRunnerTests().fixture(Path(directory))
            with redirect_stdout(StringIO()), self.assertRaises(self.api.ConvergenceError):
                self.runner.run_evaluation(source, matched, output, self.api.PPRConfig(max_iterations=1))
            self.assertFalse(output.exists())

    def test_empty_training_all_users_receive_popularity_without_fake_ppr_scores(self):
        from exam.tests.test_evaluation import EvaluationRunnerTests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source, matched, output = EvaluationRunnerTests().fixture(Path(directory))
            (matched/'train-matches.csv').write_text('user_id,tv_show_id\n', encoding='utf-8')
            with redirect_stdout(StringIO()):
                stats = self.runner.run_evaluation(source, matched, output, self.api.PPRConfig())
            self.assertEqual(stats['convergence']['converged_users'], 0)
            self.assertEqual(stats['convergence']['cold_fallback_users'], 1260)
            self.assertIsNone(stats['convergence']['maximum_fixed_point_residual'])
            with np.load(output/'ppr-scores.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['scores'].shape, (0, 0))


if __name__ == '__main__':
    unittest.main()
