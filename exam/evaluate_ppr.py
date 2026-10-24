"""Build a training co-view graph and evaluate PPR only on explicit invocation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import sparse

if __package__:
    from .ppr import PPRConfig, build_graph, predict_users, MASS_TOLERANCE
    from .train_als import prepare_inputs, _rows
    from .evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from .evaluation import evaluate_rankings
else:
    from ppr import PPRConfig, build_graph, predict_users, MASS_TOLERANCE
    from train_als import prepare_inputs, _rows
    from evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from evaluation import evaluate_rankings


def run_evaluation(source_dir: Path, matching_dir: Path, output_dir: Path,
                   config: PPRConfig, batch_size=32) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    started = time.perf_counter()
    inputs = prepare_inputs(source_dir, matching_dir)  # Aggregation only; no ALS fitting.
    print('Building distinct-user programme co-view graph...', flush=True)
    graph_started = time.perf_counter()
    graph = build_graph(inputs.data)
    graph_seconds = time.perf_counter()-graph_started
    ranked = predict_users(inputs.data, graph, inputs.users, inputs.candidates, config, batch_size,
                           progress=lambda done, total: print(f'PPR converged for {done}/{total} warm users', flush=True))
    predictions, diagnostics = ranked.predictions, ranked.diagnostics
    warm_users, saved_scores = ranked.warm_users, ranked.scores
    popularity = inputs.data.popularity
    print('Evaluating PPR with the shared fixed-user protocol...', flush=True)
    result = evaluate_rankings(popularity, _rows(inputs.matching_dir/'validation-matches.csv'),
                               inputs.candidates, inputs.users, predictions, 'personalized_pagerank')
    warm_diagnostics = [diagnostics[user] for user in warm_users]
    convergence = {
        'converged_users': len(warm_users),
        'cold_fallback_users': len(inputs.users)-len(warm_users),
        'maximum_iterations_used': max((row['iterations'] for row in warm_diagnostics), default=0),
        'maximum_fixed_point_residual': max((row['fixed_point_residual'] for row in warm_diagnostics), default=None),
        'maximum_probability_mass_error': max((row['max_mass_error'] for row in warm_diagnostics), default=None),
        'mass_tolerance': MASS_TOLERANCE,
    }
    training_window = {'start_inclusive': TRAIN_START.isoformat(), 'end_exclusive': VALIDATION_START.isoformat()}
    validation_window = {'start_inclusive': VALIDATION_START.isoformat(), 'end_exclusive': VALIDATION_END.isoformat()}
    graph_record = {
        **graph.diagnostics, 'parameters': asdict(config), 'training_window': training_window,
        'construction': 'Binary training incidence B; W=B.T@B; diagonal removed; no pruning.',
        'edge_weight': 'Number of distinct training users watching both programmes.',
        'personalization': 'Uniform over distinct qualifying training programmes per user.',
        'dangling_policy': 'Redistribute dangling probability mass to the same user personalization vector.',
        'convergence_rule': 'Absolute per-user L1 iterate change <= tolerance; check final fixed-point residual.',
        'training_users': len(inputs.data.user_ids), 'training_match_rows': popularity.training_rows,
        'graph_seconds': round(graph_seconds, 3), 'batch_size': batch_size,
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    }
    statistics = {
        **result.statistics, 'parameters': asdict(config), 'training_window': training_window,
        'validation_window': validation_window, 'graph': graph.diagnostics, 'convergence': convergence,
        'graph_seconds': round(graph_seconds, 3), 'total_seconds': round(time.perf_counter()-started, 3),
        'inputs': {'source_directory': str(inputs.source_dir.resolve()),
                   'matching_directory': str(inputs.matching_dir.resolve())},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(output_dir/'graph.npz', graph.weights)
    sparse.save_npz(output_dir/'transition.npz', graph.transition)
    np.savez_compressed(output_dir/'ppr-scores.npz', user_ids=np.asarray(warm_users, dtype=str),
                        item_ids=np.asarray(graph.item_ids, dtype=str), scores=saved_scores)
    degrees = np.asarray(graph.weights.sum(axis=1)).ravel()
    _save_csv(output_dir/'graph-items.csv', ['node_index', 'tv_show_id', 'training_distinct_users', 'weighted_degree', 'dangling'],
              ({'node_index': i, 'tv_show_id': item, 'training_distinct_users': popularity.counts[item],
                'weighted_degree': float(degrees[i]), 'dangling': bool(degrees[i] == 0)}
               for i, item in enumerate(graph.item_ids)))
    for filename, record in [('graph.json', graph_record), ('evaluation.json', statistics)]:
        (output_dir/filename).write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    _save_csv(output_dir/'per-user.csv', list(result.per_user[0]), result.per_user)
    _save_csv(output_dir/'validation-predictions.csv', ['user_id', 'tv_show_id'],
              ({'user_id': row['user_id'], 'tv_show_id': row['tv_show_id']} for row in result.per_user))
    _save_csv(output_dir/'ppr-diagnostics.csv', list(diagnostics[inputs.users[0]]),
              (diagnostics[user] for user in inputs.users))
    _save_csv(output_dir/'candidate-popularity.csv', ['tv_show_id', 'training_distinct_users'],
              ({'tv_show_id': item, 'training_distinct_users': popularity.counts.get(item, 0)}
               for item in popularity.rank(inputs.candidates)))
    print(json.dumps(statistics['cohorts'], indent=2), flush=True)
    print(f'Saved PPR outputs to {output_dir.resolve()}', flush=True)
    return statistics


def main():
    exam_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Evaluate training-only Personalized PageRank on the programme co-view graph.')
    parser.add_argument('--source-dir', type=Path, default=exam_dir/'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam_dir/'derived'/'matching')
    parser.add_argument('--output-dir', type=Path, default=exam_dir/'derived'/'ppr')
    parser.add_argument('--damping', type=float, default=0.85)
    parser.add_argument('--tolerance', type=float, default=1e-10)
    parser.add_argument('--max-iterations', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run_evaluation(args.source_dir, args.matching_dir, args.output_dir,
                    PPRConfig(args.damping, args.tolerance, args.max_iterations), args.batch_size)


if __name__ == '__main__':
    main()
