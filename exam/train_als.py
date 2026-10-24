"""Fit training-only implicit ALS and evaluate the agreed validation hold-out.

Run explicitly: python -B exam/train_als.py
Importing this module or using --help never loads data or trains a model.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np

if __package__:
    from .als import ALSConfig, Interactions, build_interactions, fit_als
    from .evaluate_popularity import load_protocol, _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from .evaluation import evaluate_rankings
else:
    from als import ALSConfig, Interactions, build_interactions, fit_als
    from evaluate_popularity import load_protocol, _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from evaluation import evaluate_rankings


def _rows(path):
    with path.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'user_id', 'tv_show_id'} <= set(reader.fieldnames or ()):
            raise ValueError(f'Missing match ID columns: {path}')
        yield from reader


def run_training(source_dir: Path, matching_dir: Path, output_dir: Path, config: ALSConfig) -> dict:
    """Fit once on train matches, then score validation; never overwrite results."""
    source_dir, matching_dir, output_dir = map(Path, (source_dir, matching_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    started = time.perf_counter()
    inputs = prepare_inputs(source_dir, matching_dir)
    return train_prepared(inputs, output_dir, config, started=started)


@dataclass(frozen=True)
class PreparedInputs:
    source_dir: Path
    matching_dir: Path
    users: tuple[str, ...]
    candidates: frozenset[str]
    data: Interactions


def prepare_inputs(source_dir: Path, matching_dir: Path) -> PreparedInputs:
    source_dir, matching_dir = map(Path, (source_dir, matching_dir))
    users, candidates = load_protocol(source_dir, matching_dir)
    print('Aggregating training-only qualifying broadcasts...', flush=True)
    data = build_interactions(_rows(matching_dir / 'train-matches.csv'))
    data.popularity.top_five(candidates)  # Reject impossible candidate pools before training.
    return PreparedInputs(source_dir, matching_dir, tuple(users), candidates, data)


def train_prepared(inputs: PreparedInputs, output_dir: Path, config: ALSConfig, *, started=None) -> dict:
    """Fit a fresh seeded model using shared immutable-by-convention training inputs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    if started is None:
        started = time.perf_counter()
    source_dir, matching_dir = inputs.source_dir, inputs.matching_dir
    users, candidates, data = inputs.users, inputs.candidates, inputs.data
    training_started = time.perf_counter()

    def progress(iteration, loss):
        print(f'ALS iteration {iteration}/{config.iterations}: objective={loss:.10g}', flush=True)
    model = fit_als(data, config, progress=progress)
    training_seconds = time.perf_counter() - training_started
    print('Ranking validation candidates and evaluating fixed users...', flush=True)
    predictions = {user: model.recommend(user, candidates) for user in users}
    result = evaluate_rankings(data.popularity, _rows(matching_dir / 'validation-matches.csv'),
                               candidates, users, predictions, 'implicit_als')
    training_window = {'start_inclusive': TRAIN_START.isoformat(),
                       'end_exclusive': VALIDATION_START.isoformat()}
    supported_candidates = len(candidates.intersection(model.item_ids))
    cold_users = sum(user not in model.user_index for user in users)
    statistics = {
        **result.statistics,
        'parameters': asdict(config),
        'training_window': training_window,
        'validation_window': {'start_inclusive': VALIDATION_START.isoformat(),
                              'end_exclusive': VALIDATION_END.isoformat()},
        'fallback': {'cold_users': cold_users,
                     'warm_users_needing_completion': (len(users)-cold_users) if supported_candidates < 5 else 0,
                     'supported_candidates': supported_candidates},
        'training_seconds': round(training_seconds, 3),
        'total_seconds': round(time.perf_counter() - started, 3),
        'inputs': {'source_directory': str(source_dir.resolve()),
                   'matching_directory': str(matching_dir.resolve())},
    }
    training_record = {
        'method': 'implicit_als',
        'parameters': asdict(config),
        'confidence': ('1 + alpha * log1p(number of qualifying training broadcast matches)'
                       if config.confidence_weighting == 'log' else
                       '1 + alpha * indicator(number of qualifying training broadcast matches > 0)'),
        'preference': '1 for observed training pairs, 0 otherwise',
        'unobserved_confidence': 1,
        'regularization': 'lambda * (sum of squared user factors + sum of squared item factors), unscaled',
        'dtype': 'float64',
        'training_window': training_window,
        'training_users': len(data.user_ids),
        'training_programmes': len(data.item_ids),
        'training_match_rows': data.popularity.training_rows,
        'training_user_item_pairs': sum(len(indices) for indices, _ in data.by_user),
        'loss_history': [{'iteration': step, 'objective': loss} for step, loss in enumerate(model.loss_history)],
        'training_seconds': round(training_seconds, 3),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'reference': 'https://yifanhu.net/PUB/cf.pdf',
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    # Unicode ID arrays preserve even IDs beyond signed int64; no pickle needed.
    np.savez_compressed(output_dir / 'model.npz',
                        user_ids=np.asarray(model.user_ids, dtype=str),
                        item_ids=np.asarray(model.item_ids, dtype=str),
                        user_factors=model.user_factors, item_factors=model.item_factors,
                        popularity_counts=np.asarray([model.popularity.counts[i] for i in model.item_ids], dtype=np.int64))
    for name, value in [('training.json', training_record), ('evaluation.json', statistics)]:
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    _save_csv(output_dir / 'validation-predictions.csv', ['user_id', 'tv_show_id'],
              ({'user_id': row['user_id'], 'tv_show_id': row['tv_show_id']} for row in result.per_user))
    _save_csv(output_dir / 'per-user.csv', list(result.per_user[0]), result.per_user)
    _save_csv(output_dir / 'candidate-popularity.csv', ['tv_show_id', 'training_distinct_users'],
              ({'tv_show_id': item, 'training_distinct_users': model.popularity.counts.get(item, 0)}
               for item in model.popularity.rank(candidates)))
    print(json.dumps(statistics['cohorts'], indent=2), flush=True)
    print(f'Saved ALS outputs to {output_dir.resolve()}', flush=True)
    return statistics


def main():
    exam_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train implicit ALS and evaluate the fixed temporal hold-out.')
    parser.add_argument('--source-dir', type=Path, default=exam_dir / 'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam_dir / 'derived' / 'matching')
    parser.add_argument('--output-dir', type=Path, default=exam_dir / 'derived' / 'als')
    parser.add_argument('--factors', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=20.0)
    parser.add_argument('--regularization', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence-weighting', choices=('log', 'binary'), default='log')
    args = parser.parse_args()
    config = ALSConfig(args.factors, args.iterations, args.alpha, args.regularization, args.seed,
                       args.confidence_weighting)
    run_training(args.source_dir, args.matching_dir, args.output_dir, config)


if __name__ == '__main__':
    main()
