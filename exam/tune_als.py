"""Fixed eight-run confidence/regularization experiment. No work on import/--help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

if __package__:
    from .als import ALSConfig
    from .train_als import prepare_inputs, train_prepared, _rows
    from .evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from .evaluation import evaluate_rankings
else:
    from als import ALSConfig
    from train_als import prepare_inputs, train_prepared, _rows
    from evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from evaluation import evaluate_rankings


def default_grid() -> tuple[ALSConfig, ...]:
    return tuple(ALSConfig(alpha=alpha, regularization=reg, confidence_weighting=weighting)
                 for weighting in ('log', 'binary') for alpha in (1.0, 20.0) for reg in (0.1, 10.0))


def select_best(rows):
    """Rank by primary validation MAP only; deterministic ties, popularity wins ties."""
    als = [row for row in rows if row['method'] == 'implicit_als']
    key = lambda row: (-row['map_at_5'], row['run_id'])
    best_als = min(als, key=key)['run_id'] if als else None
    best_overall = min(rows, key=lambda row: (-row['map_at_5'], row['method'] != 'popularity', row['run_id']))['run_id']
    return {'best_als': best_als, 'best_overall': best_overall}


def _comparison_row(run_id, stats, popularity_score, config=None, final_loss=None):
    cohorts = stats['cohorts']
    row = {
        'run_id': run_id,
        'method': 'implicit_als' if config else 'popularity',
        'confidence_weighting': config.confidence_weighting if config else None,
        'alpha': config.alpha if config else None,
        'regularization': config.regularization if config else None,
        'factors': config.factors if config else None,
        'iterations': config.iterations if config else None,
        'seed': config.seed if config else None,
        'map_at_5': cohorts['all_users']['map_at_5'],
        'delta_vs_popularity': cohorts['all_users']['map_at_5'] - popularity_score,
        'users': cohorts['all_users']['users'],
        'empty_target_users': cohorts['all_users']['empty_target_users'],
        'candidate_programmes': stats['candidate_programmes'],
    }
    for name in ('nonempty_targets', 'warm_users', 'cold_start_users'):
        row[f'{name}_map_at_5'] = cohorts[name]['map_at_5']
        row[f'{name}_count'] = cohorts[name]['users']
    row['training_seconds'] = stats.get('training_seconds')
    row['final_objective'] = final_loss
    return row


def _json(path, value):
    # Replace atomically so a failed/interrupted write leaves the previous snapshot readable.
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _comparison(path, rows):
    pending = path.with_suffix('.csv.tmp')
    _save_csv(pending, list(rows[0]), rows)
    pending.replace(path)


def run_experiment(source_dir: Path, matching_dir: Path, output_dir: Path, *, configurations=None):
    """Run a declared grid; validation selects settings but never changes fitted inputs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    configs = tuple(default_grid() if configurations is None else configurations)
    if not configs or not all(isinstance(config, ALSConfig) for config in configs):
        raise ValueError('A nonempty grid of ALSConfig values is required')
    if len(set(configs)) != len(configs):
        raise ValueError('Duplicate configurations in grid')
    if len({(c.factors, c.iterations, c.seed) for c in configs}) != 1:
        raise ValueError('Keep factors, iterations and seed fixed across configurations')
    started = time.perf_counter()
    inputs = prepare_inputs(source_dir, matching_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running',
        'selection_metric': 'all_users MAP@5; include empty targets',
        'selection_caveat': 'Validation-selected results are selection scores, not an unbiased test estimate.',
        'loss_caveat': 'Objectives differ with confidence/regularization; do not compare losses to select a model.',
        'tie_break': 'ALS ties: ascending run ID; overall tie with popularity: retain popularity.',
        'configurations': [{'run_id': f'als-{i:02d}', **asdict(c)} for i, c in enumerate(configs, 1)],
        'completed_runs': [],
        'training_window': {'start_inclusive': TRAIN_START.isoformat(), 'end_exclusive': VALIDATION_START.isoformat()},
        'validation_window': {'start_inclusive': VALIDATION_START.isoformat(), 'end_exclusive': VALIDATION_END.isoformat()},
        'inputs': {'source_directory': str(inputs.source_dir.resolve()),
                   'matching_directory': str(inputs.matching_dir.resolve())},
        'current_run': 'popularity',
    }
    _json(output_dir/'experiment.json', manifest)
    try:
        popularity = inputs.data.popularity
        top = popularity.top_five(inputs.candidates)
        result = evaluate_rankings(popularity, _rows(inputs.matching_dir/'validation-matches.csv'),
                                   inputs.candidates, inputs.users,
                                   {user: top for user in inputs.users}, 'distinct_training_user_popularity')
        baseline_dir = output_dir/'popularity'
        baseline_dir.mkdir()
        _json(baseline_dir/'evaluation.json', result.statistics)
        _save_csv(baseline_dir/'per-user.csv', list(result.per_user[0]), result.per_user)
        _save_csv(baseline_dir/'validation-predictions.csv', ['user_id', 'tv_show_id'],
                  ({'user_id': r['user_id'], 'tv_show_id': r['tv_show_id']} for r in result.per_user))
        baseline = result.statistics['cohorts']['all_users']['map_at_5']
        rows = [_comparison_row('popularity', result.statistics, baseline)]
        _comparison(output_dir/'comparison.csv', rows)
        for i, config in enumerate(configs, 1):
            run_id = f'als-{i:02d}'
            manifest['current_run'] = run_id
            _json(output_dir/'experiment.json', manifest)
            print(f'\nTuning {i}/{len(configs)}: {run_id} {asdict(config)}', flush=True)
            stats = train_prepared(inputs, output_dir/run_id, config)
            training = json.loads((output_dir/run_id/'training.json').read_text(encoding='utf-8'))
            rows.append(_comparison_row(run_id, stats, baseline, config,
                                         training['loss_history'][-1]['objective']))
            _comparison(output_dir/'comparison.csv', rows)
            manifest['completed_runs'].append(run_id)
            _json(output_dir/'experiment.json', manifest)
        manifest.update(status='complete', current_run=None, **select_best(rows))
        manifest['total_seconds'] = round(time.perf_counter()-started, 3)
        _json(output_dir/'experiment.json', manifest)
    except (Exception, KeyboardInterrupt) as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        _json(output_dir/'experiment.json', manifest)
        raise
    print(f"Best ALS: {manifest['best_als']}; best overall: {manifest['best_overall']}", flush=True)
    print(f'Saved comparison to {(output_dir / "comparison.csv").resolve()}', flush=True)
    return manifest


def main():
    exam_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run eight fixed ALS confidence/regularization configurations plus popularity.')
    parser.add_argument('--source-dir', type=Path, default=exam_dir/'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam_dir/'derived'/'matching')
    parser.add_argument('--output-dir', type=Path, default=exam_dir/'derived'/'als-tuning')
    args = parser.parse_args()
    run_experiment(args.source_dir, args.matching_dir, args.output_dir)


if __name__ == '__main__':
    main()
