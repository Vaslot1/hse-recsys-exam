"""Evaluate the popularity baseline on saved matches; run only when requested.

From the project root: python -B exam/evaluate_popularity.py
Use --help for path overrides. Existing output directories are never overwritten.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
from itertools import chain
import json
from pathlib import Path
import time

if __package__:
    from .evaluation import candidate_programmes, evaluate_popularity
else:
    from evaluation import candidate_programmes, evaluate_popularity


TRAIN_START = datetime(2020, 3, 9)
VALIDATION_START = datetime(2020, 5, 18)
VALIDATION_END = datetime(2020, 7, 27)


def _check_provenance(path):
    metadata = json.loads(path.read_text(encoding='utf-8'))
    for name, start, end in [('train', TRAIN_START, VALIDATION_START),
                             ('validation', VALIDATION_START, VALIDATION_END)]:
        partition = metadata.get('partitions', {}).get(name, {})
        if (partition.get('window_start_inclusive') != start.isoformat()
                or partition.get('window_end_exclusive') != end.isoformat()):
            raise ValueError(f'{name} match provenance does not match the agreed temporal split')


def _save_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(source_dir: Path, matching_dir: Path):
    """Shared split checks, fixed template cohort, and validation availability."""
    source_dir, matching_dir = map(Path, (source_dir, matching_dir))
    _check_provenance(matching_dir / 'data-quality.json')
    with ExitStack() as stack:
        def reader(path):
            stream = stack.enter_context(path.open('r', encoding='utf-8', newline=''))
            return csv.DictReader(stream)

        template = reader(source_dir / 'submission.csv')
        if template.fieldnames != ['user_id', 'tv_show_id']:
            raise ValueError('Submission template must have user_id,tv_show_id columns')
        users = [row['user_id'] for row in template]
        if len(users) != 1260:
            raise ValueError(f'Expected 1,260 template users, found {len(users)}')

        print('Building validation candidates from historical schedules...', flush=True)
        candidates = candidate_programmes(chain(
            reader(source_dir / 'export_arh_11-20-final.csv'),
            reader(source_dir / 'export_arh_21-30-final.csv'),
        ), VALIDATION_START, VALIDATION_END)
    # Validate uniqueness/exact IDs before any model is fitted.
    if __package__:
        from .evaluation import _cohort
    else:
        from evaluation import _cohort
    return _cohort(users), candidates


def run_evaluation(source_dir: Path, matching_dir: Path, output_dir: Path) -> dict:
    """Read inputs only; write a new result directory after evaluation succeeds."""
    source_dir, matching_dir, output_dir = map(Path, (source_dir, matching_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    started = time.perf_counter()
    users, candidates = load_protocol(source_dir, matching_dir)
    print('Fitting training-only popularity and evaluating fixed users...', flush=True)
    with ExitStack() as stack:
        def reader(path):
            return csv.DictReader(stack.enter_context(path.open('r', encoding='utf-8', newline='')))
        result = evaluate_popularity(
            reader(matching_dir / 'train-matches.csv'),
            reader(matching_dir / 'validation-matches.csv'),
            candidates, users,
        )

    statistics = {
        **result.statistics,
        'training_window': {'start_inclusive': TRAIN_START.isoformat(),
                            'end_exclusive': VALIDATION_START.isoformat()},
        'validation_window': {'start_inclusive': VALIDATION_START.isoformat(),
                              'end_exclusive': VALIDATION_END.isoformat()},
        'evaluation_seconds': round(time.perf_counter() - started, 3),
        'inputs': {'source_directory': str(source_dir.resolve()),
                   'matching_directory': str(matching_dir.resolve())},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _save_csv(output_dir / 'validation-predictions.csv', ['user_id', 'tv_show_id'],
              ({'user_id': row['user_id'], 'tv_show_id': row['tv_show_id']} for row in result.per_user))
    _save_csv(output_dir / 'per-user.csv', list(result.per_user[0]), result.per_user)
    _save_csv(output_dir / 'candidate-popularity.csv', ['tv_show_id', 'training_distinct_users'],
              result.candidate_popularity)
    (output_dir / 'evaluation.json').write_text(
        json.dumps(statistics, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(statistics['cohorts'], indent=2), flush=True)
    print(f'Saved evaluation outputs to {output_dir.resolve()}', flush=True)
    return statistics


def main():
    exam_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Evaluate training-only popularity with local MAP@5.')
    parser.add_argument('--source-dir', type=Path, default=exam_dir / 'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam_dir / 'derived' / 'matching')
    parser.add_argument('--output-dir', type=Path, default=exam_dir / 'derived' / 'popularity')
    args = parser.parse_args()
    run_evaluation(args.source_dir, args.matching_dir, args.output_dir)


if __name__ == '__main__':
    main()
