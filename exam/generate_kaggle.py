"""Generate a checked future-horizon PPR CSV on explicit invocation only."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

if __package__:
    from .als import build_interactions
    from .evaluation import _id, _integer, candidate_programmes
    from .evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from .ppr import PPRConfig, build_graph, predict_users, MASS_TOLERANCE
else:
    from als import build_interactions
    from evaluation import _id, _integer, candidate_programmes
    from evaluate_popularity import _save_csv, TRAIN_START, VALIDATION_START, VALIDATION_END
    from ppr import PPRConfig, build_graph, predict_users, MASS_TOLERANCE


EXPECTED_USERS = 1260
FUTURE_START = VALIDATION_END
FUTURE_END = datetime(2020, 10, 19)
SELECTED_CONFIG = PPRConfig(damping=0.50, tolerance=1e-10, max_iterations=200)
SUBMISSION_HEADER = ['user_id', 'tv_show_id']
MATCH_FIELDS = {'user_id', 'channel_id', 'tv_show_id', 'broadcast_start',
                'broadcast_end', 'duration_seconds', 'overlap_seconds'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(value):
    if not isinstance(value, str) or _id(value) != value:
        raise ValueError(f'Expected canonical positive decimal ID, got {value!r}')
    return value


def _submission_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != SUBMISSION_HEADER:
            raise ValueError('CSV header must be exactly user_id,tv_show_id')
        for row in reader:
            if len(row) != 2:
                raise ValueError('Every submission/template row must have exactly two fields')
            yield row


def _template_users(path):
    users = tuple(_canonical_id(row[0]) for row in _submission_rows(path))
    if len(users) != EXPECTED_USERS or len(set(users)) != EXPECTED_USERS:
        raise ValueError('Template must contain exactly 1,260 distinct users')
    return users


def validate_submission(path, users, candidates):
    """Read serialized CSV back; require exact template order and five eligible IDs."""
    users = tuple(_canonical_id(user) for user in users)
    if len(users) != EXPECTED_USERS or len(set(users)) != EXPECTED_USERS:
        raise ValueError('Expected exactly 1,260 distinct template users')
    pool = frozenset(_canonical_id(item) for item in candidates)
    seen = []
    for user, text in _submission_rows(path):
        seen.append(_canonical_id(user))
        items = tuple(_canonical_id(item) for item in text.split(' '))
        if len(items) != 5 or len(set(items)) != 5 or not set(items) <= pool:
            raise ValueError(f'User {user} needs five distinct eligible programme IDs')
    if tuple(seen) != users:
        raise ValueError('CSV users must match all 1,260 template users exactly, in template order')
    return {'users': len(seen), 'recommendations_per_user': 5, 'recommendation_slots': len(seen)*5,
            'exact_template_order': True, 'distinct_positive_eligible_ids_per_row': True}


def _timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None or result.microsecond:
        raise ValueError('Training timestamps must be naive whole-second datetimes')
    return result


def _training_metadata(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    train = record.get('partitions', {}).get('train', {})
    if (train.get('window_start_inclusive') != TRAIN_START.isoformat()
            or train.get('window_end_exclusive') != VALIDATION_START.isoformat()):
        raise ValueError('Matching metadata must identify the original March 9–May 18 training partition')
    return train


def _training_rows(path):
    """Check saved training matches, without reconstructing their raw viewing provenance."""
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not MATCH_FIELDS <= set(reader.fieldnames or ()):
            raise ValueError('Training match CSV lacks required interval/ID fields')
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(row[field] is None for field in MATCH_FIELDS):
                    raise ValueError('Malformed training row')
                for field in ('user_id', 'channel_id', 'tv_show_id'):
                    _canonical_id(row[field])
                begin, end = _timestamp(row['broadcast_start']), _timestamp(row['broadcast_end'])
                duration = _integer(row['duration_seconds'], 'duration_seconds')
                overlap = _integer(row['overlap_seconds'], 'overlap_seconds')
                available = (min(end, VALIDATION_START)-max(begin, TRAIN_START)).total_seconds()
                if duration <= 0 or (end-begin).total_seconds() != duration:
                    raise ValueError('Invalid broadcast interval/duration')
                if available <= 0 or overlap > min(duration, available) or 2*overlap < min(600, duration):
                    raise ValueError('Overlap must qualify within the original training window')
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid training match at line {line}: {error}') from error
            yield row


def run_generation(source_dir: Path, matching_dir: Path, output_dir: Path, batch_size=32) -> dict:
    source_dir, matching_dir, output_dir = map(Path, (source_dir, matching_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    started = time.perf_counter()
    inputs = {'template': source_dir/'submission.csv',
              'future_schedule': source_dir/'export_arh_31-42-final.csv',
              'training_matches': matching_dir/'train-matches.csv',
              'matching_metadata': matching_dir/'data-quality.json'}
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    users = _template_users(inputs['template'])
    metadata = _training_metadata(inputs['matching_metadata'])
    print('Building future candidates from July 27–October 19 scheduled airings...', flush=True)
    with inputs['future_schedule'].open(encoding='utf-8', newline='') as stream:
        candidates = candidate_programmes(csv.DictReader(stream), FUTURE_START, FUTURE_END)
    if len(candidates) < 5:
        raise ValueError('At least five distinct eligible future candidates are required')
    print('Building graph and popularity from the original training partition only...', flush=True)
    data = build_interactions(_training_rows(inputs['training_matches']))
    recorded_rows = metadata.get('audit', {}).get('matched_broadcasts')
    if recorded_rows is not None and _integer(recorded_rows, 'matched_broadcasts') != data.popularity.training_rows:
        raise ValueError('Training row count does not match saved matching metadata')
    graph = build_graph(data)
    ranked = predict_users(data, graph, users, candidates, SELECTED_CONFIG, batch_size,
                           progress=lambda done, total: print(f'PPR converged for {done}/{total} warm users', flush=True))
    if set(ranked.predictions) != set(users):
        raise ValueError('Predictions do not cover exactly the template cohort')
    warm_diagnostics = [ranked.diagnostics[user] for user in ranked.warm_users]
    audit = {
        'method': 'personalized_pagerank', 'parameters': asdict(SELECTED_CONFIG), 'batch_size': batch_size,
        'training_window': {'start_inclusive': TRAIN_START.isoformat(), 'end_exclusive': VALIDATION_START.isoformat()},
        'prediction_window': {'start_inclusive': FUTURE_START.isoformat(), 'end_exclusive': FUTURE_END.isoformat()},
        'training_users': len(data.user_ids), 'training_match_rows': data.popularity.training_rows,
        'candidate_programmes': len(candidates), 'cold_candidates': len(candidates-set(data.item_ids)),
        'graph': graph.diagnostics,
        'convergence': {
            'converged_users': len(ranked.warm_users), 'cold_fallback_users': len(users)-len(ranked.warm_users),
            'maximum_iterations_used': max((d['iterations'] for d in warm_diagnostics), default=0),
            'maximum_fixed_point_residual': max((d['fixed_point_residual'] for d in warm_diagnostics), default=None),
            'maximum_probability_mass_error': max((d['max_mass_error'] for d in warm_diagnostics), default=None),
            'mass_tolerance': MASS_TOLERANCE,
        },
        'policy': 'Uniform restart on distinct training programmes; training-only distinct-user popularity fallback; '
                  'repeats allowed; PPR ties by popularity then numeric ID; future schedule affects eligibility only.',
        'provenance_limit': 'Metadata and interval checks do not reconstruct or authenticate raw viewing attribution.',
        'inputs': {name: {'path': str(path.resolve()), 'sha256': input_hashes[name]} for name, path in inputs.items()},
        'source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                          ('generate_kaggle.py', 'ppr.py', 'als.py', 'evaluation.py', 'evaluate_popularity.py')},
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    }
    # Verify files did not change during this run before writing any outputs.
    if any(sha256(path) != input_hashes[name] for name, path in inputs.items()):
        raise ValueError('An input file changed during generation')
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary = output_dir/'submission.csv.tmp'
    _save_csv(temporary, SUBMISSION_HEADER,
              ({'user_id': user, 'tv_show_id': ' '.join(ranked.predictions[user])} for user in users))
    audit['csv_validation'] = validate_submission(temporary, users, candidates)
    audit['csv_sha256'] = sha256(temporary)
    _save_csv(output_dir/'ppr-diagnostics.csv', list(ranked.diagnostics[users[0]]),
              (ranked.diagnostics[user] for user in users))
    _save_csv(output_dir/'candidate-popularity.csv', ['tv_show_id', 'training_distinct_users'],
              ({'tv_show_id': item, 'training_distinct_users': data.popularity.counts.get(item, 0)}
               for item in data.popularity.rank(candidates)))
    audit['total_seconds'] = round(time.perf_counter()-started, 3)
    (output_dir/'generation-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    temporary.rename(output_dir/'submission.csv')
    print(json.dumps(audit['csv_validation'], indent=2), flush=True)
    print(f'Saved validated CSV and audit to {output_dir.resolve()}', flush=True)
    return audit


def main():
    exam_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Generate final Kaggle CSV with fixed PPR damping 0.50 and training-only fitting. Does not submit.')
    parser.add_argument('--source-dir', type=Path, default=exam_dir/'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam_dir/'derived'/'matching')
    parser.add_argument('--output-dir', type=Path, default=exam_dir/'derived'/'kaggle-ppr-050')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run_generation(args.source_dir, args.matching_dir, args.output_dir, args.batch_size)


if __name__ == '__main__':
    main()
