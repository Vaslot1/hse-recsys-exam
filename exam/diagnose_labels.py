"""Compare relevance definitions using frozen validation predictions; no fitting."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

if __package__:
    from .evaluation import _cohort, _id, _integer, average_precision_at_5
else:
    from evaluation import _cohort, _id, _integer, average_precision_at_5


TRAIN_START = datetime(2020, 3, 9)
START = datetime(2020, 5, 18)
END = datetime(2020, 7, 27)
LABELS = ('current', 'coverage80_all', 'coverage80_top5')
MATCH_FIELDS = {'user_id', 'channel_id', 'tv_show_id', 'broadcast_start',
                'broadcast_end', 'duration_seconds', 'overlap_seconds'}
RUNS = (
    ('popularity', 'popularity', 'distinct_training_user_popularity', None),
    ('als_tuned', 'als-tuning/als-06', 'implicit_als',
     {'factors': 32, 'iterations': 15, 'alpha': 1., 'regularization': 10., 'seed': 42, 'confidence_weighting': 'binary'}),
    ('ppr_085', 'ppr', 'personalized_pagerank', {'damping': .85, 'tolerance': 1e-10, 'max_iterations': 200}),
    ('ppr_050', 'ppr-damping-050', 'personalized_pagerank', {'damping': .50, 'tolerance': 1e-10, 'max_iterations': 200}),
)
POLICIES = {
    'current': 'All distinct programmes in saved qualifying validation broadcast matches.',
    'coverage80_all': 'All distinct programmes with at least one broadcast satisfying 5*overlap >= 4*full_duration.',
    'coverage80_top5': 'Up to five programmes sorted by descending number of 80%-qualified broadcasts, then ascending numeric ID.',
    'fragmented_views': 'Use saved union overlap per user/channel/broadcast. Disjoint visits to one airing accumulate; overlapping/duplicate views do not multiply seconds. One qualifying airing contributes one frequency count.',
    'broadcast_identity': 'user_id, channel_id, tv_show_id, broadcast_start, broadcast_end. Repeated airings count separately, including different channels. Duplicate saved keys are rejected, never summed.',
    'coverage_denominator': 'Full broadcast duration, including for boundary-crossing airings; numerator is union overlap clipped to validation. Saved quarantine and schedule-conflict exclusions remain in force.',
    'frequency_ties': 'Deterministic ascending exact numeric programme ID. Report any frequency tie and separately ties straddling ranks five/six. This tie-break is a local assumption.',
    'auxiliary_cohort': 'One fixed cohort of users with at least five distinct coverage80_all programmes, reused for every method and label definition.',
    'empty_targets': 'AP@5=0; include in all 1260 users. Never pad targets. With 1-4 targets divide AP by target count; with >=5 divide by 5. An empty auxiliary cohort has MAP=null.',
    'metric': 'Existing binary MAP@5; target frequency determines membership, not graded relevance or exact-position matching.',
    'limitations': 'Diagnostic proxy for the user-supplied competition description. Hidden fragmentation/count/tie conventions are unverified; this does not reproduce hidden labels. No new rankings, model fitting, validation-profile updates or Kaggle actions.',
}


def _canonical(value):
    if not isinstance(value, str) or _id(value) != value:
        raise ValueError(f'Expected canonical positive decimal ID: {value!r}')
    return value


def _rows(path, required, exact=False):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(required) <= set(header) or (exact and header != list(required)):
            raise ValueError(f'Invalid CSV header in {path}')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV row in {path}')
            yield row


def _date(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None or result.microsecond:
        raise ValueError('Expected naive whole-second timestamp')
    return result


@dataclass(frozen=True)
class LabelData:
    users: tuple
    targets: dict
    frequencies: dict
    profiles: dict
    auxiliary_users: tuple
    counts: dict


def build_labels(match_rows, users, candidates) -> LabelData:
    users = tuple(_cohort(users))
    pool = frozenset(_canonical(item) for item in candidates)
    current = {user: set() for user in users}
    frequency = {user: Counter() for user in users}
    seen = set()
    counts = {'match_rows': 0, 'cohort_match_rows': 0, 'coverage80_match_rows': 0, 'cohort_coverage80_match_rows': 0}
    for number, row in enumerate(match_rows, 2):
        try:
            user, channel, item = (_canonical(row[k]) for k in ('user_id', 'channel_id', 'tv_show_id'))
            begin, end = _date(row['broadcast_start']), _date(row['broadcast_end'])
            duration, overlap = (_integer(row[k], k) for k in ('duration_seconds', 'overlap_seconds'))
            if duration <= 0 or (end-begin).total_seconds() != duration:
                raise ValueError('Invalid broadcast duration')
            available = (min(end, END)-max(begin, START)).total_seconds()
            if available <= 0 or overlap > min(duration, available) or 2*overlap < min(600, duration):
                raise ValueError('Saved overlap must qualify within the original validation window')
            key = (user, channel, item, begin, end)
            if key in seen:
                raise ValueError('Duplicate saved user/broadcast match; fragments must already be unioned')
            seen.add(key)
            high = 5*overlap >= 4*duration
            counts['match_rows'] += 1
            counts['coverage80_match_rows'] += high
            if user in current:
                if item not in pool:
                    raise ValueError('Validation target outside the saved candidate pool')
                counts['cohort_match_rows'] += 1
                current[user].add(item)
                if high:
                    counts['cohort_coverage80_match_rows'] += 1
                    frequency[user][item] += 1
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f'Invalid validation match at line {number}: {error}') from error
    profiles, top = {}, {}
    for user in users:
        ordered = sorted(frequency[user], key=lambda item: (-frequency[user][item], int(item)))
        top[user] = tuple(ordered[:5])
        cutoff = frequency[user][ordered[4]] if len(ordered) >= 5 else None
        crossing = len(ordered) > 5 and cutoff == frequency[user][ordered[5]]
        profiles[user] = {
            'user_id': user, 'current_programmes': len(current[user]), 'coverage80_programmes': len(ordered),
            'coverage80_broadcasts': sum(frequency[user].values()), 'at_least_five_coverage80': len(ordered) >= 5,
            'any_frequency_tie': len(set(frequency[user].values())) < len(ordered),
            'cutoff_tie': crossing,
            'cutoff_frequency': cutoff,
            'cutoff_tie_size': sum(n == cutoff for n in frequency[user].values()) if crossing else 0,
            'cutoff_tied_ids': ' '.join(item for item in ordered if crossing and frequency[user][item] == cutoff),
        }
    targets = {'current': {u: frozenset(items) for u, items in current.items()},
               'coverage80_all': {u: frozenset(items) for u, items in frequency.items()}, 'coverage80_top5': top}
    return LabelData(users, targets, frequency, profiles,
                     tuple(u for u in users if profiles[u]['at_least_five_coverage80']), counts)


def score_predictions(method, predictions, labels):
    if set(predictions) != set(labels.users):
        raise ValueError('Predictions must cover exactly the template users')
    table, per_user = [], []
    auxiliary = set(labels.auxiliary_users)
    for name in LABELS:
        scored = []
        for user in labels.users:
            target, ranked = labels.targets[name][user], predictions[user]
            row = {'method': method, 'label_definition': name, 'user_id': user,
                   'at_least_five_coverage80': user in auxiliary, 'relevant_count': len(target),
                   'tv_show_id': ' '.join(ranked), 'hits_at_5': len(set(ranked) & set(target)),
                   'ap_at_5': average_precision_at_5(target, ranked)}
            scored.append(row)
        per_user.extend(scored)
        for cohort, group in [('all_users', scored), ('at_least_five_coverage80', [r for r in scored if r['at_least_five_coverage80']])]:
            table.append({'method': method, 'label_definition': name, 'cohort': cohort, 'users': len(group),
                          'empty_target_users': sum(r['relevant_count'] == 0 for r in group),
                          'relevant_user_item_pairs': sum(r['relevant_count'] for r in group),
                          'map_at_5': math.fsum(r['ap_at_5'] for r in group)/len(group) if group else None})
    return table, per_user


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _close(actual, expected, description):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'{description} differs from saved result: {actual} versus {expected}')


def _user_rows(path, fields, users, exact=False):
    data = list(_rows(path, fields, exact))
    if [r['user_id'] for r in data] != list(users):
        raise ValueError(f'Expected exact 1,260 template users in order in {path}')
    return data


def _load_run(directory, method, parameters, users):
    stats = json.loads((directory/'evaluation.json').read_text(encoding='utf-8'))
    for key, begin, end in [('training_window', TRAIN_START, START), ('validation_window', START, END)]:
        if stats.get(key) != {'start_inclusive': begin.isoformat(), 'end_exclusive': end.isoformat()}:
            raise ValueError(f'Wrong saved {key} in {directory}')
    if stats.get('method') != method or stats.get('metric') != 'MAP@5' or stats.get('parameters') != parameters:
        raise ValueError(f'Wrong saved model configuration in {directory}')
    candidate_rows = list(_rows(directory/'candidate-popularity.csv', ['tv_show_id', 'training_distinct_users'], True))
    candidates = {}
    for row in candidate_rows:
        item = _canonical(row['tv_show_id'])
        support = _integer(row['training_distinct_users'], 'training_distinct_users')
        if item in candidates or support < 0:
            raise ValueError(f'Invalid candidate support in {directory}')
        candidates[item] = support
    if len(candidates) < 5 or len(candidates) != stats['candidate_programmes']:
        raise ValueError(f'Incorrect saved candidate count in {directory}')
    predictions = {}
    for row in _user_rows(directory/'validation-predictions.csv', ['user_id', 'tv_show_id'], users, True):
        ranked = tuple(_canonical(item) for item in row['tv_show_id'].split(' '))
        average_precision_at_5((), ranked)
        if not set(ranked) <= candidates.keys():
            raise ValueError(f'Ineligible saved prediction in {directory}')
        predictions[row['user_id']] = ranked
    per = _user_rows(directory/'per-user.csv', ['user_id', 'tv_show_id', 'relevant_ids', 'relevant_count', 'hits_at_5', 'ap_at_5'], users)
    return stats, candidates, predictions, per


def _check_saved(stats, per, predictions, labels, summary):
    for row in per:
        user = row['user_id']
        original = row['relevant_ids'].split(' ') if row['relevant_ids'] else []
        target = labels.targets['current'][user]
        if (len(original) != len(set(original)) or any(_canonical(i) != i for i in original)
                or set(original) != target or _integer(row['relevant_count'], 'relevant_count') != len(target)):
            raise ValueError(f'Current labels differ from saved labels for user {user}')
        if row['tv_show_id'] != ' '.join(predictions[user]):
            raise ValueError(f'Prediction file differs from saved per-user ranking for user {user}')
        if _integer(row['hits_at_5'], 'hits_at_5') != len(target & set(predictions[user])):
            raise ValueError(f'Saved hit count differs for user {user}')
        _close(average_precision_at_5(target, predictions[user]), row['ap_at_5'], f'User {user} AP@5')
    saved = stats['cohorts']['all_users']
    if saved['users'] != summary['users'] or saved['empty_target_users'] != summary['empty_target_users']:
        raise ValueError('Saved cohort counts differ from reconstructed current labels')
    _close(summary['map_at_5'], saved['map_at_5'], 'Full-cohort MAP@5')
    for name, actual in [('validation_match_rows', labels.counts['match_rows']),
                         ('validation_match_rows_in_cohort', labels.counts['cohort_match_rows']),
                         ('relevant_user_item_pairs', summary['relevant_user_item_pairs'])]:
        if stats[name] != actual:
            raise ValueError(f'Saved {name} differs from input matches')


def _save(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(source_dir, matching_dir, runs_dir, output_dir):
    source_dir, matching_dir, runs_dir, output_dir = map(Path, (source_dir, matching_dir, runs_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    started = time.perf_counter()
    files = [source_dir/'submission.csv', matching_dir/'data-quality.json', matching_dir/'validation-matches.csv']
    for _, folder, _, _ in RUNS:
        files.extend(runs_dir/folder/name for name in ('evaluation.json', 'validation-predictions.csv', 'candidate-popularity.csv', 'per-user.csv'))
    hashes = {path: _hash(path) for path in files}
    users = tuple(_canonical(r['user_id']) for r in _rows(files[0], ['user_id','tv_show_id'], True))
    if len(users) != 1260 or len(set(users)) != 1260:
        raise ValueError('Template must have exactly 1,260 distinct users')
    metadata = json.loads(files[1].read_text(encoding='utf-8'))
    for name, begin, end in [('train', TRAIN_START, START), ('validation', START, END)]:
        partition = metadata.get('partitions', {}).get(name, {})
        if (partition.get('window_start_inclusive') != begin.isoformat()
                or partition.get('window_end_exclusive') != end.isoformat()):
            raise ValueError(f'Matching metadata does not identify the original {name} window')
    saved = {name: _load_run(runs_dir/folder, method, params, users) for name, folder, method, params in RUNS}
    pool = saved['popularity'][1]
    if any(run[1] != pool for run in saved.values()):
        raise ValueError('Saved methods do not share the same candidate IDs and training support')
    print('Constructing three label definitions from saved validation broadcast matches...', flush=True)
    labels = build_labels(_rows(files[2], MATCH_FIELDS), users, pool)
    recorded_rows = metadata['partitions']['validation'].get('audit', {}).get('matched_broadcasts')
    if recorded_rows is not None and _integer(recorded_rows, 'matched_broadcasts') != labels.counts['match_rows']:
        raise ValueError('Matching metadata row count differs from saved validation matches')
    comparison, per_user = [], []
    for name, (stats, _, predictions, old_per) in saved.items():
        print(f'Checking frozen {name} predictions and reproducing saved current-label scores...', flush=True)
        table, scored = score_predictions(name, predictions, labels)
        _check_saved(stats, old_per, predictions, labels, table[0])
        comparison.extend(table)
        per_user.extend(scored)
    audit = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python_version': platform.python_version(),
        'current_labels_reproduce_saved_results': True,
        'training_window': {'start_inclusive': TRAIN_START.isoformat(), 'end_exclusive': START.isoformat()},
        'validation_window': {'start_inclusive': START.isoformat(), 'end_exclusive': END.isoformat()},
        'cohorts': {'all_users': len(users), 'at_least_five_coverage80': len(labels.auxiliary_users)},
        'labels': {name: {'empty_users': sum(not labels.targets[name][u] for u in users),
                          'users_with_one_to_four_targets': sum(0 < len(labels.targets[name][u]) < 5 for u in users),
                          'relevant_pairs': sum(len(labels.targets[name][u]) for u in users)} for name in LABELS},
        'frequency_ties': {'users_with_any_tie': sum(p['any_frequency_tie'] for p in labels.profiles.values()),
                           'users_with_tie_crossing_fifth_place': sum(p['cutoff_tie'] for p in labels.profiles.values())},
        'candidate_programmes': len(pool), 'match_counts': labels.counts, 'policies': POLICIES,
        'models': {name: {'directory': str((runs_dir/folder).resolve()), 'parameters': params} for name, folder, _, params in RUNS},
        'inputs': [{'path': str(path.resolve()), 'sha256': value} for path, value in hashes.items()],
        'source_sha256': {name: _hash(Path(__file__).with_name(name)) for name in ('diagnose_labels.py','evaluation.py')},
        'elapsed_seconds': round(time.perf_counter()-started, 3),
    }
    if any(_hash(path) != value for path, value in hashes.items()):
        raise ValueError('Input changed during diagnostic; no results saved')
    audit['input_hashes_unchanged'] = True
    output_dir.mkdir(parents=True, exist_ok=False)
    _save(output_dir/'comparison.csv', list(comparison[0]), comparison)
    _save(output_dir/'per-user.csv', list(per_user[0]), per_user)
    label_rows = []
    for user in users:
        label_rows.append({**labels.profiles[user],
                           'current_ids': ' '.join(sorted(labels.targets['current'][user], key=int)),
                           'coverage80_ids': ' '.join(sorted(labels.targets['coverage80_all'][user], key=int)),
                           'top5_ids': ' '.join(labels.targets['coverage80_top5'][user])})
    _save(output_dir/'labels.csv', list(label_rows[0]), label_rows)
    _save(output_dir/'qualifying-frequencies.csv', ['user_id','tv_show_id','qualified_broadcasts'],
          ({'user_id': user, 'tv_show_id': item, 'qualified_broadcasts': labels.frequencies[user][item]}
           for user in users for item in sorted(labels.frequencies[user], key=lambda i: (-labels.frequencies[user][i], int(i)))))
    (output_dir/'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    (output_dir/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(comparison, indent=2), flush=True)
    print(f'Saved diagnostic outputs to {output_dir.resolve()}', flush=True)
    return audit


def main():
    exam = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Diagnose three relevance definitions using four frozen validation runs; no training.')
    parser.add_argument('--source-dir', type=Path, default=exam/'dataset-sweet-tv')
    parser.add_argument('--matching-dir', type=Path, default=exam/'derived'/'matching')
    parser.add_argument('--runs-dir', type=Path, default=exam/'derived')
    parser.add_argument('--output-dir', type=Path, default=exam/'derived'/'label-diagnostic')
    args = parser.parse_args()
    run_diagnostic(args.source_dir, args.matching_dir, args.runs_dir, args.output_dir)


if __name__ == '__main__':
    main()
