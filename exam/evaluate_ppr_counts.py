"""Count-restart PPR on the saved graph and frozen targets; run only explicitly."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import sparse

if __package__:
    from .als import build_interactions
    from .diagnose_labels import RUNS, _rows, _canonical, _load_run, _close, _hash, _save
    from .evaluation import _integer
    from .generate_kaggle import _training_metadata, _training_rows
    from .ppr import CoViewGraph, PPRConfig, predict_users, transition_matrix, _nonnegative_square, MASS_TOLERANCE
    from .tie_sensitivity import SEEDS, SAMPLING_POLICY, analyze, score_methods
else:
    from als import build_interactions
    from diagnose_labels import RUNS, _rows, _canonical, _load_run, _close, _hash, _save
    from evaluation import _integer
    from generate_kaggle import _training_metadata, _training_rows
    from ppr import CoViewGraph, PPRConfig, predict_users, transition_matrix, _nonnegative_square, MASS_TOLERANCE
    from tie_sensitivity import SEEDS, SAMPLING_POLICY, analyze, score_methods


CONFIG = PPRConfig(damping=.50, tolerance=1e-10, max_iterations=200)
TRAIN_WINDOW = {'start_inclusive': '2020-03-09T00:00:00', 'end_exclusive': '2020-05-18T00:00:00'}
VALIDATION_WINDOW = {'start_inclusive': '2020-05-18T00:00:00', 'end_exclusive': '2020-07-27T00:00:00'}
SENSITIVITY_NOTE = ('Empirical sensitivity to 20 seeded cutoff-tie resolutions, shared across methods. '
                    'Sample min/max are not exact bounds; sample SD is not a confidence interval. '
                    'No claim to reproduce the hidden Kaggle tie-break; no seed or resolution is selected for performance.')


def _unique_training_rows(path):
    seen = set()
    for row in _training_rows(path):
        key = (row['user_id'], row['channel_id'], row['tv_show_id'],
               datetime.fromisoformat(row['broadcast_start']), datetime.fromisoformat(row['broadcast_end']))
        if key in seen:
            raise ValueError('Duplicate saved training broadcast would inflate restart counts')
        seen.add(key)
        yield row


def load_graph(directory, data):
    """Load the exact saved matrices; only check normalization, never fit new edges."""
    record = json.loads((directory/'graph.json').read_text(encoding='utf-8'))
    if record.get('parameters') != asdict(CONFIG) or record.get('training_window') != TRAIN_WINDOW:
        raise ValueError('Saved graph must be the original PPR .50 training configuration')
    items = list(_rows(directory/'graph-items.csv', ['node_index','tv_show_id','training_distinct_users','weighted_degree','dangling']))
    ids = tuple(_canonical(r['tv_show_id']) for r in items)
    if ids != data.item_ids or [int(r['node_index']) for r in items] != list(range(len(ids))):
        raise ValueError('Saved graph node order differs from original training items')
    if any(int(r['training_distinct_users']) != data.popularity.counts[r['tv_show_id']] for r in items):
        raise ValueError('Graph support differs from training snapshot')
    for name, value in [('nodes', len(ids)), ('training_users', len(data.user_ids)),
                        ('training_match_rows', data.popularity.training_rows),
                        ('training_user_item_pairs', sum(len(i) for i, _ in data.by_user))]:
        if record.get(name) != value:
            raise ValueError(f'Saved graph {name} differs from training snapshot')
    weights = _nonnegative_square(sparse.load_npz(directory/'graph.npz'))
    transition = _nonnegative_square(sparse.load_npz(directory/'transition.npz'))
    if weights.shape != (len(ids), len(ids)) or transition.shape != weights.shape:
        raise ValueError('Saved graph dimensions do not match nodes')
    if np.any(weights.diagonal() != 0) or (weights != weights.T).nnz or np.any(weights.data != np.floor(weights.data)):
        raise ValueError('Saved graph must have symmetric integer co-view counts and no self-loops')
    expected = transition_matrix(weights)
    difference = transition-expected
    if float(np.max(np.abs(difference.data), initial=0)) > 1e-14:
        raise ValueError('Saved transition does not normalize the unchanged co-view graph')
    degree = np.asarray(weights.sum(axis=1)).ravel()
    if any(float(r['weighted_degree']) != degree[i] or r['dangling'] != str(bool(degree[i] == 0)) for i, r in enumerate(items)):
        raise ValueError('Saved graph node diagnostics disagree with matrices')
    if weights.nnz != record['directed_nonzeros'] or weights.nnz//2 != record['undirected_edges'] or int((degree == 0).sum()) != record['dangling_nodes']:
        raise ValueError('Saved graph size diagnostics disagree with matrices')
    return CoViewGraph(ids, weights, transition, record)


def load_targets(directory, users, candidates, audit):
    """Use saved diagnostic targets; check the frequency table without rematching logs."""
    frequencies = {u: {} for u in users}
    for row in _rows(directory/'qualifying-frequencies.csv', ['user_id','tv_show_id','qualified_broadcasts'], True):
        u, item = _canonical(row['user_id']), _canonical(row['tv_show_id'])
        n = _integer(row['qualified_broadcasts'], 'qualified_broadcasts')
        if u not in frequencies or item not in candidates or item in frequencies[u] or n <= 0:
            raise ValueError('Invalid frozen qualifying-frequency entry')
        frequencies[u][item] = n
    rows = list(_rows(directory/'labels.csv', ['user_id','coverage80_programmes','coverage80_broadcasts',
                                              'at_least_five_coverage80','cutoff_tie','coverage80_ids','top5_ids']))
    if [r['user_id'] for r in rows] != list(users):
        raise ValueError('Frozen labels must match exact template users in order')
    targets = {}
    for row in rows:
        user = row['user_id']
        ordered = sorted(frequencies[user], key=lambda i: (-frequencies[user][i], int(i)))
        top = tuple(row['top5_ids'].split(' ')) if row['top5_ids'] else ()
        high = row['coverage80_ids'].split(' ') if row['coverage80_ids'] else []
        if (top != tuple(ordered[:5]) or high != sorted(ordered, key=int)
                or int(row['coverage80_programmes']) != len(ordered)
                or int(row['coverage80_broadcasts']) != sum(frequencies[user].values())
                or row['at_least_five_coverage80'] != str(len(ordered) >= 5)):
            raise ValueError(f'Frozen target/frequency/cohort inconsistency for user {user}')
        crossing = len(ordered) > 5 and frequencies[user][ordered[4]] == frequencies[user][ordered[5]]
        if row['cutoff_tie'] != str(crossing):
            raise ValueError('Frozen cutoff-tie flag differs from frequencies')
        targets[user] = top
    auxiliary = tuple(u for u in users if len(frequencies[u]) >= 5)
    if audit['cohorts'] != {'all_users': len(users), 'at_least_five_coverage80': len(auxiliary)}:
        raise ValueError('Frozen diagnostic cohort sizes differ')
    profile = {'empty_users': sum(not targets[u] for u in users),
               'users_with_one_to_four_targets': sum(0 < len(targets[u]) < 5 for u in users),
               'relevant_pairs': sum(len(targets[u]) for u in users)}
    if profile != audit['labels']['coverage80_top5']:
        raise ValueError('Frozen target cardinalities differ from diagnostic audit')
    return targets, frequencies, auxiliary


def run_experiment(source_dir, matching_dir, runs_dir, diagnostic_dir, output_dir):
    source_dir, matching_dir, runs_dir, diagnostic_dir, output_dir = map(Path, (source_dir, matching_dir, runs_dir, diagnostic_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_dir}')
    started = time.perf_counter()
    diagnostic_audit = json.loads((diagnostic_dir/'audit.json').read_text(encoding='utf-8'))
    if (diagnostic_audit.get('training_window') != TRAIN_WINDOW or diagnostic_audit.get('validation_window') != VALIDATION_WINDOW
            or not diagnostic_audit.get('current_labels_reproduce_saved_results') or not diagnostic_audit.get('input_hashes_unchanged')):
        raise ValueError('Expected successful diagnostic using the unchanged split')
    # Verify the exact old inputs identified by the diagnostic, not arbitrary paths from the manifest.
    old_inputs = [source_dir/'submission.csv', matching_dir/'data-quality.json', matching_dir/'validation-matches.csv']
    for _, folder, _, _ in RUNS:
        old_inputs.extend(runs_dir/folder/name for name in ('evaluation.json','validation-predictions.csv','candidate-popularity.csv','per-user.csv'))
    manifest = {Path(r['path']).resolve(): r['sha256'] for r in diagnostic_audit['inputs']}
    if set(manifest) != {p.resolve() for p in old_inputs}:
        raise ValueError('Diagnostic input paths differ from requested frozen run inputs')
    graph_dir = runs_dir/'ppr-damping-050'
    inputs = old_inputs + [matching_dir/'train-matches.csv']
    inputs.extend(diagnostic_dir/name for name in ('audit.json','comparison.json','labels.csv','qualifying-frequencies.csv'))
    inputs.extend(graph_dir/name for name in ('graph.json','graph-items.csv','graph.npz','transition.npz'))
    hashes = {p: _hash(p) for p in inputs}
    if any(hashes[p] != manifest[p.resolve()] for p in old_inputs):
        raise ValueError('An original diagnostic input has changed; refusing to mix snapshots')
    users = tuple(_canonical(r['user_id']) for r in _rows(source_dir/'submission.csv', ['user_id','tv_show_id'], True))
    if len(users) != 1260 or len(set(users)) != 1260:
        raise ValueError('Expected 1,260 distinct exact users')
    saved = {name: _load_run(runs_dir/folder, method, params, users) for name, folder, method, params in RUNS}
    candidates = saved['popularity'][1]
    if any(run[1] != candidates for run in saved.values()):
        raise ValueError('Saved comparator candidates/support differ')
    predictions = {name: run[2] for name, run in saved.items()}
    targets, frequencies, auxiliary = load_targets(diagnostic_dir, users, candidates, diagnostic_audit)
    old_comparison, _ = score_methods(predictions, targets, users, auxiliary)
    frozen = json.loads((diagnostic_dir/'comparison.json').read_text(encoding='utf-8'))
    frozen_top = {(r['method'],r['cohort']):r for r in frozen if r['label_definition'] == 'coverage80_top5'}
    if set(frozen_top) != {(r['method'],r['cohort']) for r in old_comparison}:
        raise ValueError('Frozen top-five comparator table is incomplete')
    for row in old_comparison:
        prior = frozen_top[row['method'],row['cohort']]
        for key in ('users','empty_target_users','relevant_user_item_pairs'):
            if row[key] != prior[key]:
                raise ValueError(f'Frozen comparator {key} differs')
        if row['map_at_5'] is None:
            if prior['map_at_5'] is not None:
                raise ValueError('Frozen empty-cohort metric must be null')
        else:
            _close(row['map_at_5'], prior['map_at_5'], 'Frozen comparator MAP')
    print('Aggregating original training watch counts; loading unchanged saved PPR graph...', flush=True)
    metadata = _training_metadata(matching_dir/'data-quality.json')
    data = build_interactions(_unique_training_rows(matching_dir/'train-matches.csv'))
    recorded = metadata.get('audit', {}).get('matched_broadcasts')
    if recorded is not None and recorded != data.popularity.training_rows:
        raise ValueError('Training row count differs from matching audit')
    if any(data.popularity.counts.get(i, 0) != n for i, n in candidates.items()):
        raise ValueError('Training popularity differs from saved fallback support')
    graph = load_graph(graph_dir, data)
    print('Computing PPR with normalized training-count restart, damping 0.50...', flush=True)
    ranked = predict_users(data, graph, users, candidates, CONFIG, batch_size=32, restart_weighting='watch_counts',
                           progress=lambda done,total: print(f'Count-restart PPR converged for {done}/{total} warm users', flush=True))
    predictions['ppr_counts_050'] = ranked.predictions
    comparison, per_user = score_methods(predictions, targets, users, auxiliary)
    print('Comparing five methods under 20 shared seeded cutoff-tie resolutions...', flush=True)
    sensitivity = analyze(predictions, frequencies, users)
    warm = [ranked.diagnostics[u] for u in ranked.warm_users]
    audit = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'parameters': asdict(CONFIG), 'batch_size': 32,
        'restart_weighting': 'watch_counts',
        'restart_definition': 's_ui = number of original qualifying training broadcasts for (u,i) / total qualifying broadcasts for u. No 80% filter applied to training.',
        'training_watch_threshold': 'O >= min(300 seconds, 0.5 * full broadcast duration); saved original union/conflict/quarantine conventions retained.',
        'training_window': TRAIN_WINDOW, 'validation_window': VALIDATION_WINDOW,
        'saved_graph_reused': True, 'graph_directory': str(graph_dir.resolve()),
        'graph_nodes': len(graph.item_ids), 'graph_directed_edges': graph.weights.nnz,
        'training_users': len(data.user_ids), 'training_match_rows': data.popularity.training_rows,
        'frozen_comparator_scores_reproduced': True, 'canonical_targets': 'Saved coverage80_top5 diagnostic targets with numeric-ID frequency ties.',
        'cohorts': {'all_users': len(users), 'at_least_five_coverage80': len(auxiliary)},
        'candidate_programmes': len(candidates), 'seeds': list(SEEDS), 'sampling_policy': SAMPLING_POLICY,
        'sensitivity_interpretation': SENSITIVITY_NOTE,
        'convergence': {'converged_users': len(warm), 'cold_fallback_users': len(users)-len(warm),
                        'maximum_iterations_used': max((r['iterations'] for r in warm), default=0),
                        'maximum_fixed_point_residual': max((r['fixed_point_residual'] for r in warm), default=None),
                        'maximum_probability_mass_error': max((r['max_mass_error'] for r in warm), default=None),
                        'mass_tolerance': MASS_TOLERANCE},
        'inputs': [{'path': str(p.resolve()), 'sha256': h} for p,h in hashes.items()],
        'source_sha256': {name:_hash(Path(__file__).with_name(name)) for name in
                          ('evaluate_ppr_counts.py','tie_sensitivity.py','ppr.py','als.py','diagnose_labels.py','evaluation.py','generate_kaggle.py')},
        'provenance_limit': 'Hashes and consistency checks bind this run to files now present, not to unauthenticated raw viewing attribution or an independently reconstructed historical graph.',
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'elapsed_seconds': round(time.perf_counter()-started,3),
    }
    if any(_hash(p) != h for p,h in hashes.items()):
        raise ValueError('An input changed during the experiment; no output saved')
    audit['input_hashes_unchanged'] = True
    output_dir.mkdir(parents=True, exist_ok=False)
    _save(output_dir/'validation-predictions.csv', ['user_id','tv_show_id'],
          ({'user_id':u,'tv_show_id':' '.join(ranked.predictions[u])} for u in users))
    _save(output_dir/'ppr-diagnostics.csv', list(ranked.diagnostics[users[0]]), (ranked.diagnostics[u] for u in users))
    np.savez_compressed(output_dir/'ppr-scores.npz', user_ids=np.asarray(ranked.warm_users,dtype=str),
                        item_ids=np.asarray(graph.item_ids,dtype=str), scores=ranked.scores)
    for filename, rows in [('comparison.csv',comparison), ('per-user.csv',per_user),
                            ('tie-sensitivity.csv',sensitivity['per_seed']), ('tie-sensitivity-summary.csv',sensitivity['summary']),
                            ('tie-targets.csv',sensitivity['target_rows'])]:
        _save(output_dir/filename, list(rows[0]), rows)
    evaluation = {'method':'ppr_counts_050','metric':'MAP@5','label_definition':'frozen_coverage80_top5',
                  'parameters':asdict(CONFIG),'restart_weighting':'watch_counts',
                  'cohorts':{r['cohort']:{k:v for k,v in r.items() if k not in ('method','cohort')}
                             for r in comparison if r['method']=='ppr_counts_050'}}
    for filename, value in [('audit.json',audit),('evaluation.json',evaluation),('comparison.json',comparison),
                             ('tie-sensitivity-summary.json',{'interpretation':SENSITIVITY_NOTE,'results':sensitivity['summary']})]:
        (output_dir/filename).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(evaluation,indent=2),flush=True)
    print(f'Saved count-restart results and empirical tie sensitivity to {output_dir.resolve()}',flush=True)
    return audit


def main():
    exam = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Evaluate PPR .50 with training-count restart on saved graph and frozen labels; 20 shared seeded tie resolutions.')
    parser.add_argument('--source-dir',type=Path,default=exam/'dataset-sweet-tv')
    parser.add_argument('--matching-dir',type=Path,default=exam/'derived'/'matching')
    parser.add_argument('--runs-dir',type=Path,default=exam/'derived')
    parser.add_argument('--diagnostic-dir',type=Path,default=exam/'derived'/'label-diagnostic')
    parser.add_argument('--output-dir',type=Path,default=exam/'derived'/'ppr-count-restart-050')
    args=parser.parse_args()
    run_experiment(args.source_dir,args.matching_dir,args.runs_dir,args.diagnostic_dir,args.output_dir)


if __name__=='__main__':main()
