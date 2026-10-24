"""Shared seeded cutoff-tie resolutions; empirical sensitivity, not bounds."""
import hashlib
import math
import random
import statistics

if __package__:
    from .evaluation import _cohort, _id, _integer, average_precision_at_5
else:
    from evaluation import _cohort, _id, _integer, average_precision_at_5


SEEDS = tuple(range(20))
SAMPLING_POLICY = ('Seeds 0..19; per-user Python Random seeded by big-endian SHA-256 integer of '
                   'UTF-8 ppr-tie-sensitivity-v1|seed|user_id; sample without replacement from '
                   'numeric-ID-sorted fifth-frequency tie. Targets shared across all methods.')


def resolve_targets(frequencies, seed):
    """Retain all items above cutoff; randomly select only ambiguous cutoff members."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError('Use one of the 20 fixed seeds 0..19')
    targets = {}
    for user, counts in frequencies.items():
        if _id(user) != user:
            raise ValueError('Exact canonical user ID required')
        for item, count in counts.items():
            if _id(item) != item or _integer(count, 'frequency') <= 0:
                raise ValueError('Frequencies require positive counts and canonical IDs')
        counts = {item: _integer(count, 'frequency') for item, count in counts.items()}
        ordered = sorted(counts, key=lambda item: (-counts[item], int(item)))
        if len(ordered) <= 5:
            targets[user] = tuple(ordered)
            continue
        cutoff = counts[ordered[4]]
        higher = [i for i in ordered if counts[i] > cutoff]
        tied = sorted((i for i in ordered if counts[i] == cutoff), key=int)
        required = 5-len(higher)
        if len(tied) == required:
            targets[user] = tuple(ordered[:5])
        else:
            material = f'ppr-tie-sensitivity-v1|{seed}|{user}'.encode('utf-8')
            rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
            targets[user] = tuple(higher + sorted(rng.sample(tied, required), key=int))
    return targets


def score_methods(predictions, targets, users, auxiliary_users):
    """One target realization, shared across methods; keep saved recommendation order."""
    users = tuple(_cohort(users))
    if set(targets) != set(users):
        raise ValueError('Targets must cover exactly the user cohort')
    auxiliary = set(auxiliary_users)
    if not auxiliary <= set(users):
        raise ValueError('Auxiliary users must belong to the full cohort')
    comparison, per_user = [], []
    for method, rankings in predictions.items():
        if set(rankings) != set(users):
            raise ValueError('Each method must cover exactly the same cohort')
        scored = []
        for user in users:
            target, ranked = targets[user], rankings[user]
            row = {'method': method, 'user_id': user, 'at_least_five_coverage80': user in auxiliary,
                   'relevant_count': len(target), 'target_ids': ' '.join(target),
                   'tv_show_id': ' '.join(ranked), 'hits_at_5': len(set(target)&set(ranked)),
                   'ap_at_5': average_precision_at_5(target, ranked)}
            scored.append(row)
        per_user.extend(scored)
        for cohort, group in [('all_users', scored), ('at_least_five_coverage80', [r for r in scored if r['at_least_five_coverage80']])]:
            comparison.append({'method': method, 'cohort': cohort, 'users': len(group),
                               'empty_target_users': sum(r['relevant_count'] == 0 for r in group),
                               'relevant_user_item_pairs': sum(r['relevant_count'] for r in group),
                               'map_at_5': math.fsum(r['ap_at_5'] for r in group)/len(group) if group else None})
    return comparison, per_user


def analyze(predictions, frequencies, users):
    """20 paired target perturbations; do not interpret sampled min/max as exact bounds."""
    users = tuple(_cohort(users))
    if set(frequencies) != set(users) or 'ppr_050' not in predictions:
        raise ValueError('Need full frequency cohort and uniform PPR .50 reference')
    auxiliary = tuple(u for u in users if len(frequencies[u]) >= 5)
    per_seed, target_rows = [], []
    for seed in SEEDS:
        targets = resolve_targets(frequencies, seed)
        target_rows.extend({'seed': seed, 'user_id': user, 'target_ids': ' '.join(targets[user])} for user in users)
        comparison, _ = score_methods(predictions, targets, users, auxiliary)
        baseline = {r['cohort']: r['map_at_5'] for r in comparison if r['method'] == 'ppr_050'}
        per_seed.extend({'seed': seed, **r, 'paired_delta_vs_ppr_050':
                         r['map_at_5']-baseline[r['cohort']] if r['map_at_5'] is not None else None} for r in comparison)
    summary = []
    for method in predictions:
        for cohort in ('all_users', 'at_least_five_coverage80'):
            rows = [r for r in per_seed if r['method'] == method and r['cohort'] == cohort]
            maps = [r['map_at_5'] for r in rows if r['map_at_5'] is not None]
            deltas = [r['paired_delta_vs_ppr_050'] for r in rows if r['paired_delta_vs_ppr_050'] is not None]
            summary.append({'method': method, 'cohort': cohort, 'users': rows[0]['users'], 'seeds': len(SEEDS),
                            'map_mean': statistics.mean(maps) if maps else None,
                            'map_sample_sd': statistics.stdev(maps) if maps else None,
                            'map_sample_min': min(maps) if maps else None, 'map_sample_max': max(maps) if maps else None,
                            'paired_delta_mean_vs_ppr_050': statistics.mean(deltas) if deltas else None,
                            'paired_delta_sample_sd_vs_ppr_050': statistics.stdev(deltas) if deltas else None,
                            'paired_delta_sample_min_vs_ppr_050': min(deltas) if deltas else None,
                            'paired_delta_sample_max_vs_ppr_050': max(deltas) if deltas else None,
                            'paired_wins_vs_ppr_050': sum(d > 1e-12 for d in deltas) if deltas else None,
                            'paired_ties_vs_ppr_050': sum(abs(d) <= 1e-12 for d in deltas) if deltas else None,
                            'paired_losses_vs_ppr_050': sum(d < -1e-12 for d in deltas) if deltas else None})
    return {'per_seed': per_seed, 'summary': summary, 'target_rows': target_rows}
