"""MAP@5 and distinct-training-user popularity for the agreed local protocol."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import fsum
import re
from typing import Iterable, Mapping


Identifier = str | int
Row = Mapping[str, str | int]


def _integer(value, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, (str, int))
            or re.fullmatch(r'-?[0-9]+', str(value)) is None):
        raise ValueError(f'{name} must be an exact integer, not {value!r}')
    return int(value)


def _id(value, *, allow_zero=False) -> str:
    number = _integer(value, 'ID')
    if number < (0 if allow_zero else 1):
        raise ValueError('IDs must be positive; zero is an unlabelled programme')
    return str(number)


def _field(row, name):
    try:
        return row[name]
    except KeyError:
        raise ValueError(f'Missing required field: {name}') from None


def _ids(values: Iterable[Identifier]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError('Expected an iterable of IDs, not one string')
    return [_id(value) for value in values]


def _cohort(users: Iterable[Identifier]) -> list[str]:
    normalized = _ids(users)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError('Evaluation cohort must be nonempty with unique users')
    return normalized


def _user_mapping(mapping):
    normalized = {}
    for user, value in mapping.items():
        key = _id(user)
        if key in normalized:
            raise ValueError('Duplicate user IDs after numeric normalization')
        normalized[key] = value
    return normalized


def average_precision_at_5(relevant_items: Iterable[Identifier],
                           predictions: Iterable[Identifier]) -> float:
    """Binary relevance; exactly five unique positive predictions, empty target = 0.

    Invalid predictions raise even when the target is empty. Repeated relevance
    entries are collapsed. IDs are compared exactly without floating point.
    """
    ranked = _ids(predictions)
    if len(ranked) != 5 or len(set(ranked)) != 5:
        raise ValueError('Predictions must contain exactly five distinct programme IDs')
    relevant = set(_ids(relevant_items))
    if not relevant:
        return 0.0
    hits = 0
    terms = []
    for rank, item in enumerate(ranked, 1):
        if item in relevant:
            hits += 1
            terms.append(hits / rank)
    return fsum(terms) / min(5, len(relevant))


def mean_average_precision_at_5(relevant_by_user: Mapping,
                                predictions_by_user: Mapping,
                                users: Iterable[Identifier]) -> float:
    """Macro-average over the explicit cohort, including users with empty targets.

    Missing relevance means an empty set. Predictions must cover exactly the
    cohort. Candidate eligibility is checked by the evaluation orchestration.
    """
    cohort = _cohort(users)
    targets = _user_mapping(relevant_by_user)
    predictions = _user_mapping(predictions_by_user)
    if set(predictions) != set(cohort):
        raise ValueError('Prediction users must match the evaluation cohort exactly')
    return fsum(average_precision_at_5(targets.get(user, ()), predictions[user])
                for user in cohort) / len(cohort)


@dataclass(frozen=True)
class PopularityModel:
    counts: dict[str, int]
    training_users: frozenset[str]
    training_rows: int

    def rank(self, candidates: Iterable[Identifier]) -> tuple[str, ...]:
        pool = set(_ids(candidates))
        return tuple(sorted(pool, key=lambda item: (-self.counts.get(item, 0), int(item))))

    def top_five(self, candidates: Iterable[Identifier]) -> tuple[str, ...]:
        ranked = self.rank(candidates)
        if len(ranked) < 5:
            raise ValueError('At least five distinct eligible candidates are required')
        return ranked[:5]


def fit_popularity(training_rows: Iterable[Row]) -> PopularityModel:
    """Count distinct users per programme using qualifying training matches only."""
    viewers = defaultdict(set)
    users = set()
    rows = 0
    for row in training_rows:
        user = _id(_field(row, 'user_id'))
        item = _id(_field(row, 'tv_show_id'))
        viewers[item].add(user)
        users.add(user)
        rows += 1
    return PopularityModel({item: len(group) for item, group in viewers.items()},
                           frozenset(users), rows)


def candidate_programmes(schedule_rows: Iterable[Row], start: datetime,
                         end: datetime) -> frozenset[str]:
    """Nonzero IDs with positive-length airings overlapping [start, end)."""
    for bound in (start, end):
        if not isinstance(bound, datetime) or bound.tzinfo is not None or bound.microsecond:
            raise ValueError('Candidate window must use naive whole-second datetimes')
    if start >= end:
        raise ValueError('Candidate window start must precede end')
    candidates = set()
    for row in schedule_rows:
        item = _id(_field(row, 'tv_show_id'), allow_zero=True)
        duration = _integer(_field(row, 'duration'), 'duration')
        value = _field(row, 'start_time')
        if not isinstance(value, str):
            raise ValueError('Schedule start_time must be a timestamp string')
        begin = datetime.strptime(value, '%d.%m.%Y %H:%M:%S')
        if item != '0' and duration > 0 and begin < end and begin + timedelta(seconds=duration) > start:
            candidates.add(item)
    return frozenset(candidates)


@dataclass(frozen=True)
class PopularityEvaluation:
    top_five: tuple[str, ...]
    per_user: tuple[dict, ...]
    candidate_popularity: tuple[dict, ...]
    statistics: dict


def _cohort_statistics(rows):
    count = len(rows)
    return {
        'users': count,
        'empty_target_users': sum(row['relevant_count'] == 0 for row in rows),
        'map_at_5': fsum(row['ap_at_5'] for row in rows) / count if count else None,
    }


def evaluate_popularity(training_rows: Iterable[Row], validation_rows: Iterable[Row],
                        candidates: Iterable[Identifier], users: Iterable[Identifier]) -> PopularityEvaluation:
    """Fit on training only and evaluate the global list with the shared protocol."""
    cohort = _cohort(users)
    pool = frozenset(_ids(candidates))
    model = fit_popularity(training_rows)
    ranked = model.top_five(pool)
    result = evaluate_rankings(model, validation_rows, pool, cohort,
                               {user: ranked for user in cohort}, 'distinct_training_user_popularity')
    result.statistics['top_five'] = list(ranked)
    candidate_counts = tuple({'tv_show_id': item, 'training_distinct_users': model.counts.get(item, 0)}
                             for item in model.rank(pool))
    return PopularityEvaluation(ranked, result.per_user, candidate_counts, result.statistics)


@dataclass(frozen=True)
class RankingEvaluation:
    per_user: tuple[dict, ...]
    statistics: dict


def evaluate_rankings(model: PopularityModel, validation_rows: Iterable[Row],
                      candidates: Iterable[Identifier], users: Iterable[Identifier],
                      predictions_by_user: Mapping, method: str) -> RankingEvaluation:
    """Evaluate supplied rankings on exactly the shared fixed user cohort.

    The popularity model is already fitted on training. Validation is only
    relevance, never fitting. Repeats from training remain eligible.
    A validation target outside the shared candidate pool raises rather than being
    silently discarded. Non-cohort validation users do not contribute to scores.
    """
    cohort = _cohort(users)
    cohort_set = set(cohort)
    pool = frozenset(_ids(candidates))
    predictions = _user_mapping(predictions_by_user)
    if set(predictions) != cohort_set:
        raise ValueError('Prediction users must match the evaluation cohort exactly')
    for user in cohort:
        ranked = _ids(predictions[user])
        average_precision_at_5((), ranked)  # Validate length and uniqueness even without targets.
        if not set(ranked) <= pool:
            raise ValueError(f'Predictions for user {user} contain ineligible programmes')
        predictions[user] = ranked
    relevant = defaultdict(set)
    validation_count = cohort_rows = 0
    for row in validation_rows:
        user = _id(_field(row, 'user_id'))
        item = _id(_field(row, 'tv_show_id'))
        validation_count += 1
        if user not in cohort_set:
            continue
        cohort_rows += 1
        if item not in pool:
            raise ValueError(f'Validation target {item} for user {user} is outside the candidate pool')
        relevant[user].add(item)

    per_user = []
    for user in cohort:
        target = relevant[user]
        ranked = predictions[user]
        per_user.append({
            'user_id': user,
            'has_training_history': user in model.training_users,
            'relevant_count': len(target),
            'relevant_ids': ' '.join(sorted(target, key=int)),
            'cold_item_relevant_count': sum(item not in model.counts for item in target),
            'tv_show_id': ' '.join(ranked),
            'hits_at_5': len(set(ranked) & target),
            'ap_at_5': average_precision_at_5(target, ranked),
        })
    statistics = {
        'method': method,
        'metric': 'MAP@5',
        'ground_truth': 'Local qualifying-watch proxy; not asserted equivalent to Kaggle hidden labels.',
        'empty_target_policy': 'AP@5 = 0; include in the primary cohort denominator.',
        'repeat_policy': 'Previously watched programmes allowed; five distinct IDs per list.',
        'training_match_rows': model.training_rows,
        'training_users': len(model.training_users),
        'training_programmes': len(model.counts),
        'candidate_programmes': len(pool),
        'zero_support_candidates': sum(item not in model.counts for item in pool),
        'validation_match_rows': validation_count,
        'validation_match_rows_in_cohort': cohort_rows,
        'relevant_user_item_pairs': sum(row['relevant_count'] for row in per_user),
        'cold_item_relevant_pairs': sum(row['cold_item_relevant_count'] for row in per_user),
        'cohorts': {
            'all_users': _cohort_statistics(per_user),
            'nonempty_targets': _cohort_statistics([row for row in per_user if row['relevant_count']]),
            'warm_users': _cohort_statistics([row for row in per_user if row['has_training_history']]),
            'cold_start_users': _cohort_statistics([row for row in per_user if not row['has_training_history']]),
        },
    }
    return RankingEvaluation(tuple(per_user), statistics)
