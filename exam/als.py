"""Implicit weighted ALS with exact float64 solves and sparse observed pairs.

Objective: sum c_ui (p_ui - x_u.T y_i)^2 + lambda (||X||^2 + ||Y||^2).
p_ui = 1[n_ui > 0]; c_ui = 1 + alpha * w(n_ui).
w(n) is log1p(n) by default, or binary 1[n > 0].
Unobserved pairs have p=0, c=1. Only training users/items get factors.
"""
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

if __package__:
    from .evaluation import PopularityModel, _field, _id, _ids
else:
    from evaluation import PopularityModel, _field, _id, _ids


@dataclass(frozen=True)
class ALSConfig:
    factors: int = 32
    iterations: int = 15
    alpha: float = 20.0
    regularization: float = 0.1
    seed: int = 42
    confidence_weighting: str = 'log'

    def __post_init__(self):
        for name, minimum in [('factors', 1), ('iterations', 1), ('seed', 0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('alpha', 'regularization'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.alpha < 0 or self.regularization <= 0:
            raise ValueError('alpha must be nonnegative and regularization positive')
        if self.confidence_weighting not in ('log', 'binary'):
            raise ValueError('confidence_weighting must be log or binary')


@dataclass(frozen=True)
class Interactions:
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    by_user: tuple
    by_item: tuple
    popularity: PopularityModel


def build_interactions(training_rows) -> Interactions:
    """One saved qualifying broadcast row adds one count; exact IDs stay strings."""
    pairs = Counter()
    for row in training_rows:
        pairs[(_id(_field(row, 'user_id')), _id(_field(row, 'tv_show_id')))] += 1
    users = tuple(sorted({u for u, _ in pairs}, key=int))
    items = tuple(sorted({i for _, i in pairs}, key=int))
    user_index = {u: k for k, u in enumerate(users)}
    item_index = {i: k for k, i in enumerate(items)}
    by_user = [[] for _ in users]
    by_item = [[] for _ in items]
    popularity = dict.fromkeys(items, 0)
    for (user, item), count in pairs.items():
        u, i = user_index[user], item_index[item]
        by_user[u].append((i, count))
        by_item[i].append((u, count))
        popularity[item] += 1

    def freeze(groups):
        return tuple((np.array([i for i, _ in sorted(group)], dtype=np.int64),
                      np.array([n for _, n in sorted(group)], dtype=np.float64)) for group in groups)
    return Interactions(users, items, freeze(by_user), freeze(by_item),
                        PopularityModel(popularity, frozenset(users), sum(pairs.values())))


def _confidence_weight(counts, mode):
    if mode == 'log':
        return np.log1p(counts)
    if mode == 'binary':
        return (np.asarray(counts) > 0).astype(np.float64)
    raise ValueError('confidence_weighting must be log or binary')


def solve_factors(fixed_factors, observations, *, alpha, regularization, confidence_weighting='log'):
    """Solve one ALS half-step, including confidence-one unobserved pairs."""
    ALSConfig(alpha=alpha, regularization=regularization, confidence_weighting=confidence_weighting)
    fixed = np.asarray(fixed_factors, dtype=np.float64)
    if fixed.ndim != 2 or not np.isfinite(fixed).all():
        raise ValueError('Factors must be a finite two-dimensional matrix')
    base = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
    result = np.zeros((len(observations), fixed.shape[1]), dtype=np.float64)
    for index, (indices, counts) in enumerate(observations):
        indices, counts = np.asarray(indices), np.asarray(counts, dtype=np.float64)
        if (indices.ndim != 1 or counts.shape != indices.shape or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= len(fixed))
                or len(np.unique(indices)) != len(indices)
                or not np.isfinite(counts).all() or np.any(counts <= 0)):
            raise ValueError('Observed indices must be unique/in-range with finite positive counts')
        if not len(indices):
            continue
        observed = fixed[indices]
        extra_confidence = alpha * _confidence_weight(counts, confidence_weighting)
        normal = base + (observed.T * extra_confidence) @ observed
        rhs = observed.T @ (1 + extra_confidence)
        result[index] = np.linalg.solve(normal, rhs)
    if not np.isfinite(result).all():
        raise ValueError('ALS produced nonfinite factors')
    return result


def implicit_objective(user_factors, item_factors, by_user, config):
    """Full training objective without constructing the dense score matrix."""
    x, y = user_factors, item_factors
    # All pairs initially contribute score^2 at confidence 1 and preference 0.
    total = float(np.sum((x.T @ x) * (y.T @ y)))
    for user, (indices, counts) in enumerate(by_user):
        score = y[indices] @ x[user]
        confidence = 1 + config.alpha * _confidence_weight(counts, config.confidence_weighting)
        total += float(np.sum(confidence * (1 - score)**2 - score**2))
    total += config.regularization * float(np.sum(x*x) + np.sum(y*y))
    if not math.isfinite(total) or total < -1e-8:
        raise ValueError('ALS objective is nonfinite or negative')
    return max(0.0, total)


@dataclass(frozen=True)
class ALSModel:
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    user_factors: np.ndarray
    item_factors: np.ndarray
    popularity: PopularityModel
    config: ALSConfig
    loss_history: tuple[float, ...]

    @cached_property
    def user_index(self):
        return {user: index for index, user in enumerate(self.user_ids)}

    @cached_property
    def item_index(self):
        return {item: index for index, item in enumerate(self.item_ids)}

    def recommend(self, user, candidates) -> tuple[str, ...]:
        user = _id(user)
        pool = frozenset(_ids(candidates))
        fallback = self.popularity.rank(pool)
        if len(fallback) < 5:
            raise ValueError('At least five distinct eligible candidates are required')
        if user not in self.user_index:
            return fallback[:5]
        supported = [item for item in fallback if item in self.item_index]
        indices = [self.item_index[item] for item in supported]
        scores = self.item_factors[indices] @ self.user_factors[self.user_index[user]]
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite ALS recommendation scores')
        # Stable sort preserves popularity/numeric-ID order when scores tie.
        ranked = sorted(zip(supported, scores), key=lambda pair: -pair[1])
        selected = [item for item, _ in ranked[:5]]
        for item in fallback:
            if len(selected) == 5:
                break
            if item not in selected:
                selected.append(item)
        return tuple(selected)


def fit_als(data: Interactions, config: ALSConfig, progress=None) -> ALSModel:
    """Fit only supplied training interactions, with a fixed seeded initialization."""
    rng = np.random.default_rng(config.seed)
    x = np.zeros((len(data.user_ids), config.factors), dtype=np.float64)
    y = rng.normal(0, 0.01, (len(data.item_ids), config.factors))
    losses = [implicit_objective(x, y, data.by_user, config)]
    for iteration in range(1, config.iterations + 1):
        x = solve_factors(y, data.by_user, alpha=config.alpha, regularization=config.regularization,
                          confidence_weighting=config.confidence_weighting)
        y = solve_factors(x, data.by_item, alpha=config.alpha, regularization=config.regularization,
                          confidence_weighting=config.confidence_weighting)
        loss = implicit_objective(x, y, data.by_user, config)
        if loss > losses[-1] + 1e-8 * max(1.0, abs(losses[-1])):
            raise ValueError('ALS objective increased beyond numerical tolerance')
        losses.append(loss)
        if progress:
            progress(iteration, loss)
    return ALSModel(data.user_ids, data.item_ids, x, y, data.popularity, config, tuple(losses))
