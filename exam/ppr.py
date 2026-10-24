"""Training programme co-view graph and checked, batched Personalized PageRank."""
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy import sparse

if __package__:
    from .evaluation import _ids, _cohort
else:
    from evaluation import _ids, _cohort


MASS_TOLERANCE = 1e-10
TRANSITION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PPRConfig:
    damping: float = 0.85
    tolerance: float = 1e-10
    max_iterations: int = 200

    def __post_init__(self):
        for name in ('damping', 'tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.damping < 1 or self.tolerance <= 0:
            raise ValueError('Require 0 <= damping < 1 and tolerance > 0')
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')


def _nonnegative_square(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Graph matrix must be square')
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Graph weights/probabilities must be finite and nonnegative')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _check_transition(transition):
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    dangling = row_sums == 0
    error = float(np.max(np.abs(row_sums[~dangling]-1), initial=0))
    if not np.isfinite(row_sums).all() or error > TRANSITION_TOLERANCE:
        raise ValueError('Non-dangling transition rows must sum to one')
    return dangling, error


def transition_matrix(weights):
    """Normalize outgoing weights; dangling rows stay zero until personalized repair."""
    weights = _nonnegative_square(weights)
    degree = np.asarray(weights.sum(axis=1)).ravel()
    if not np.isfinite(degree).all():
        raise ValueError('Graph degree overflow')
    inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    transition = (sparse.diags(inverse) @ weights).tocsr()
    _check_transition(transition)
    return transition


@dataclass(frozen=True)
class CoViewGraph:
    item_ids: tuple[str, ...]
    weights: sparse.csr_matrix
    transition: sparse.csr_matrix
    diagnostics: dict

    @cached_property
    def item_index(self):
        return {item: index for index, item in enumerate(self.item_ids)}


def build_graph(data) -> CoViewGraph:
    """Each distinct training user contributes one to each pair of watched programmes."""
    lengths = np.array([len(indices) for indices, _ in data.by_user], dtype=np.int64)
    columns = np.concatenate([indices for indices, _ in data.by_user]) if lengths.sum() else np.array([], dtype=np.int64)
    rows = np.repeat(np.arange(len(data.user_ids)), lengths)
    incidence = sparse.csr_matrix((np.ones(len(columns)), (rows, columns)),
                                  shape=(len(data.user_ids), len(data.item_ids)))
    weights = (incidence.T @ incidence).tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    transition = transition_matrix(weights)
    dangling, row_error = _check_transition(transition)
    symmetry = weights-weights.T
    symmetry_error = float(np.max(np.abs(symmetry.data), initial=0))
    if symmetry_error != 0 or np.any(weights.diagonal() != 0):
        raise ValueError('Co-view graph must be symmetric without self-loops')
    diagnostics = {
        'nodes': len(data.item_ids),
        'undirected_edges': weights.nnz // 2,
        'directed_nonzeros': weights.nnz,
        'dangling_nodes': int(dangling.sum()),
        'training_user_item_pairs': int(incidence.nnz),
        'max_transition_row_sum_error': row_error,
        'symmetry_error': symmetry_error,
        'min_edge_weight': float(weights.data.min()) if weights.nnz else None,
        'max_edge_weight': float(weights.data.max()) if weights.nnz else None,
        'min_positive_transition': float(transition.data.min()) if transition.nnz else None,
        'max_transition': float(transition.data.max()) if transition.nnz else None,
    }
    return CoViewGraph(data.item_ids, weights, transition, diagnostics)


class ConvergenceError(RuntimeError):
    """Power iteration exhausted its budget without satisfying the absolute L1 check."""


@dataclass(frozen=True)
class PPRResult:
    scores: np.ndarray  # Nodes x personalization vectors.
    iterations: np.ndarray
    last_delta: np.ndarray
    residual: np.ndarray
    max_mass_error: np.ndarray


def _check_probability_columns(scores):
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('PageRank probabilities must be finite and nonnegative')
    error = np.abs(scores.sum(axis=0)-1)
    if np.any(error > MASS_TOLERANCE):
        raise ValueError('PageRank probability mass is not conserved')
    return error


def personalized_pagerank(transition, seeds, config: PPRConfig) -> PPRResult:
    """Use row-stochastic P, column-vector scores and personalized dangling repair.

    Stop each column when ||x_next-x||_1 <= tolerance (not node-count scaled).
    Validate mass at every step without renormalizing iterates. Cold users must
    use the external popularity fallback; zero-mass seed vectors are rejected.
    """
    transition = _nonnegative_square(transition)
    dangling, _ = _check_transition(transition)
    seeds = np.array(seeds, dtype=np.float64, copy=True)
    if seeds.ndim == 1:
        seeds = seeds[:, None]
    if (seeds.ndim != 2 or seeds.shape[0] != transition.shape[0] or not seeds.shape[1]
            or not np.isfinite(seeds).all() or np.any(seeds < 0)):
        raise ValueError('Personalization must be finite/nonnegative with one row per graph node')
    totals = seeds.sum(axis=0)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError('Each personalization vector needs positive finite mass')
    seeds /= totals
    x = seeds.copy()
    mass_error = _check_probability_columns(x)
    iterations = np.zeros(seeds.shape[1], dtype=np.int64)
    last_delta = np.full(seeds.shape[1], np.inf)
    transpose = transition.T.tocsr()

    def step(previous, restart):
        dangling_mass = previous[dangling].sum(axis=0)
        return config.damping * (transpose @ previous) + restart * (1-config.damping + config.damping*dangling_mass)

    for iteration in range(1, config.max_iterations+1):
        active = np.flatnonzero(iterations == 0)
        previous = x[:, active]
        updated = step(previous, seeds[:, active])
        mass_error[active] = np.maximum(mass_error[active], _check_probability_columns(updated))
        delta = np.abs(updated-previous).sum(axis=0)
        last_delta[active] = delta
        x[:, active] = updated
        iterations[active[delta <= config.tolerance]] = iteration
        if np.all(iterations > 0):
            break
    else:
        raise ConvergenceError(f'{int((iterations == 0).sum())} personalization vectors did not converge '
                               f'after {config.max_iterations} iterations; max L1 change={last_delta.max():.6g}')
    next_x = step(x, seeds)
    mass_error = np.maximum(mass_error, _check_probability_columns(next_x))
    residual = np.abs(next_x-x).sum(axis=0)
    if np.any(residual > config.tolerance + 1e-14):
        raise ConvergenceError('Final PageRank fixed-point residual exceeds tolerance')
    return PPRResult(x, iterations, last_delta, residual, mass_error)


def rank_scores(graph: CoViewGraph, scores, popularity, candidates) -> tuple[str, ...]:
    """Rank supported eligible nodes, retain repeats, and complete using popularity."""
    pool = frozenset(_ids(candidates))
    fallback = popularity.rank(pool)
    if len(fallback) < 5:
        raise ValueError('At least five distinct eligible candidates are required')
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(graph.item_ids),) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Expected one finite nonnegative score per graph node')
    supported = [item for item in fallback if item in graph.item_index]
    # Stable ordering retains popularity then numeric ID for equal PPR scores.
    supported.sort(key=lambda item: -scores[graph.item_index[item]])
    selected = supported[:5]
    for item in fallback:
        if len(selected) == 5:
            break
        if item not in selected:
            selected.append(item)
    return tuple(selected)


@dataclass(frozen=True)
class UserPredictions:
    predictions: dict
    diagnostics: dict
    warm_users: tuple[str, ...]
    scores: np.ndarray  # Warm users x graph nodes, in the listed orders.


def predict_users(data, graph: CoViewGraph, users, candidates, config: PPRConfig,
                  batch_size=32, progress=None, *, restart_weighting='uniform') -> UserPredictions:
    """Shared evaluation/final ranking; candidate availability never changes fitting."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if restart_weighting not in ('uniform', 'watch_counts'):
        raise ValueError('restart_weighting must be uniform or watch_counts')
    users = _cohort(users)
    candidates = frozenset(_ids(candidates))
    if tuple(data.item_ids) != tuple(graph.item_ids):
        raise ValueError('Graph node order must match training interaction item order')
    user_index = {user: i for i, user in enumerate(data.user_ids)}
    warm_users = [user for user in users if user in user_index]
    popularity = data.popularity
    fallback = popularity.top_five(candidates)
    predictions = {}
    diagnostics = {}
    saved_scores = np.zeros((len(warm_users), len(graph.item_ids)), dtype=np.float64)
    for user in users:
        if user not in user_index:
            predictions[user] = fallback
            diagnostics[user] = {
                'user_id': user, 'status': 'popularity_fallback', 'iterations': 0,
                'last_l1_change': None, 'fixed_point_residual': None, 'l1_error_bound': None,
                'max_mass_error': None, 'probability_sum': None, 'minimum_probability': None,
            }
    for offset in range(0, len(warm_users), batch_size):
        batch = warm_users[offset:offset+batch_size]
        seeds = np.zeros((len(graph.item_ids), len(batch)), dtype=np.float64)
        for col, user in enumerate(batch):
            indices, counts = data.by_user[user_index[user]]
            if restart_weighting == 'uniform':
                seeds[indices, col] = 1 / len(indices)
            else:
                total = counts.sum()
                if not np.isfinite(counts).all() or np.any(counts <= 0) or not np.isfinite(total) or total <= 0:
                    raise ValueError('Restart watch counts must have positive finite mass')
                seeds[indices, col] = counts / total
        result = personalized_pagerank(graph.transition, seeds, config)
        saved_scores[offset:offset+len(batch)] = result.scores.T
        for col, user in enumerate(batch):
            scores = result.scores[:, col]
            predictions[user] = rank_scores(graph, scores, popularity, candidates)
            diagnostics[user] = {
                'user_id': user, 'status': 'converged', 'iterations': int(result.iterations[col]),
                'last_l1_change': float(result.last_delta[col]),
                'fixed_point_residual': float(result.residual[col]),
                'l1_error_bound': float(result.residual[col] / (1-config.damping)),
                'max_mass_error': float(result.max_mass_error[col]),
                'probability_sum': float(scores.sum()), 'minimum_probability': float(scores.min()),
            }
        if progress is not None:
            progress(offset+len(batch), len(warm_users))
    return UserPredictions(predictions, diagnostics, tuple(warm_users), saved_scores)
