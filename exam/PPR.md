# Personalized PageRank on the programme co-view graph

Core: `ppr.py`. Runner: `evaluate_ppr.py`. This is the second eligible algorithm, alongside implicit ALS. Popularity remains the additional baseline.

## Commands

From the project root, install dependencies if needed:

```powershell
python -m pip install -r exam/requirements-ppr.txt
```

Run the checks, which use tiny synthetic inputs only:

```powershell
python -B -m unittest discover -s exam/tests -p test_ppr.py -v
```

Run all exam tests:

```powershell
python -B -m unittest discover -s exam/tests -v
```

Build the training graph, compute personalized scores and evaluate validation:

```powershell
python -B exam/evaluate_ppr.py
```

The original full experiment was not run during implementation; the user subsequently ran both damping configurations. The selected damping **0.50** achieved validation MAP@5 **0.7382751322751323**; the original **0.85** run achieved **0.7373518518518519**. Both saved runs are preserved. The evaluation command's defaults remain damping **0.85**, absolute L1 tolerance **1e-10**, maximum **200** iterations and batches of **32** warm users. `--help` loads no data. Options: `--damping`, `--tolerance`, `--max-iterations`, `--batch-size`, `--source-dir`, `--matching-dir`, `--output-dir`.

The default output directory is `exam/derived/ppr`. Existing output directories are refused. For another run use, for example, `--output-dir exam/derived/ppr-run2`. Changing batch size controls working memory; it does not change the model definition, though floating-point summation differences are possible. Implementation checks used NumPy 2.4.2 and SciPy 1.17.1. Every experiment records its actual versions.

## Graph and personalization

Use all qualifying matches from the training partition `[2020-03-09, 2020-05-18)`. Let `B[u,i]=1` if user u watched programme i at least once. Form `W=B.T@B`, then remove its diagonal. Thus an edge weight is the number of distinct training users who watched both programmes. Repeated broadcasts do not multiply edge weights. The graph is undirected with no self-loops or pruning. Keep every training programme, including isolated nodes and programmes unavailable in validation.

For every non-isolated node, normalize outgoing weights to obtain `P[i,j]=W[i,j]/sum_j W[i,j]`. Non-dangling rows sum to one; dangling rows are stored as zero. For a warm user, restart vector s is uniform over their distinct qualifying training programmes, regardless of repeat counts or future availability.

With column-vector probabilities, one iteration is:

```text
x_next = damping * P.T @ x
         + s * ((1 - damping) + damping * sum(x[dangling_nodes]))
```

Initialize `x=s`. Redistributing dangling mass to s makes the effective transition preserve probability mass for each user. This convention is also documented in the [NetworkX PageRank implementation](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html). Our solver uses NumPy/SciPy; NetworkX is not a dependency.

## Numerical checks and convergence

- Reject non-square matrices, negative/nonfinite weights or probabilities, and non-dangling transition rows whose sums differ from one by more than `1e-12`.
- Normalize a finite, nonnegative, positive-mass seed vector once. Reject zero-mass seeds; cold users use the external popularity fallback.
- At every iteration, require finite nonnegative scores and a total mass within `1e-10` of one. Do not renormalize iterates to hide missing mass.
- Stop each user's column independently when `sum(abs(x_next-x)) <= tolerance`. This is an absolute L1 tolerance, not multiplied by the node count. Freeze converged columns while the rest continue.
- Compute the final fixed-point residual `||F(x)-x||_1`, check it against tolerance with a `1e-14` roundoff allowance, and report `residual/(1-damping)` as the contraction-based L1 error bound (subject to floating-point error).
- Raise `ConvergenceError` if the maximum iterations are exhausted. Do not silently use unconverged scores or replace a numerical failure with popularity. The runner creates output files only after all warm users converge and evaluation succeeds.

Known-answer results from synthetic checks:

| Check | Expected | Observed |
| --- | --- | --- |
| Co-view edges from user histories `{1,2}`, `{1,2}`, `{1,3}`, `{4}` | `W[1,2]=2`, `W[1,3]=1`; node 4 dangling | Exact match; repeats ignored |
| Row 1 transition | `[0, 2/3, 1/3, 0]` | Matches |
| Two-node cycle, s=`[1,0]`, damping 0.85 | `[20/37, 17/37]` | `[0.540540540563313, 0.4594594594366869]` |
| Edge 1→2, node 2 dangling, s=`[1,0]`, damping 0.5 | `[2/3, 1/3]` | `[0.6666666666569654, 0.3333333333430346]` |
| All nodes dangling, s proportional to `[1,3]` | `[0.25,0.75]` in one iteration | Matches |
| Independent dense linear solve on three nodes | Same personalized stationary distribution | Maximum difference `1.9978907417339542e-11` |

The cycle converged in 146 iterations, with final residual `8.42583225413307e-11` and maximum mass error `2.220446049250313e-16`. The one-edge dangling example converged in 35 iterations with residual `2.9103830456733704e-11` and zero measured mass error. Both final probability sums were 1. Tests also cover zero damping, disconnected components, batch/scalar agreement, explicit nonconvergence, invalid inputs, exact large IDs, ties, fallback and empty training.

These are agent-run implementation checks, not dataset performance or student personal verification.

## Evaluation and fallback

Reuse `prepare_inputs` for match aggregation and `load_protocol` for the same split checks, candidate pool and 1,260 exact template IDs. The shared helper resides in `train_als.py` but no ALS fitting is called. As with previous runners, partition metadata is checked; the match files are assumed to correspond to that metadata, without reconstructing matching or cryptographic provenance checks.

Diffusion uses the full training graph. Only afterward restrict ranking to the common validation candidates from `[2020-05-18, 2020-07-27)`. Rank graph-supported candidates by decreasing PPR, then distinct-training-user popularity, then ascending exact numeric programme ID. Previously watched programmes remain eligible. Zero-score graph nodes remain eligible with the same tie-breaking policy. Complete lists with fewer than five supported candidates from the shared popularity order; cold candidate items stay eligible for completion. Cold users receive the same five popularity recommendations directly.

Use the existing `evaluate_rankings` for MAP@5, relevance and auxiliary cohort reporting. Cold-item targets remain in relevance. Validation data cannot influence graph edges, restart vectors or PPR scores. Synthetic file tests verify that changing validation rows leaves the graph and scores unchanged, and that original input bytes are preserved.

## Outputs

| File | Contents |
| --- | --- |
| `graph.npz` | CSR co-view weights; load with `scipy.sparse.load_npz` |
| `transition.npz` | CSR normalized transitions, dangling rows zero |
| `graph-items.csv` | Exact item IDs in node order, training support, weighted degree and dangling flag |
| `graph.json` | Graph construction, configuration, versions, transition checks and training statistics |
| `ppr-scores.npz` | Warm-user IDs, exact item IDs and full probability matrix shaped `(warm users, graph nodes)`; use `numpy.load(..., allow_pickle=False)` |
| `ppr-diagnostics.csv` | Every template user: status, iterations, L1 change, residual, error bound, mass error, sum and minimum probability. Cold users have fallback status and blank probability diagnostics. |
| `evaluation.json` | Shared MAP/cohort results, configuration, graph and convergence summaries, windows and timing |
| `validation-predictions.csv` | Five distinct ranked eligible programme IDs per template user |
| `per-user.csv` | Relevance, recommendations, hits and AP@5 |
| `candidate-popularity.csv` | Complete common fallback ordering including zero-support items |

All graph/node/score artifacts use the same node order. The graph can be dense despite sparse storage because long histories create many co-view pairs; batching bounds the intermediate score matrices without changing or pruning the graph. Persisting full warm-user scores also uses `8 * warm_users * graph_nodes` bytes of array memory, independently of batch size. No final refit, hyperparameter search or Kaggle prediction is performed by the evaluation command. Selection and measured comparisons are recorded in [decisions.md](D:/Projects/hse-recsys/exam/decisions.md).

## Final Kaggle CSV generation

From the project root:

```powershell
python -B exam/generate_kaggle.py
```

This separate command fixes the selected configuration at damping **0.50**, tolerance **1e-10**, maximum **200** iterations and default batch size **32**. It rebuilds graph, restart histories and popularity using only `derived/matching/train-matches.csv` for **[2020-03-09, 2020-05-18)**. It does not read validation matches or reuse validation histories. Candidate IDs come only from positive-duration, nonzero-ID airings in `dataset-sweet-tv/export_arh_31-42-final.csv` that overlap **[2020-07-27, 2020-10-19)**, using the same half-open availability rule. The full training graph remains intact; future availability only filters ranked outputs. Shared `ppr.predict_users` supplies the same repeat allowance, ties, cold-user fallback and cold-item completion as evaluation.

The generator checks training-window metadata and every saved training interval, qualifying overlap and recorded row count when present. It preserves exact IDs as strings, checks all **1,260** distinct template users, and reads the written CSV back to verify the exact `user_id,tv_show_id` header, template order, and five distinct positive eligible IDs separated by single spaces. The temporary CSV is renamed to `submission.csv` only after successful validation and audit writing. These checks cannot reconstruct or authenticate the original viewing attribution; saved matches are still trusted derived inputs.

Outputs go to a new directory, **`exam/derived/kaggle-ppr-050`**:

- `submission.csv`: final ranked prediction rows, with no index column.
- `generation-audit.json`: configuration, windows, training/graph/candidate counts, convergence, CSV validation, input/source/output SHA-256, versions and runtime. Input hashes are compared before and after processing to detect changes during the run; they do not prove historical provenance.
- `ppr-diagnostics.csv`: convergence checks or explicit popularity-fallback status for every user.
- `candidate-popularity.csv`: eligible future IDs and their training support, in fallback order.

Existing output directories are refused; use `--output-dir exam/derived/kaggle-ppr-050-run2` for a separate rerun. Other options are `--source-dir`, `--matching-dir`, and `--batch-size`. Original data, previous evaluation configuration and saved runs are preserved. No Kaggle score is calculated or submission uploaded by this script.

The generator was checked with synthetic fixtures only, including a 1,260-user template with an ID larger than signed 64-bit range, training/future boundary checks, cold users/items, invalid CSV rejection and validation-file invariance. The complete exam suite passed **66 tests**. **Real generation, real-CSV validation and Kaggle submission remain pending.**

## Separate count-weighted restart experiment

Run from the project root when ready:

```powershell
python -B exam/evaluate_ppr_counts.py
```

This creates a separate **`exam/derived/ppr-count-restart-050`** directory. Existing directories are refused. It requires the saved `ppr-damping-050` graph and completed `label-diagnostic` outputs, plus the original training matches and four comparator runs. It does not rerun ALS or rebuild graph edges. This experiment has **not** been run on the real dataset during implementation.

Only the restart distribution changes:

```text
n[u,i] = number of qualifying original training broadcasts for user u and programme i
s[u,i] = n[u,i] / sum_j n[u,j]
```

Counts retain the original training watch threshold `O >= min(300 seconds, 0.5 * full broadcast duration)`. They are not viewing seconds, raw visit counts, distinct-user counts or counts filtered at 80%. Fragmented visits within one airing already form one union-overlap match; repeated qualifying airings contribute additional counts. Duplicate saved user/channel/programme/broadcast keys are rejected. Counts include all original training programmes before candidate filtering, including programmes not available in validation.

The runner loads the exact saved `ppr-damping-050/graph.npz`, `transition.npz` and node order. It checks node/support counts, training windows, symmetric nonnegative integer co-view weights, dimensions, degree diagnostics and transition normalization, without changing graph edges. Damping remains **0.50**, tolerance **1e-10**, maximum **200** iterations, batch size **32**. The original split, saved candidate pool, repeat allowance, probability checks, convergence rule, ranking ties, distinct-user popularity fallback and cold-item completion are unchanged. Dangling mass returns to the same user's new count-weighted restart distribution. The shared PPR helper still defaults to **uniform** restart, so existing commands keep their previous behaviour.

### Frozen primary comparison

The primary targets are read from the existing diagnostic's **`coverage80_top5`** target lists. Frequencies must agree with those lists under the existing numeric-ID tie-break. The runner checks the diagnostic's input hashes, configurations, cohort sizes and original comparison scores before computing the variant.

Compare five methods: popularity, tuned ALS `als-06`, saved PPR 0.85, saved uniform PPR 0.50, and count-restart PPR 0.50. Only the last method gets new predictions; the other four reuse their saved lists unchanged. Report the full 1,260-user cohort and the same frozen subset with at least five distinct 80%-qualified programmes. Empty-target AP is zero; shorter targets are not padded; an empty auxiliary cohort produces null MAP. Predictions are evaluated with the existing binary MAP@5 formula.

### Twenty shared seeded tie resolutions

This is **sensitivity analysis, not exact bounds**. Seeds are fixed to integers **0 through 19**; there is no search for a favourable seed.

For each seed and user, keep all programmes whose qualifying frequency exceeds the fifth-place cutoff. If the cutoff is tied across included/excluded programmes, choose the required number uniformly without replacement from the tied programmes sorted by numeric ID. Leave unambiguous memberships and users with at most five qualified programmes unchanged. Changes to the ordering of target IDs have no effect on binary MAP.

The local generator is Python `random.Random`, seeded by the big-endian integer represented by SHA-256 of UTF-8 `ppr-tie-sensitivity-v1|seed|user_id`. This isolates each user/seed from iteration order. It uses only frozen validation frequencies, never a method's predictions or scores. Each seed's realized targets are computed once and shared across **all five methods**, while predictions remain fixed. The realized targets and Python version are saved for reproducibility.

Report per-seed MAP for both cohorts and, over the 20 seeds, empirical mean, sample standard deviation, observed minimum and maximum. Also report paired MAP differences against **saved uniform PPR 0.50**, with their mean/sample SD/min/max and win/tie/loss counts (absolute difference <= `1e-12` counts as a tie). Empty-cohort statistics are null. These observed ranges are not guaranteed extrema; the SD is not a confidence interval or significance test. No sampled target set is selected as a new ground truth. The hidden competition tie-break remains unknown.

### Outputs and checks

| File | Contents |
| --- | --- |
| `validation-predictions.csv` | New count-restart variant's five ranked IDs for every template user |
| `ppr-scores.npz`, `ppr-diagnostics.csv` | New warm-user probability matrix, exact node/user IDs and all-user convergence/fallback diagnostics |
| `evaluation.json` | Variant's primary frozen-target MAP and cohort counts |
| `comparison.csv`, `comparison.json` | All five methods on frozen numeric-tie targets, both cohorts: ten rows |
| `per-user.csv` | Frozen-target AP, hits and rankings for each user/method |
| `tie-targets.csv` | Twenty realized target sets for each user, shared across methods |
| `tie-sensitivity.csv` | Twenty seeds × five methods × two cohorts: 200 rows, including paired deltas |
| `tie-sensitivity-summary.csv`, `.json` | Empirical sensitivity summaries and paired comparisons; JSON explicitly labels interpretation limits |
| `audit.json` | Restart definition, fixed configuration/split, graph reference, training/candidate/cohort counts, seeds/sampling procedure, convergence, versions and source/input hashes |

All consumed input files are read-only and hashed before/after processing. Saved diagnostic input hashes must still match; original score/prediction/target artifacts are not overwritten. These checks bind this run to the supplied snapshots, but do not independently authenticate raw viewing attribution or reconstruct the historical graph. No refitting on weeks 21–30, final Kaggle prediction or submission occurs.

Path overrides: `--source-dir`, `--matching-dir`, `--runs-dir`, `--diagnostic-dir`, `--output-dir`. Because frozen provenance includes resolved input paths, moving an old diagnostic to another data tree without regenerating its provenance is rejected.

Synthetic checks:

```powershell
python -B -m unittest discover -s exam/tests -p test_ppr_counts.py -v
```

A two-node cycle with training counts `(3,1)`, damping 0.50 and restart `(3/4,1/4)` must converge to `(7/12,5/12)`; uniform restart gives `(1/2,1/2)`. Reversing counts reverses the personalized ordering while leaving the graph and popularity unchanged. Tests also cover exact IDs, fallback, shared reproducible sampling independent of methods, fixed cohorts, empty targets, frozen-score reproduction, changed-target/graph/input rejection and preservation of all input bytes in a temporary 1,260-user fixture. No real experiment is run by these tests.

## Final experiment with an eight-minute budget

Prepared command (not executed during implementation):

```powershell
python -B exam/final_ppr_experiment.py --budget-seconds 480 --output-dir exam/derived/ppr-final-8min
```

The command refuses an existing output directory. It reuses the saved graph, completed count-restart 0.50 scores/results, the separately verified count-restart Kaggle submission, frozen validation targets and the **saved** `tie-targets.csv` resolutions. It never fits on validation watches or refits on weeks 11–30. No Kaggle upload is performed.

The experiment sequence is fixed:

1. Verify and reuse count-weighted PPR at **0.50**, including its original validation scores, saved shared-tie results and existing future CSV. Publish this verified baseline before reading training/future rows again for new trials.
2. Evaluate original-threshold count-weighted restart at **0.05, 0.10 and 0.25** on the same graph. Together with the cached 0.50 result, these form the damping comparison.
3. At the best completed damping from step 2, evaluate one further variant whose restart counts include only original **training** broadcasts satisfying `5*overlap_seconds >= 4*duration_seconds`. Do not filter or rebuild graph edges. Original training popularity remains the fallback. Users with no 80%-qualified training watches receive that same fallback, with their count recorded.
4. Select the highest **full-cohort frozen-target MAP@5** among completed verified configurations, including the reused baseline. Exact score ties keep the earlier completed configuration. Auxiliary metrics and tie sensitivity do not drive selection.

Every eligible configuration includes predictions and metrics for all 1,260 users and the unchanged auxiliary cohort, all 20 saved tie resolutions, paired sensitivity differences versus saved count-restart 0.50 and uniform PPR 0.50, and a verified future-schedule Kaggle CSV. Saved ties are checked against frozen frequencies; no new tie sampling or favourable-seed selection occurs. The sensitivity mean, sample SD, observed range and paired win/tie/loss counts describe the 20 realizations, **not exact bounds, confidence intervals or significance tests**.

### Budget and publication

The supervisor starts the wall clock before child startup. Heavy preparation, scoring, sensitivity, CSV read-back, hash checks and publication all run inside the child process. With the 480-second setting, the worker stops at **475 seconds**, reserving five seconds for process shutdown and the small status record. If a calculation blocks, the supervisor terminates the worker; it does not wait for a model iteration to finish. As with any local process, OS shutdown or filesystem stalls are not hard real-time guarantees.

Each configuration's files are immutable once written. A trial becomes eligible only after numerical checks, validation evaluation, all 20 tie checks and CSV verification complete. The worker atomically publishes **`selection.json`**, pointing to the winner's separate immutable CSV at:

```text
exam/derived/ppr-final-8min/completed/<selected-configuration>/submission.csv
```

The exact path and SHA-256 are printed when the command ends. There is deliberately no repeatedly overwritten root-level CSV that could disagree with a selection record after interruption. `selection.json` is the authoritative commit: a partial directory, or a directory not listed in its `completed_configurations`, is ineligible even if some artifacts exist. If time expires, the last committed best verified selection remains usable. The original submissions and completed experiments stay untouched.

Baseline verification still has to finish: if its inputs are missing, inconsistent, or cannot be verified before the budget expires, the command fails explicitly and leaves the original submission intact rather than claim an unverified export. Once the baseline is committed, timeout during training preparation or any trial preserves it or a better completed selection.

### Output evidence

- `selection.json`: selected configuration, restart policy, damping/solver parameters, full/auxiliary MAP, completed results, exact CSV path/hash, source hashes and selection rule.
- `run-status.json`: budget, elapsed time, timeout/worker status and selected configuration/hash.
- `context.json`: frozen cohorts, labels, candidate IDs, windows and input hashes.
- `results.csv`: verified configuration comparison; `selection.json` remains authoritative if interruption occurs during a later table update.
- `completed/<configuration>/`: separate `submission.csv`, validation predictions, per-user evaluation, numerical diagnostics, shared-tie sensitivity results/summary and result metadata.
- `attempts.json` and `worker.log`: trial progress/failures. Incomplete trials are never used as fallback.

The graph and candidate rules are unchanged. A cached ranker precomputes the existing popularity/numeric-ID tie order and uses stable score sorting; tests compare it directly with the original ranker, including score ties, unavailable items and cold completion. The source schedule is checked again before new trials, while the cached baseline is tied to the prior verified CSV and unchanged source hashes.

Tests use synthetic fixtures only and cover the full staged sequence, strict-profile fallback, known ranking parity, exact target/cohort preservation, checkpoint hash/metric verification, atomic selection and forced worker timeout. Run them separately with:

```powershell
python -B -m unittest discover -s exam/tests -p test_final_ppr.py -v
```
