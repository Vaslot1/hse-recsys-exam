# Implicit-feedback matrix factorization with ALS

Implementation: `als.py`. Training and validation entry point: `train_als.py`.

## Run explicitly

From the project root, install the dependency if it is missing:

```powershell
python -m pip install -r exam/requirements-als.txt
```

Train on the existing training matches and evaluate the fixed validation hold-out:

```powershell
python -B exam/train_als.py --factors 32 --iterations 15 --alpha 20 --regularization 0.1 --seed 42
```

These are also the defaults for `python -B exam/train_als.py`. They are initial, untuned settings. No dataset training or ALS dataset score was produced during implementation. Only tiny synthetic numerical tests were run. The implementation was checked with NumPy 2.4.2; every run records the actual Python and NumPy versions.

The command reads the original schedules/template and saved matching outputs, then creates `exam/derived/als/`. It refuses an existing output directory; use `--output-dir exam/derived/als-run2` for another run. `--source-dir` and `--matching-dir` override inputs. `--help` does not load data or train.

## Model and confidence

The weighted implicit-feedback objective follows [Hu, Koren and Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*](https://yifanhu.net/PUB/cf.pdf). Our confidence transformation and defaults are implementation choices:

```text
n_ui = number of qualifying training broadcast matches for user u, programme i
p_ui = 1 if n_ui > 0, otherwise 0
c_ui = 1 + alpha * log(1 + n_ui)

L = sum over ALL training-user/training-item pairs of c_ui * (p_ui - x_u.T y_i)^2
    + lambda * (sum_u ||x_u||^2 + sum_i ||y_i||^2)
```

Repeated broadcasts increase confidence logarithmically; they do not change binary preference. Missing pairs have preference zero and confidence one. This is low-confidence absence of observed preference, not a proven dislike. Regularization is unscaled by user activity or item support. Scores are raw dot products, not probabilities.

For each user, the exact update solves:

```text
A = Y.T Y + sum_observed (c_ui - 1) * y_i y_i.T + lambda I
b = sum_observed c_ui * y_i
A x_u = b
```

The item update swaps users and items. The `Y.T Y` term includes missing pairs; the observed correction adds only `c - 1`, so confidence is not counted twice. The implementation uses float64, `numpy.linalg.solve`, sparse observed-pair lists and small factor-space Gram matrices. It never allocates a dense user-by-item confidence matrix.

Numeric-sorted string ID mappings, a fixed seed and fixed iteration count make repeated runs reproducible within the same numerical environment. BLAS/platform differences can produce small floating-point differences. The full training objective is recorded before updates and after every complete iteration; nonfinite results or a material increase raise an error. Decreasing training loss does not establish an optimal validation setting.

## Shared protocol

- Fit on all users/items in `[2020-03-09, 2020-05-18)` matching output. Items unavailable during validation still contribute to training.
- Use the same `load_protocol` helper as popularity: `[2020-05-18, 2020-07-27)` candidates from both historical schedules, positive-duration airings, nonzero IDs, and exact 1,260-user template cohort.
- Saved partition boundaries are checked in `data-quality.json`; like the popularity runner, this trusts the match files to correspond to that metadata. It does not reconstruct matching or cryptographically certify provenance.
- Rank supported eligible items by descending ALS score, then distinct-training-user popularity, then ascending exact numeric ID. Previously watched items remain eligible.
- A cold user receives the same popularity top five as the baseline. If fewer than five candidates have fitted factors, complete the list from that same popularity order, skipping selected IDs. Unseen candidates retain zero popularity and are eligible for completion. They do not receive random factors or scores comparable to fitted dot products. Finite negative ALS scores are valid ranks; nonfinite scores raise an error.
- Use the shared `evaluate_rankings` implementation for binary relevance, MAP@5, empty-target handling, cold-item target counts and cohort reports. Validation targets cannot change fitted factors. No validation refit, hyperparameter search or Kaggle-horizon prediction is performed.

## Saved outputs

| File | Contents |
| --- | --- |
| `model.npz` | Float64 user/item factors; matching Unicode user/item ID arrays; training popularity counts aligned with item IDs. Read using `numpy.load(path, allow_pickle=False)`. |
| `training.json` | Configuration, confidence definition, split, dimensions, versions, training time and objective history including iteration zero |
| `evaluation.json` | Shared primary and auxiliary MAP@5, cohort counts, candidates, cold targets, fallback counts, configuration, windows and paths |
| `validation-predictions.csv` | Exactly five ranked eligible IDs for every template user |
| `per-user.csv` | Relevant IDs, predictions, hits, AP@5 and warm/cold information |
| `candidate-popularity.csv` | Full common fallback ranking, including zero-support candidates |

Existing original files and derived results are opened read-only. The new output directory is created after fitting and evaluation succeed.

## Numerical verification

Run all exam tests:

```powershell
python -B -m unittest discover -s exam/tests -v
```

The ALS tests use synthetic fixtures only:

1. **Hand-calculated scalar update.** Fixed item factors `[1, 2]`, only item 1 observed, observed confidence 3, missing confidence 1, lambda 1: `A = 3*1^2 + 1*2^2 + 1 = 8`, `b = 3`, so `x = 3/8 = 0.375`. Actual: **0.375**. With these factors, the complete weighted loss including both factor regularizers is **6.875**, matching the independent calculation.
2. **Dense normal-equation reference.** For three items and two factors, the independent dense confidence-matrix solve gives `[0.22859003773544773, 0.45718007547089534]`. Maximum absolute difference from the sparse update: **5.551115123125783e-17**.
3. **Two-user, three-item toy fixture.** Two factors, alpha 3, lambda 0.2, seed 7, four iterations. Objective history: **13.534204443928 → 6.892801912419 → 1.929724823139 → 1.590958539508 → 1.409997181328**. Repeating the same fit produces identical factor arrays in the tested environment.
4. **Protocol and persistence.** Tests check exact large IDs, repeated broadcasts, negative-score ordering, popularity/numeric ties, repeat eligibility, cold users/items, unavailable items, empty training, invalid configuration and nonfinite factors. A synthetic file run checks the fixed cohort, saved model mappings, unchanged input bytes and refusal to overwrite. Changing only validation targets leaves both factor arrays identical.

These are agent-run implementation checks, not student personal verification or evidence of performance on the dataset. ALS is now the first selected eligible method; a second eligible method and the other unresolved decisions remain open.

## Small confidence and regularization experiment

The tuning runner evaluates this fixed grid; it does not expand the search based on intermediate scores:

| Run | Confidence weighting | Alpha | Regularization |
| --- | --- | ---: | ---: |
| `als-01` | log | 1 | 0.1 |
| `als-02` | log | 1 | 10 |
| `als-03` | log | 20 | 0.1 |
| `als-04` | log | 20 | 10 |
| `als-05` | binary | 1 | 0.1 |
| `als-06` | binary | 1 | 10 |
| `als-07` | binary | 20 | 0.1 |
| `als-08` | binary | 20 | 10 |

All runs use **32 factors, 15 iterations and seed 42**, with a fresh seeded initialization for each fit. `als-03` reproduces the original configuration. Log confidence is `1 + alpha*log1p(n)`; binary confidence is `1 + alpha*indicator(n>0)`. Binary confidence gives all observed pairs equal confidence regardless of repeated broadcasts; missing pairs still have confidence one. These settings test weaker confidence, suppression of repeat counts and stronger regularization while keeping the rest fixed.

Run from the project root:

```powershell
python -B exam/tune_als.py
```

The dataset experiment has **not been run during implementation**. Only tiny synthetic experiments were executed by tests. `--help` lists path overrides without training. To rerun later, use a new `--output-dir`; existing directories are refused.

The runner prepares training interactions, the candidate pool and template users once. Every configuration reuses those inputs, the shared ranker/fallback and the same validation evaluator. Validation only evaluates/selects settings, never fits factors or updates histories. Popularity is recomputed from the same training inputs for a directly comparable baseline; old saved scores are not hardcoded or overwritten.

Outputs are written under `exam/derived/als-tuning/`:

- `comparison.csv`: nine rows after completion (popularity plus eight ALS runs), including configuration, primary MAP@5, difference from popularity, auxiliary MAP@5 with cohort sizes, empty-target count, candidate count, training time and final objective. Scores retain full precision. Rows follow declared run order.
- `experiment.json`: the grid, split, input paths, completion status, completed runs, deterministic selection rule, best ALS run and best overall run including popularity.
- `als-01/` through `als-08/`: model factors, exact ID mappings, configuration and objective history, per-user results, predictions and evaluation metrics, using the same artifacts as the single-run command.
- `popularity/`: recomputed baseline metrics, per-user results and predictions.

Selection uses **primary MAP@5 over all 1,260 users**, including empty targets. ALS ties break by ascending run ID; popularity wins an overall tie. Final objectives are diagnostics only: different confidence/regularization values define different objectives, so their loss magnitudes must not select the winner. No automatic final refit or promotion to a Kaggle submission occurs.

The maximum validation score is a **selection score**, not an unbiased estimate on an untouched test set. The same hold-out has already been inspected; report that tuning history. The runner saves comparison rows after every completed run. A caught error or Ctrl+C records `failed` or `interrupted`; abrupt process termination may leave `running`. Only `complete` confirms the full grid finished. Completed results are preserved; automatic resume is not implemented.

The single-run command also accepts `--confidence-weighting binary`; its default remains `log`. New numerical tests verify binary confidence against `x=3/8` and objective `6.875`, and check the original logarithmic loss sequence remains unchanged. Synthetic integration tests cover the comparison table, model artifacts, identical cohort/candidate counts, input preservation, deterministic winner selection and partial results after failure.
