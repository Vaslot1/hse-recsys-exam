# Popularity baseline and local MAP@5

Run from the project root, after `process_matching.py` has produced the two match CSVs and `data-quality.json`:

```powershell
python -B exam/evaluate_popularity.py
```

This evaluates the saved training and validation matches. It does not repeat matching. Python 3.10+ and the standard library are sufficient.

## Protocol

- Fit popularity on all qualifying training matches from `[2020-03-09, 2020-05-18)`. Count distinct users per programme, so repeated broadcasts or duplicate interactions do not inflate popularity.
- Use all nonzero programme IDs with positive-duration scheduled airings overlapping `[2020-05-18, 2020-07-27)` as candidates. Both historical schedule files are read because airings cross file boundaries. Validation viewing never affects popularity.
- Sort candidates by descending distinct-user count, then ascending numeric programme ID. Unseen candidates have count zero. Every user, including cold-start users, receives the same five distinct IDs. Previously watched programmes remain eligible under decisions D05–D06.
- Construct binary relevant sets from qualifying validation matches. Repeated matches collapse to one relevant programme; cold items remain in the denominator. A cohort target outside the candidate pool raises an error instead of being dropped.
- For each user, sum precision at relevant ranks 1–5 and divide by `min(5, number of relevant programmes)`. Empty targets receive AP@5 = 0. Primary MAP@5 averages over exactly the 1,260 unique submission-template users in template order.
- Report auxiliary means for nonempty targets, warm users and cold-start users, with counts. A warm user has any qualifying training history. An empty auxiliary cohort has JSON `null` for its mean.

For example, relevant `{1, 2, 3}` and predictions `[1, 9, 2, 8, 7]` give AP@5 = `(1 + 2/3) / 3 = 5/9`. Prediction lists with duplicates, zero IDs, or a length other than five are rejected. IDs remain exact integers/strings throughout.

The runner checks the saved partition boundaries in `data-quality.json`. This is a metadata consistency check, not a cryptographic verification of match-file contents. It assumes match rows were produced by the agreed matcher.

## Outputs

The command creates `exam/derived/popularity/` containing:

| File | Contents |
| --- | --- |
| `evaluation.json` | Primary and auxiliary MAP@5, cohort sizes, empty-target counts, training/candidate/target statistics, windows and input paths |
| `validation-predictions.csv` | Five ranked IDs for each of the 1,260 users; validation-horizon predictions |
| `per-user.csv` | Relevant IDs, training-history flag, cold-item target count, predictions, hits and AP@5 |
| `candidate-popularity.csv` | Complete eligible candidate ranking with training distinct-user counts, including zeros |

Original files and matching outputs are opened read-only. An existing output directory is refused. For another run, choose a new directory:

```powershell
python -B exam/evaluate_popularity.py --output-dir exam/derived/popularity-run2
```

`--source-dir` and `--matching-dir` override the input directories. `--help` displays options without evaluating data.

## Tests and limitations

```powershell
python -B -m unittest discover -s exam/tests -v
```

Known-answer tests cover rank-sensitive precision, the capped denominator, empty targets, binary relevance, exact IDs, distinct-user popularity, numeric ties, candidate boundaries, repeat eligibility and cold-start groups. Runner tests use temporary synthetic inputs in the tests directory and remove them afterward; they verify saved results and unchanged input bytes.

The full evaluation has not been run as part of implementation. Local relevance remains the agreed qualifying-watch proxy; its equivalence to Kaggle hidden labels is unresolved. Popularity is the mandatory additional baseline, not one of the two eligible algorithms. No final refitting or Kaggle-horizon prediction is performed by this command. Existing open decisions remain open.

## Separate diagnostic: frozen predictions, three label definitions

Run from the project root when ready:

```powershell
python -B exam/diagnose_labels.py
```

This diagnostic does not fit a model, generate new predictions, change the split or submit to Kaggle. It reads the exact saved validation rankings for popularity (`derived/popularity`), tuned ALS (`derived/als-tuning/als-06`: binary confidence, alpha 1, regularization 10), PPR 0.85 (`derived/ppr`) and PPR 0.50 (`derived/ppr-damping-050`). Original evaluations, predictions, match files and source data remain unchanged. Full-dataset diagnostic evaluation has **not** been run during implementation.

The supplied competition description specifies an 80% screen-time filter followed by each user's five most frequently watched programmes. This differs from the original local labels. The diagnostic compares:

| Label name | Relevant programmes |
| --- | --- |
| `current` | Every distinct programme present in the original qualifying validation matches |
| `coverage80_all` | Every distinct programme with at least one broadcast whose saved union overlap covers at least 80% of its full duration |
| `coverage80_top5` | Up to five programmes ranked by their number of broadcasts passing the 80% filter |

Coverage is tested exactly with integers: `5 * overlap_seconds >= 4 * duration_seconds`. All 80%-qualified broadcasts necessarily meet the original `min(300 seconds, half the duration)` threshold, so filtering the saved matches cannot lose an 80%-qualified broadcast **within the existing matcher's cleaning and overlap conventions**. We do not reconstruct viewing fragments or change exclusions. Both the original train `[2020-03-09, 2020-05-18)` and validation `[2020-05-18, 2020-07-27)` windows are checked against saved metadata.

### Fragments, frequency and ties

The matcher already unions overlapping/duplicate viewing intervals per user/channel and accumulates disjoint visits to the same airing. For example, two disjoint 40-second visits to a 100-second airing yield one 80%-qualified broadcast, not two views. The diagnostic counts each saved `(user, channel, programme, broadcast start, broadcast end)` once. It rejects duplicate keys rather than inflate frequency or try to merge already-aggregated records. Different airings, including airings on different channels, count separately. The full broadcast duration remains the denominator even when only part of an airing overlaps the validation partition. Quarantined views, unlabelled programmes and ambiguous schedule intervals retain the original matcher's treatment.

Top-five membership uses descending qualifying-broadcast frequency, then **ascending exact numeric programme ID** for equal counts. The audit reports users with any frequency tie and users whose tie straddles ranks five and six; `labels.csv` lists those cutoff ties and their sizes. The numeric tie-break and fragmented-view/count conventions are explicit local assumptions: the supplied competition description does not fully specify them. The resulting labels are a closer diagnostic proxy, not a claim to recreate the hidden solution exactly.

The AP@5 implementation is unchanged: binary target membership, precision accumulated at relevant recommendation ranks, divided by `min(5, number of targets)`. Frequency determines target membership and stored target ordering; it does not create graded relevance or a positional matching metric. The saved recommendation order is never changed.

### Cohorts and empty targets

Every method/label combination reports these same two cohorts:

- `all_users`: all **1,260** exact template users, with empty-target AP@5 set to zero.
- `at_least_five_coverage80`: one fixed subset having **at least five distinct programmes passing 80% coverage**, defined before truncation to the top five and reused across all three label definitions and all methods.

Using a fixed auxiliary cohort keeps label comparisons from silently changing users. Report its actual size; do not assume the local period gives five qualifying programmes to every user. Users with one to four targets keep those targets without padding, and their AP denominator is that target count. Users with zero targets remain in the full cohort. If the auxiliary cohort is empty, its MAP is `null` in JSON and blank in CSV, not zero. All comparison rows include cohort size, empty-target count and relevant-pair count. The audit also reports users with one to four targets for each label definition.

### Input checks and outputs

Before saving any output, the diagnostic checks all four saved configurations and windows, exact prediction users and ranking validity, and identical candidate IDs/training-support counts. It reconstructs `current` labels from validation matches and requires exact agreement with saved relevant sets, hits and rankings, plus per-user AP and full-cohort MAP within absolute tolerance `1e-12`. It checks saved row counts and empty-target counts. A mismatch fails explicitly; it does not repair or overwrite an old result.

Default output is a **new** directory `exam/derived/label-diagnostic`. Existing output directories are refused. Options are `--source-dir`, `--matching-dir`, `--runs-dir` (root containing the four saved run directories), and `--output-dir`.

| File | Contents |
| --- | --- |
| `comparison.csv`, `comparison.json` | 24 rows: four methods × three label definitions × two cohorts, including MAP@5 and cohort/target counts |
| `per-user.csv` | Frozen predictions, AP@5, hits, target count and auxiliary-cohort membership for every user/method/label |
| `labels.csv` | All three target lists, qualifying broadcast counts, fixed cohort membership and frequency/cutoff-tie details |
| `qualifying-frequencies.csv` | Every user/programme 80%-qualified broadcast count, sorted by frequency then numeric ID |
| `audit.json` | Windows, fixed cohort sizes, target sparsity, ties, policy assumptions, saved model configurations, runtime and input/source SHA-256 hashes |

Input hashes are compared before/after processing. This detects files changing during the run, but does not establish historical provenance or authenticate raw viewing attribution. Only the AP metric is imported; no training or model-scoring module is called. Kaggle-horizon files and reported Kaggle scores are not inputs.

Synthetic checks for this diagnostic:

```powershell
python -B -m unittest discover -s exam/tests -p test_label_diagnostic.py -v
```

These cover known AP values under all three labels, exact/odd-duration coverage boundaries, boundary-crossing airings, fragmented-view aggregation, repeated airings, cutoff ties, empty targets/cohorts, large IDs, invalid matches and changed saved artifacts. Integration tests use a temporary 1,260-user fixture and verify input bytes remain unchanged. They do not run the real diagnostic.
