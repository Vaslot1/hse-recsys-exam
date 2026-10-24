# SWEET.TV programme recommendation — GSB exam

Valerii Bobrov / Vaslot1. Code and final submission: https://github.com/Vaslot1/hse-recsys-exam

- [PDF report](report/bobrov_valerii_exam_report.pdf)
- [Final local submission CSV](submission.csv)
- [Kaggle score screenshot](exam/derived/report/kaggle-screenshot.png)
- [Final configuration and verification](exam/derived/ppr-final-8min/completed/coverage80-d025/result.json)
- [Comparison table](exam/derived/ppr-final-8min/results.csv)
- [Two-user recommendation evidence](exam/derived/report/two-users.json)

## Selected configuration and evidence

Programme co-view Personalized PageRank, damping 0.25, restart proportional to each user's training watch counts after at least 80% programme coverage. The graph and popularity fallback retain the original training interactions. All fitting uses 2020-03-09 through 2020-05-18 (end exclusive); validation ends 2020-07-27. No validation refit.

Frozen coverage80 top-five validation MAP@5: **0.3670050705467372** over all 1,260 users; **0.38814492753623187** over the fixed 1,150-user subset with at least five qualifying programmes. All five final experiment configurations completed in 210.344 seconds. Twenty shared seeded tie resolutions are empirical sensitivity analysis, not confidence intervals or exact bounds.

The screenshot supplied by the student shows a highest private score **0.34776**, public **0.35400**. It does not expose a submission ID or hash: it verifies the displayed scores, but cannot independently bind the uploaded bytes to the local CSV. This repository preserves the selected local CSV exactly; no such binding is fabricated.

CSV SHA-256: `16dae00a24730568033467a8e09aec4dc4b63e963888b064c74277b4d48792de`.

The CSV passes exact-template checks for 1,260 users with five distinct nonzero future-eligible programme IDs each. Automated checks are not a claim of personal student verification. The native AI-session export is a separate course deliverable and is not published here.

## Code and checks

Python 3.12.10; recorded NumPy 2.4.2 and SciPy 1.17.1. Install dependencies and run the small tests:

```powershell
python -m pip install -r exam/requirements-als.txt -r exam/requirements-ppr.txt
python -B -m unittest discover -s exam/tests -v
```

The audit passed 86 tests. Raw competition data and large generated matches, graphs and factor arrays are intentionally not distributed. Obtain the competition data under its terms and put the five files in `exam/dataset-sweet-tv/` with the names expected by the scripts. Default output directories must not already contain completed runs; use fresh directories where supported.

Execution sequence for a fresh reproduction (expensive; not run by cloning):

```powershell
python -B exam/process_matching.py
python -B exam/evaluate_popularity.py
python -B exam/train_als.py
python -B exam/tune_als.py
python -B exam/evaluate_ppr.py
python -B exam/evaluate_ppr.py --damping 0.50 --output-dir exam/derived/ppr-damping-050
python -B exam/diagnose_labels.py
python -B exam/evaluate_ppr_counts.py
```

See [PPR instructions](exam/PPR.md), [ALS instructions](exam/ALS.md), [matching](exam/MATCHING.md) and [evaluation](exam/EVALUATION.md) for arguments and output dependencies. The final bounded experiment reuses completed count-restart results, graph and an already verified count-restart Kaggle export; those large cache dependencies are not all included here, so it is not a stand-alone fresh-clone command. Saved results permit inspection without running training. Source audit paths inside JSON refer to the original Windows workspace.

The method documentation includes historical experiments; the selected configuration above and `selection.json` supersede earlier uniform-restart selections. Label definitions must not be mixed when comparing scores. The PDF is five body pages plus references and records outstanding verification limitations.
