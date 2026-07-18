# Rexo ERP — Financial Anomaly & Fraud Detection Module

An anomaly/fraud detection module built for a construction/infrastructure ERP system (Rexo ERP), designed to flag unusual or suspicious expense records — duplicates, unusually large amounts, irregular vendor/approver activity, off-hours submissions — that a finance team would otherwise have to catch manually.

Built as part of a software engineering internship project. This repo contains the full pipeline: synthetic data generation with controlled ground truth, detection logic, evaluation against that ground truth, and a reviewer-facing HTML dashboard.

## Results

Evaluated on 1,950 synthetic expense records (150 injected anomalies across 5 types, ~7.7% anomaly rate):

| Metric | Value |
|---|---|
| Precision (at best-F1 threshold) | 74.6% |
| Recall | 92.0% |
| F1 | 0.824 |
| Precision @ top 25 ranked | 100% |
| Average Precision (PR-AUC) | 0.869 |
| Records flagged for review | 185 / 1,950 (9.5%) |

Recall by anomaly type: duplicates and extreme outliers are caught perfectly (100%), unusual-approver and round-number patterns are strong (92–97%), off-hours submissions are the weakest category (70%) — a known limitation, not hidden.

Full breakdown in [`reports/evaluation_results.txt`](reports/evaluation_results.txt).

## Approach

The pipeline is an **ensemble of three signal families**, each producing a 0–1 suspicion score per record, combined into one weighted `suspicion_score` used to rank records for review.

**1. Rule-based checks** (deterministic, fully explainable)
- Duplicate entries — same vendor + site + amount within a 3-day window
- Round-number amounts — exact multiples of ₹25,000 ≥ ₹50,000 (real invoices rarely land on round figures)
- Off-hours submission — weekends or 11 PM–5 AM
- Unusual approver — a vendor's invoice approved by someone outside that vendor's normal approver set, above the vendor's median amount

**2. Statistical outlier detection**
Robust z-score (median + MAD, not mean/std) computed **per vendor/category group** rather than globally — a cement supplier and an equipment rental company have very different normal spending ranges, so a single global threshold doesn't work. Median/MAD is used instead of mean/std specifically because a handful of the very outliers being detected would otherwise distort the mean and mask themselves.

**3. Isolation Forest (ML)**
Catches multivariate anomalies the rules and stats don't check for directly — e.g. an unremarkable amount at a normal hour, but from a vendor+approver pairing that almost never co-occurs. Chosen over One-Class SVM for scalability, no kernel tuning needed on mixed features, and because it natively outputs a continuous score usable for ranking (not just a binary label).

Why an ensemble rather than one method: rules are precise but only catch what you explicitly checked for; per-group statistics generalize "is this normal for this vendor" better than any fixed rule; Isolation Forest catches combinations neither of the others sees — but is the least explainable of the three, so it's weighted lowest (15%) rather than being the primary driver.

## Project structure

```
rexo-fraud-detection/
├── src/
│   ├── generate_data.py      # synthetic dataset generator with 5 controlled anomaly types
│   ├── detect_anomalies.py   # rule-based + statistical + Isolation Forest ensemble pipeline
│   ├── evaluate.py           # precision/recall/F1/confusion matrix + threshold sweep vs ground truth
│   └── build_dashboard.py    # generates the self-contained reviewer HTML dashboard
├── data/
│   ├── expenses.csv          # generated synthetic dataset (with ground-truth labels)
│   └── scored_expenses.csv   # dataset scored + ranked by suspicion_score
├── reports/
│   ├── rexo_suspicious_records_report.html   # ranked, filterable dashboard for finance reviewers
│   └── evaluation_results.txt                # full metrics output
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

python src/generate_data.py       # generates data/expenses.csv
python src/detect_anomalies.py    # scores records -> data/scored_expenses.csv
python src/evaluate.py            # prints precision/recall/F1/confusion matrix
python src/build_dashboard.py     # builds reports/rexo_suspicious_records_report.html
```

Then open `reports/rexo_suspicious_records_report.html` directly in a browser — no server required.

## The dashboard

A single self-contained HTML file (vanilla JS, no build step) for a finance reviewer to triage flagged records:
- Sortable/searchable ranked table, filterable by category or risk level
- Plain-English "reasons flagged" per record (e.g. *"possible duplicate entry; submitted off-hours/weekend"*) so a reviewer doesn't need to understand the underlying model to trust a flag
- Summary stats: total scanned, flagged count, precision in top-ranked results

## Known limitations / next steps

- Ground truth is synthetic, injected to match the exact signals this pipeline checks for — it validates the pipeline works *as designed*, not that it generalizes to real-world fraud patterns not anticipated here (e.g. structuring/split transactions just under an approval threshold, collusion between vendor and approver).
- Off-hours detection is the weakest signal (70% recall) — worth revisiting with a softer, context-aware definition of "unusual" hours per site rather than one fixed window.
- Isolation Forest is fit on the full dataset in this demo; in production it should be trained on a historical "known-clean" window and re-fit periodically as spending patterns shift.
