"""
detect_anomalies.py
--------------------
Rexo ERP anomaly/fraud detection pipeline.

Architecture: an ENSEMBLE of three signal families, each producing a score
in [0, 1] per record. We combine them into a single weighted suspicion_score
used to rank records for a finance reviewer.

    1. RULE-BASED   -- deterministic, fully explainable checks
    2. STATISTICAL  -- per-vendor/category robust outlier detection
    3. ML            -- Isolation Forest over engineered features, to catch
                        multivariate patterns the rules/stats miss

Why an ensemble and not just one method (this is the thing to be able to
defend in an interview):
  - Rules alone are precise but brittle -- they only catch what you thought
    to check for.
  - Per-group statistics generalize the "is this amount normal FOR THIS
    VENDOR" question, which a single global rule can't.
  - Isolation Forest catches multivariate anomalies (weird combination of
    features) that no single-feature rule or stat check would trigger, but
    is the least explainable of the three, so we keep it as ~15% of the
    final score rather than the primary driver.

Each function below is intentionally isolated and commented so any one
signal can be explained, tuned, or removed independently.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ---------------------------------------------------------------------------
# 1. RULE-BASED SIGNALS
# ---------------------------------------------------------------------------

def flag_duplicates(df):
    """
    Flags near-duplicate submissions: same vendor + site + amount (rounded
    to the rupee) within a 3-day window. Grouping instead of pairwise
    comparison keeps this O(n log n) rather than O(n^2).
    """
    df = df.sort_values("submitted_at").copy()
    df["_amount_r"] = df["amount"].round(0)
    score = pd.Series(0.0, index=df.index)

    for _, group in df.groupby(["vendor", "site_id", "_amount_r"]):
        if len(group) < 2:
            continue
        times = group["submitted_at"].sort_values()
        # Any pair within 3 days of each other -> both flagged
        for i in range(len(times) - 1):
            gap = (times.iloc[i + 1] - times.iloc[i]).days
            if gap <= 3:
                score.loc[times.index[i]] = 1.0
                score.loc[times.index[i + 1]] = 1.0

    df["rule_duplicate"] = score
    return df.drop(columns="_amount_r").sort_index()


def flag_round_numbers(df):
    """
    Flags amounts that are suspiciously 'round': exact multiples of 25,000
    that are also >= 50,000. Real invoices (materials cost, labor hours,
    equipment day-rates) essentially never land on exact round rupee
    figures at this scale -- when they do, it suggests an estimated or
    fabricated number rather than a real line-item total.
    """
    is_round = (df["amount"] % 25000 == 0) & (df["amount"] >= 50000)
    df = df.copy()
    df["rule_round_number"] = is_round.astype(float)
    return df


def flag_off_hours(df):
    """
    Flags weekend submissions or submissions in the 11 PM - 5 AM window.
    Legitimate site/procurement entries are made during business operations;
    off-hours submission is a common heuristic for automated/fraudulent
    entry or after-the-fact backdating.
    """
    df = df.copy()
    hour = df["submitted_at"].dt.hour
    weekday = df["submitted_at"].dt.weekday
    is_weekend = weekday >= 5
    is_odd_hour = (hour >= 23) | (hour <= 5)
    df["rule_off_hours"] = (is_weekend | is_odd_hour).astype(float)
    return df


def flag_unusual_approver(df):
    """
    For each vendor, determine its 'regular' approvers as those responsible
    for >= 15% of that vendor's historical approvals. A record is flagged if
    its approver is NOT in that regular set AND the amount is above the
    vendor's median (i.e., a high-value approval routed around the usual
    approver -- the actual risk scenario, not just any one-off approval).

    NOTE: computed from the dataset itself (unsupervised) -- this only uses
    approver/vendor/amount, never the ground-truth label.
    """
    df = df.copy()
    vendor_approver_freq = (
        df.groupby("vendor")["approver"]
        .value_counts(normalize=True)
        .rename("freq")
        .reset_index()
    )
    regular_pairs = set(
        tuple(x) for x in vendor_approver_freq.loc[
            vendor_approver_freq["freq"] >= 0.15, ["vendor", "approver"]
        ].values
    )
    vendor_median = df.groupby("vendor")["amount"].median()

    def score_row(row):
        is_irregular = (row["vendor"], row["approver"]) not in regular_pairs
        is_high_value = row["amount"] > vendor_median[row["vendor"]]
        return 1.0 if (is_irregular and is_high_value) else 0.0

    df["rule_unusual_approver"] = df.apply(score_row, axis=1)
    return df


# ---------------------------------------------------------------------------
# 2. STATISTICAL SIGNAL: robust per-vendor/category outlier score
# ---------------------------------------------------------------------------

def statistical_outlier_score(df):
    """
    Robust z-score of amount within each (vendor, category) group, using
    median and MAD (median absolute deviation) instead of mean/std.

    Why robust stats instead of a plain z-score: mean/std are themselves
    distorted by the extreme outliers we're trying to detect (a single
    12x-inflated invoice drags the mean up and inflates std, which then
    *masks* the very outlier we want to catch). Median/MAD are far less
    sensitive to a handful of extreme points, so the score stays meaningful
    even with a few injected anomalies sitting in the same group.

    Score is the robust z-score, clipped and min-max normalized to [0, 1]
    per group so it combines cleanly with the other 0-1 signals.
    """
    df = df.copy()
    MAD_CONST = 1.4826  # scales MAD to be a consistent estimator of std
                         # under a normal distribution

    def robust_z(group):
        median = group.median()
        mad = (group - median).abs().median() * MAD_CONST
        if mad == 0:
            mad = group.std() if group.std() > 0 else 1.0
        return (group - median).abs() / mad

    df["stat_robust_z"] = (
        df.groupby(["vendor", "category"])["amount"]
        .transform(robust_z)
    )
    # Normalize to 0-1: z >= 6 is treated as maximally suspicious (roughly
    # matches our injected extreme_outlier multiplier range of 6-12x)
    df["stat_score"] = (df["stat_robust_z"] / 6.0).clip(0, 1)
    return df


# ---------------------------------------------------------------------------
# 3. ML SIGNAL: Isolation Forest over engineered features
# ---------------------------------------------------------------------------

def isolation_forest_score(df, contamination=0.08, random_state=42):
    """
    Isolation Forest isolates points that are easy to separate from the rest
    of the data with few random splits -- anomalies need fewer splits to
    isolate than normal points, by construction. Chosen over One-Class SVM
    because: (a) it scales better and needs no kernel/gamma tuning on mixed
    numeric/categorical-encoded features, (b) it doesn't assume a single
    dense "normal" region shape, and (c) it natively outputs a continuous
    anomaly score, which we need for ranking (not just an in/out label).

    contamination=0.08 is set close to our expected anomaly rate (~7.7%)
    -- in production this would be tuned on a validation slice, not read
    off the test labels.

    Features are intentionally a mix of the signals above PLUS raw context
    (hour, weekday, vendor/approver frequency) so the model can catch
    *combinations* the individual rules don't check for, e.g. a
    mid-range amount at a normal hour but from a vendor+approver pairing
    that almost never co-occurs.
    """
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    features["amount_log"] = np.log1p(df["amount"])
    features["stat_robust_z"] = df["stat_robust_z"]
    features["hour"] = df["submitted_at"].dt.hour
    features["weekday"] = df["submitted_at"].dt.weekday
    features["is_weekend"] = (features["weekday"] >= 5).astype(int)

    # Frequency encoding: how common is this vendor / approver / category
    # / (vendor, approver) pair overall? Rare combinations are informative
    # for an isolation-based model without needing one-hot explosion.
    features["vendor_freq"] = df["vendor"].map(df["vendor"].value_counts(normalize=True))
    features["approver_freq"] = df["approver"].map(df["approver"].value_counts(normalize=True))
    pair_counts = df.groupby(["vendor", "approver"]).size()
    features["vendor_approver_freq"] = df.apply(
        lambda r: pair_counts[(r["vendor"], r["approver"])] / len(df), axis=1
    )
    features["category_code"] = df["category"].astype("category").cat.codes

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(features)
    # decision_function: higher = more normal. We flip sign and min-max
    # normalize so higher = more anomalous, consistent with our other scores.
    raw = -iso.decision_function(features)
    df["ml_score"] = (raw - raw.min()) / (raw.max() - raw.min())
    return df


# ---------------------------------------------------------------------------
# 4. ENSEMBLE: combine all signals into one ranked suspicion score
# ---------------------------------------------------------------------------

# Weights reflect how much we trust each signal. Rule-based checks (duplicate,
# round number) are near-certain fraud/error indicators when they fire, so
# they carry more weight than the softer statistical/ML signals, which are
# suggestive rather than conclusive on their own.
WEIGHTS = {
    "rule_duplicate": 0.25,
    "rule_round_number": 0.15,
    "rule_off_hours": 0.10,
    "rule_unusual_approver": 0.15,
    "stat_score": 0.20,
    "ml_score": 0.15,
}


def run_pipeline(df):
    df = df.copy()
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])

    df = flag_duplicates(df)
    df = flag_round_numbers(df)
    df = flag_off_hours(df)
    df = flag_unusual_approver(df)
    df = statistical_outlier_score(df)
    df = isolation_forest_score(df)

    df["suspicion_score"] = sum(df[col] * w for col, w in WEIGHTS.items())
    df["suspicion_score"] = df["suspicion_score"].round(4)

    # Human-readable reason string: which signals fired, for the reviewer UI
    def reasons(row):
        r = []
        if row["rule_duplicate"] == 1.0:
            r.append("possible duplicate entry")
        if row["rule_round_number"] == 1.0:
            r.append("suspiciously round amount")
        if row["rule_off_hours"] == 1.0:
            r.append("submitted off-hours/weekend")
        if row["rule_unusual_approver"] == 1.0:
            r.append("unusual approver for this vendor")
        if row["stat_score"] > 0.5:
            r.append("amount far outside vendor's typical range")
        if row["ml_score"] > 0.7:
            r.append("unusual combination of features (model flagged)")
        return "; ".join(r) if r else "no strong signal"

    df["flag_reasons"] = df.apply(reasons, axis=1)
    return df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(os.path.join(DATA_DIR, "expenses.csv"))
    scored = run_pipeline(raw)
    scored.to_csv(os.path.join(DATA_DIR, "scored_expenses.csv"), index=False)
    print(scored[["record_id", "vendor", "amount", "suspicion_score",
                   "is_anomaly", "flag_reasons"]].head(15).to_string(index=False))
