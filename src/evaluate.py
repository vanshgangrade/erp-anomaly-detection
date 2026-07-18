"""
evaluate.py
-----------
Evaluates the anomaly detection pipeline against injected ground-truth
labels. This is the part that turns "it flagged N records" into numbers you
can actually defend: precision, recall, F1, a confusion matrix, and a
precision/recall-vs-threshold sweep so we can justify *where* the operating
threshold is set.

Two complementary views, because a finance team consumes this differently:

1. THRESHOLD-BASED classification: "flag everything with suspicion_score
   >= T" -- this is what a rule in production looks like.
2. TOP-K ranking quality: "of the top K most suspicious records a reviewer
   actually has time to check, how many are real anomalies" -- this is what
   the reviewer actually experiences, since in practice they work down a
   ranked list, not a yes/no flag.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
)


def evaluate_at_threshold(df, threshold):
    y_true = df["is_anomaly"].values
    y_pred = (df["suspicion_score"] >= threshold).astype(int).values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_count": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def sweep_thresholds(df, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.55, 0.025)
    rows = [evaluate_at_threshold(df, t) for t in thresholds]
    return pd.DataFrame(rows)


def top_k_precision_recall(df, k_values=None):
    """Of the top-K ranked records, what fraction are true anomalies
    (precision@K), and what fraction of ALL true anomalies did we capture
    in the top K (recall@K)? This mirrors how a reviewer actually works:
    top-down through the ranked list."""
    total_anomalies = df["is_anomaly"].sum()
    if k_values is None:
        k_values = [25, 50, 75, 100, 150, 200, 300]
    rows = []
    for k in k_values:
        top_k = df.head(k)
        tp = top_k["is_anomaly"].sum()
        rows.append({
            "k": k,
            "precision_at_k": tp / k,
            "recall_at_k": tp / total_anomalies,
        })
    return pd.DataFrame(rows)


def best_f1_threshold(df):
    y_true = df["is_anomaly"].values
    scores = df["suspicion_score"].values
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f1s[best_idx]


def per_anomaly_type_recall(df, threshold):
    """Recall broken down by anomaly type -- tells us which fraud patterns
    the pipeline is strong/weak on, which is far more useful for iterating
    than a single blended recall number."""
    flagged = df["suspicion_score"] >= threshold
    anomalies = df[df["is_anomaly"] == 1].copy()
    anomalies["caught"] = flagged.loc[anomalies.index]
    return (
        anomalies.groupby("anomaly_type")["caught"]
        .agg(["sum", "count"])
        .assign(recall=lambda x: x["sum"] / x["count"])
        .rename(columns={"sum": "caught", "count": "total"})
    )


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    df = pd.read_csv(os.path.join(DATA_DIR, "scored_expenses.csv"))

    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)
    sweep = sweep_thresholds(df)
    print(sweep.to_string(index=False))

    print()
    print("=" * 70)
    print("BEST-F1 OPERATING THRESHOLD")
    print("=" * 70)
    t, p, r, f1 = best_f1_threshold(df)
    print(f"threshold={t:.4f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    print()
    print("=" * 70)
    print(f"CONFUSION MATRIX @ threshold={t:.4f}")
    print("=" * 70)
    result = evaluate_at_threshold(df, t)
    print(f"                 Predicted Normal   Predicted Anomaly")
    print(f"Actual Normal    {result['true_negatives']:>16}   {result['false_positives']:>17}")
    print(f"Actual Anomaly   {result['false_negatives']:>16}   {result['true_positives']:>17}")
    print(f"\nPrecision: {result['precision']:.3f}  |  Recall: {result['recall']:.3f}  |  F1: {result['f1']:.3f}")
    print(f"Flagged {result['flagged_count']} of {len(df)} records "
          f"({result['flagged_count']/len(df)*100:.1f}%) for review")

    print()
    print("=" * 70)
    print("PRECISION/RECALL @ TOP-K (reviewer workflow view)")
    print("=" * 70)
    topk = top_k_precision_recall(df)
    print(topk.to_string(index=False))

    print()
    print("=" * 70)
    print(f"RECALL BY ANOMALY TYPE @ threshold={t:.4f}")
    print("=" * 70)
    print(per_anomaly_type_recall(df, t))

    print()
    ap = average_precision_score(df["is_anomaly"], df["suspicion_score"])
    print(f"Average Precision (area under PR curve): {ap:.3f}")
