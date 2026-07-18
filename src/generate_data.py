"""
generate_data.py
-----------------
Generates a synthetic construction/infrastructure ERP expense dataset for
Rexo ERP's anomaly detection module.

Design goals:
1. Realistic *normal* data: each vendor has a characteristic category and a
   stable spending distribution (lognormal, since real-world expense amounts
   are right-skewed — lots of small POs, a few big ones).
2. Five clearly defined, independently injected anomaly types, each with an
   explicit ground-truth label (`is_anomaly`, `anomaly_type`). This is what
   lets us compute real precision/recall later instead of eyeballing results.
3. Anomalies are injected as a *final pass* on top of an otherwise clean
   dataset, so the "normal" data never accidentally contains the anomaly
   pattern we're trying to detect (avoids leaking easy signal).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Reference data: vendors, categories, sites, approvers
# ---------------------------------------------------------------------------

CATEGORIES = ["materials", "labor", "equipment", "misc"]

SITES = [f"SITE-{i:03d}" for i in range(1, 11)]  # 10 project/site IDs

APPROVERS = [
    "R. Mehta", "S. Kulkarni", "A. Fernandes", "P. Singh",
    "N. Rao", "J. D'Souza", "K. Iyer", "M. Sharma",
]

# Each vendor specializes in one category and has a characteristic spend
# profile: (mean, std) of the underlying normal distribution IN LOG SPACE.
# This produces a right-skewed (lognormal) distribution of amounts, which is
# typical for real procurement data — many small/medium POs, a long tail of
# large ones.
VENDOR_PROFILES = {
    # materials vendors
    "Shree Cement Suppliers":      ("materials",  (10.5, 0.45)),   # ~₹36k median
    "BuildRight Steel & Rebar":    ("materials",  (11.2, 0.5)),    # ~₹73k median
    "Konkan Aggregates Pvt Ltd":   ("materials",  (10.1, 0.4)),
    "Goa Timber & Plywood":        ("materials",  (9.8, 0.4)),
    # labor / manpower contractors
    "Everest Manpower Contractors":("labor",       (10.8, 0.35)),
    "Sagar Labour Solutions":      ("labor",       (10.4, 0.35)),
    "Union Skilled Workforce":     ("labor",       (11.0, 0.4)),
    # equipment rental
    "PowerLift Cranes & Equipment":("equipment",   (11.8, 0.5)),
    "TerraMove Heavy Machinery":   ("equipment",   (12.0, 0.55)),
    "Coastal Scaffolding Rentals": ("equipment",   (10.6, 0.4)),
    # misc (site admin, utilities, consumables, permits)
    "Zenith Office & Site Supplies":("misc",       (8.8, 0.5)),
    "SafeSite PPE & Consumables":  ("misc",        (9.0, 0.45)),
    "Municipal Permits & Fees":    ("misc",        (9.6, 0.3)),
}

VENDORS = list(VENDOR_PROFILES.keys())

# Each vendor is typically approved by 2-3 "regular" approvers (mirrors real
# org structure: procurement is usually routed to a small set of people per
# vendor/category). This regularity is what lets us later flag an approver
# who suddenly approves for a vendor they've never touched.
VENDOR_REGULAR_APPROVERS = {
    v: list(RNG.choice(APPROVERS, size=RNG.integers(2, 4), replace=False))
    for v in VENDORS
}

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 6, 30)
BUSINESS_HOURS = list(range(8, 19))  # 8 AM - 6 PM


def random_business_datetime():
    """A timestamp on a weekday, during business hours -- 'normal' behavior."""
    span_days = (END_DATE - START_DATE).days
    while True:
        d = START_DATE + timedelta(days=int(RNG.integers(0, span_days)))
        if d.weekday() < 5:  # Mon-Fri
            hour = int(RNG.choice(BUSINESS_HOURS))
            minute = int(RNG.integers(0, 60))
            return d.replace(hour=hour, minute=minute)


def gen_normal_record(record_id):
    vendor = RNG.choice(VENDORS)
    category, (mu, sigma) = VENDOR_PROFILES[vendor]
    amount = round(float(RNG.lognormal(mu, sigma)), 2)
    approver = RNG.choice(VENDOR_REGULAR_APPROVERS[vendor])
    return {
        "record_id": record_id,
        "vendor": vendor,
        "category": category,
        "amount": amount,
        "site_id": RNG.choice(SITES),
        "approver": approver,
        "submitted_at": random_business_datetime(),
        "is_anomaly": 0,
        "anomaly_type": "none",
    }


def generate_normal_dataset(n=1800):
    records = [gen_normal_record(i) for i in range(n)]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Anomaly injection
# ---------------------------------------------------------------------------
# Each function takes the clean df and the running next-id counter, and
# returns a list of new anomaly rows. Anomalies are NEW rows appended to the
# dataset (mirroring reality: a fraudulent/erroneous entry is itself a row
# in the ledger), not modifications of existing normal rows -- except
# "duplicate" and "unusual_approver", which by definition reference an
# existing legitimate transaction.

def inject_duplicates(df, next_id, n=30):
    """Same vendor+amount+site re-submitted within a few days -- classic
    double-entry / duplicate-invoice error or intentional double billing."""
    base_rows = df.sample(n=n, random_state=1).to_dict("records")
    new_rows = []
    for i, row in enumerate(base_rows):
        dup = row.copy()
        dup["record_id"] = next_id + i
        # Re-submitted 0-3 days later, small time-of-day jitter
        delta_days = int(RNG.integers(0, 4))
        dup["submitted_at"] = row["submitted_at"] + timedelta(
            days=delta_days, minutes=int(RNG.integers(-30, 30))
        )
        dup["is_anomaly"] = 1
        dup["anomaly_type"] = "duplicate"
        new_rows.append(dup)
    return new_rows


def inject_round_numbers(df, next_id, n=25):
    """Suspiciously 'round' amounts (multiples of 25k/50k/100k) are a classic
    red flag: real invoices rarely land on exact round numbers; round
    numbers often indicate estimated/fabricated figures."""
    base_rows = df.sample(n=n, random_state=2).to_dict("records")
    round_units = [25000, 50000, 100000]
    new_rows = []
    for i, row in enumerate(base_rows):
        r = row.copy()
        r["record_id"] = next_id + i
        unit = int(RNG.choice(round_units))
        multiplier = int(RNG.integers(2, 8))
        r["amount"] = float(unit * multiplier)
        r["submitted_at"] = random_business_datetime()
        r["is_anomaly"] = 1
        r["anomaly_type"] = "round_number"
        new_rows.append(r)
    return new_rows


def inject_extreme_outliers(df, next_id, n=30):
    """Amount far outside the vendor/category's typical range (6-12x the
    vendor's median) -- e.g. data entry error (extra zero) or inflated
    billing."""
    base_rows = df.sample(n=n, random_state=3).to_dict("records")
    new_rows = []
    for i, row in enumerate(base_rows):
        r = row.copy()
        r["record_id"] = next_id + i
        multiplier = float(RNG.uniform(6, 12))
        r["amount"] = round(row["amount"] * multiplier, 2)
        r["submitted_at"] = random_business_datetime()
        r["is_anomaly"] = 1
        r["anomaly_type"] = "extreme_outlier"
        new_rows.append(r)
    return new_rows


def inject_off_hours(df, next_id, n=30):
    """Entries submitted at odd hours (1-4 AM) or on weekends -- unusual for
    a legitimate site/procurement workflow, common in ERP fraud heuristics."""
    base_rows = df.sample(n=n, random_state=4).to_dict("records")
    new_rows = []
    for i, row in enumerate(base_rows):
        r = row.copy()
        r["record_id"] = next_id + i
        span_days = (END_DATE - START_DATE).days
        d = START_DATE + timedelta(days=int(RNG.integers(0, span_days)))
        if RNG.random() < 0.5:
            # weekend, normal hours
            while d.weekday() < 5:
                d += timedelta(days=1)
            hour = int(RNG.choice(BUSINESS_HOURS))
        else:
            # weekday, odd hour
            hour = int(RNG.choice([0, 1, 2, 3, 4, 23]))
        r["submitted_at"] = d.replace(hour=hour, minute=int(RNG.integers(0, 60)))
        r["is_anomaly"] = 1
        r["anomaly_type"] = "off_hours"
        new_rows.append(r)
    return new_rows


def inject_unusual_approver(df, next_id, n=35):
    """A high-value transaction approved by someone who is NOT in that
    vendor's normal approver set -- classic segregation-of-duties red flag
    (approval routed around the usual controls)."""
    base_rows = df.sample(n=n, random_state=5).to_dict("records")
    new_rows = []
    for i, row in enumerate(base_rows):
        r = row.copy()
        r["record_id"] = next_id + i
        vendor = r["vendor"]
        regulars = set(VENDOR_REGULAR_APPROVERS[vendor])
        outsiders = [a for a in APPROVERS if a not in regulars]
        r["approver"] = RNG.choice(outsiders) if outsiders else RNG.choice(APPROVERS)
        # Bump amount up (2-4x) so it's also a *high-value* unusual approval,
        # which is the realistic risk scenario finance teams care about.
        r["amount"] = round(row["amount"] * float(RNG.uniform(2, 4)), 2)
        r["submitted_at"] = random_business_datetime()
        r["is_anomaly"] = 1
        r["anomaly_type"] = "unusual_approver"
        new_rows.append(r)
    return new_rows


def generate_full_dataset():
    normal_df = generate_normal_dataset(n=1800)
    next_id = normal_df["record_id"].max() + 1

    anomaly_rows = []
    for fn, n in [
        (inject_duplicates, 30),
        (inject_round_numbers, 25),
        (inject_extreme_outliers, 30),
        (inject_off_hours, 30),
        (inject_unusual_approver, 35),
    ]:
        rows = fn(normal_df, next_id, n=n)
        anomaly_rows.extend(rows)
        next_id += n

    anomaly_df = pd.DataFrame(anomaly_rows)
    full_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    # Shuffle so anomalies aren't trivially clustered at the end of the file
    full_df = full_df.sample(frac=1, random_state=7).reset_index(drop=True)
    full_df["record_id"] = range(1, len(full_df) + 1)  # reassign clean IDs
    return full_df


if __name__ == "__main__":
    df = generate_full_dataset()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "expenses.csv"), index=False)
    print(f"Generated {len(df)} records "
          f"({df['is_anomaly'].sum()} anomalies, "
          f"{df['is_anomaly'].mean()*100:.1f}% anomaly rate)")
    print(df["anomaly_type"].value_counts())
