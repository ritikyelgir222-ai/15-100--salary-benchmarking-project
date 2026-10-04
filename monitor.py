"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index): the standard, interview-
recognizable metric for feature drift, with established severity
thresholds (< 0.1 stable, 0.1-0.25 moderate, > 0.25 significant).

WHY THIS SCRIPT SIMULATES A "NEW BATCH": this dataset is a single
2020-2022 snapshot with no natural second period to compare against --
the same limitation as the churn/attrition/insurance projects in this
series. The held-out test set is reused as a stand-in, which confirms
the monitoring code path works, not that real market drift is caught.

WHAT A REAL DEPLOYMENT SHOULD ALSO TRACK: salary benchmarks need
refreshing more often than most models in this series -- the market
itself moves (EDA found a 30% jump from 2020 to 2022 alone). A stale
salary model isn't just less accurate, it's actively misleading for
compensation decisions.
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import pct_within_tolerance

ACCURACY_DROP_THRESHOLD = 0.05
DRIFT_FEATURES = ["experience_level_ord", "is_cross_border", "work_year", "company_size_ord"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/salary_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    X = engineered[feature_cols]
    salary = engineered["salary_in_usd"]
    y = np.log1p(salary)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, salary_for_stratify=salary)
    salary_test = np.expm1(y_test)

    print("=== Feature Drift (PSI): train distribution vs. held-out batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    pred_salary = np.expm1(model.predict(X_test))
    pct_within = pct_within_tolerance(salary_test, pred_salary)

    print(f"\n=== Performance on held-out batch ===")
    print(f"% within +/-20%: {pct_within:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_pct_within = json.load(f)["test_pct_within_20pct"]

    drop = baseline_pct_within - pct_within
    if drop > ACCURACY_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: pct-within-20% dropped by {drop:.3f} (threshold: {ACCURACY_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (pct-within-20% change: {drop:.3f}, threshold: {ACCURACY_DROP_THRESHOLD})")

    print("\nNOTE: a real deployment should refresh this model at least annually --")
    print("EDA found a 30% average salary increase from 2020 to 2022 alone. A stale")
    print("salary benchmark doesn't just lose accuracy, it actively misleads")
    print("compensation decisions in a period of rising market pay.")


if __name__ == "__main__":
    run_monitoring_check()
