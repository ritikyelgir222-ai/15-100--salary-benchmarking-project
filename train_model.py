"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
Every modeling choice below has a WHY comment. This project's Phase 9 is
unusual in this series: the business validation isn't just "how accurate
is the model" -- it's "does the model's benchmark expose a real pay
equity question," directly following up on the cross-border pay gap EDA
found.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

SALARY_TOLERANCE = 0.20  # same domain-appropriate looseness as the insurance-cost project -- salary varies a lot by geography/role beyond what 6 features can fully capture
UNDERPAID_THRESHOLD = 0.80  # flagged as a pay-equity review candidate if actual pay is < 80% of the model's benchmark


def pct_within_tolerance(y_true, y_pred, tolerance=SALARY_TOLERANCE):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pct_error = np.abs(y_pred - y_true) / y_true
    return float((pct_error <= tolerance).mean())


def train_and_evaluate():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    X = engineered[feature_cols]
    salary = engineered["salary_in_usd"]
    y = np.log1p(salary)  # EDA found salary right-skewed (skew ~1.67)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, salary_for_stratify=salary)
    salary_train = np.expm1(y_train)
    salary_val = np.expm1(y_val)
    salary_test = np.expm1(y_test)

    experiment_log = []

    # -----------------------------------------------------------------
    # Baseline: Linear Regression on 2 features
    # WHY THESE 2: experience_level_ord and is_cross_border are the two
    # single strongest, most obvious signals from EDA -- an honest floor
    # before reaching for a more complex model.
    # -----------------------------------------------------------------
    baseline_feats = ["experience_level_ord", "is_cross_border"]
    scaler = StandardScaler()
    Xb_train = scaler.fit_transform(X_train[baseline_feats])
    Xb_val = scaler.transform(X_val[baseline_feats])

    lr = LinearRegression()
    lr.fit(Xb_train, y_train)
    lr_val_salary = np.expm1(lr.predict(Xb_val))
    experiment_log.append({
        "model": "linear_regression (baseline, experience + cross-border)",
        "val_mae": round(mean_absolute_error(salary_val, lr_val_salary), 2),
        "val_pct_within_20pct": round(pct_within_tolerance(salary_val, lr_val_salary), 4),
    })

    # -----------------------------------------------------------------
    # Candidate: Random Forest
    # WHY TRIED: captures interactions (e.g. the experience/company-size
    # combination, or how the remote_ratio U-shape interacts with
    # location) beyond what the linear baseline can represent.
    # -----------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_salary = np.expm1(rf.predict(X_val))
    experiment_log.append({
        "model": "random_forest",
        "val_mae": round(mean_absolute_error(salary_val, rf_val_salary), 2),
        "val_pct_within_20pct": round(pct_within_tolerance(salary_val, rf_val_salary), 4),
    })

    # -----------------------------------------------------------------
    # Final candidate: XGBoost
    # SELECTION CRITERION: chosen by val_pct_within_20pct (the
    # business-relevant metric), consistent with the rest of this series
    # -- not assumed to win by default (see Day 10, where Random Forest
    # won instead). Whichever wins here is selected programmatically.
    # -----------------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_val_salary = np.expm1(xgb.predict(X_val))
    xgb_val_mae = mean_absolute_error(salary_val, xgb_val_salary)
    xgb_val_pct_within = pct_within_tolerance(salary_val, xgb_val_salary)
    experiment_log.append({
        "model": "xgboost",
        "val_mae": round(xgb_val_mae, 2),
        "val_pct_within_20pct": round(xgb_val_pct_within, 4),
    })

    print("=== Experiment Log (validation set) ===")
    log_df = pd.DataFrame(experiment_log)
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    rf_val_pct = experiment_log[1]["val_pct_within_20pct"]
    if rf_val_pct >= xgb_val_pct_within:
        final_model, final_name = rf, "random_forest"
        final_val_pct = rf_val_pct
    else:
        final_model, final_name = xgb, "xgboost"
        final_val_pct = xgb_val_pct_within
    print(f"\n=== Selected final model: {final_name} (higher val_pct_within_20pct) ===")

    # -----------------------------------------------------------------
    # Phase 9a: Standard evaluation on the held-out TEST set
    # -----------------------------------------------------------------
    test_pred_salary = np.expm1(final_model.predict(X_test))
    test_mae = mean_absolute_error(salary_test, test_pred_salary)
    test_mape = mean_absolute_percentage_error(salary_test, test_pred_salary)
    test_r2 = r2_score(salary_test, test_pred_salary)
    test_pct_within = pct_within_tolerance(salary_test, test_pred_salary)

    # -----------------------------------------------------------------
    # Phase 9b: THE EQUITY-FLAGGING ANALYSIS -- the real business
    # validation for this project.
    # WHY THIS IS THE ACTUAL DELIVERABLE, NOT JUST ACCURACY METRICS: the
    # BRD for this project isn't "predict salary as accurately as
    # possible" in the abstract -- it's "support fair, data-backed
    # compensation decisions." EDA found cross-border employees earn
    # $76,758 vs. $115,558 on average. This step checks whether that gap
    # holds up as a MODEL-FLAGGED equity signal (comparing actual pay
    # against each employee's OWN experience/location/company-size-
    # adjusted benchmark, not just the raw group average) -- a more
    # rigorous, actionable version of the raw EDA finding, since it
    # controls for the legitimate reasons two employees might be paid
    # differently (different experience levels, different company
    # sizes) before flagging a gap as potentially inequitable.
    # -----------------------------------------------------------------
    engineered_test = engineered.loc[X_test.index].copy()
    engineered_test["predicted_salary"] = test_pred_salary
    engineered_test["underpaid_flag"] = (
        engineered_test["salary_in_usd"] < UNDERPAID_THRESHOLD * engineered_test["predicted_salary"]
    )

    underpaid_rate_cross_border = float(
        engineered_test.loc[engineered_test["is_cross_border"] == 1, "underpaid_flag"].mean()
    ) if (engineered_test["is_cross_border"] == 1).sum() > 0 else None
    underpaid_rate_same_country = float(
        engineered_test.loc[engineered_test["is_cross_border"] == 0, "underpaid_flag"].mean()
    )
    n_cross_border_test = int((engineered_test["is_cross_border"] == 1).sum())

    business_summary = {
        "final_model": final_name,
        "test_mae": round(test_mae, 2),
        "test_mape": round(test_mape, 4),
        "test_r2": round(test_r2, 4),
        "test_pct_within_20pct": round(test_pct_within, 4),
        "n_test_employees": len(salary_test),
        "n_cross_border_in_test": n_cross_border_test,
        "underpaid_rate_cross_border": round(underpaid_rate_cross_border, 4) if underpaid_rate_cross_border is not None else None,
        "underpaid_rate_same_country": round(underpaid_rate_same_country, 4),
        "note": (
            "underpaid_flag = actual salary < 80% of the model's own "
            "experience/location/company-size-adjusted benchmark for that "
            "employee -- a more rigorous check than the raw EDA group averages, "
            "since it controls for legitimate pay-driving factors before "
            "flagging a gap as a potential equity concern. n_cross_border_in_test "
            f"is small ({n_cross_border_test} of {len(salary_test)} test rows), so "
            "treat the underpaid-rate comparison as directional, not statistically "
            "definitive, on a dataset this size."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n=== Top 10 Feature Importances (final model) ===")
        print(importances.head(10).round(4).to_string())
        importances.to_csv("outputs/feature_importance.csv", header=["importance"])

    joblib.dump(final_model, "outputs/salary_model.joblib")
    joblib.dump(feature_cols, "outputs/feature_columns.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": "RandomForestRegressor" if final_name == "random_forest" else "XGBRegressor",
            "data_source": "Data Science Job Salaries (ai-jobs.net aggregation, 2020-2022)",
            "n_features": len(feature_cols),
            "features": feature_cols,
            "training_rows": len(X_train),
            "target_transform": "log1p(salary_in_usd)",
            "validation_pct_within_20pct": round(final_val_pct, 4),
            "test_pct_within_20pct": round(test_pct_within, 4),
            "intended_use": "Benchmark expected salary for a role/experience/location/company-size combination, and flag employees whose actual pay falls notably below their benchmark for pay-equity review.",
            "known_limitations": (
                "Small dataset (607 rows), US-dominated (58% of records), "
                "2020-2022 only -- salary levels since then have moved "
                "(inflation, AI-role demand). The underpaid-flag analysis "
                "uses only 8.4% cross-border representation in the full "
                "dataset, an even smaller absolute count in the test split -- "
                "directional, not a statistically powered equity audit."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/salary_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
