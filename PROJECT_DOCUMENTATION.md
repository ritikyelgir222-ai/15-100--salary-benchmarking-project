# Project Documentation: Salary Benchmarking Model
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. Every number below came from actually running the
code in this repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Benchmark expected salary by role/experience/location/company size, and flag pay-equity review candidates |
| **Client context** | HR / Total Rewards (compensation) team |
| **Data source** | Data Science Job Salaries — 607 real records, 2020-2022 |
| **Final model** | Random Forest Regressor, 27 features (chosen programmatically over XGBoost) |
| **Test accuracy** | R² = 0.3545, 43.48% within ±20% |
| **Business framing** | Support fair, data-backed compensation decisions — not just predict a number |

---

## 2. Architecture

```
data/ds_salaries.csv
        │
        ▼
data_loader.py ──────► loads raw CSV
        │
        ▼
clean_and_engineer.py ─► drops raw local-currency columns (leakage
        │                risk), ordinal-encodes experience_level &
        │                company_size (verified ordinal in EDA),
        │                one-hot encodes remote_ratio (verified NOT
        │                ordinal), groups 50 job titles into families,
        │                engineers is_cross_border
        ▼
   split.py ───────────► salary-quartile-stratified train/val/test split
        │
        ▼
train_model.py ───────► baseline, Random Forest, XGBoost;
        │                PROGRAMMATIC selection; equity-flagging analysis
        ▼
outputs/salary_model.joblib, feature_columns.joblib
        │
        └──────────────► app.py ─── FastAPI /benchmark endpoint,
                                     returns benchmark + equity flag
```

---

## 3. Data Dictionary & Encoding Decisions

| Raw column | Decision | Verified reasoning |
|---|---|---|
| `salary`, `salary_currency` | **Dropped** | Local-currency amount is not comparable across countries and leaks target information; `salary_in_usd` is the correct standardized target |
| `experience_level` | Ordinal (1-4) | EDA confirmed average pay rises monotonically EN ($61.6K) → MI ($88K) → SE ($138.6K) → EX ($199.4K) |
| `company_size` | Ordinal (1-3) | EDA confirmed S ($77.6K) < M ($116.9K) < L ($119.2K) |
| `employment_type` | Binary `is_full_time` | 96.9% of records are FT; the remaining 3 types combined are only 19 rows |
| `remote_ratio` | **One-hot** (not ordinal, despite being numeric) | EDA found a non-monotonic U-shape: hybrid (50%, $80.8K) pays LESS than both in-office (0%, $106.4K) and fully remote (100%, $122.5K) |
| `job_title` (50 unique) | Grouped into ~7 families | Too few examples per individual title (607 rows total) to support reliable one-hot categories |
| `company_location` (many countries) | Top 10 + "Other" | Top 10 covers 88% of records |
| — | **Engineered: `is_cross_border`** | `employee_residence != company_location` — directly motivated by the EDA equity finding |

---

## 4. Key EDA Findings

| Cut | Finding |
|---|---|
| Experience level | EN $61,643 → MI $87,996 → SE $138,617 → EX $199,392 — clean monotonic progression |
| Work year | 2020: $95,813 → 2021: $99,854 → 2022: $124,522 — a real ~30% market increase over 2 years |
| Company size | S $77,633 < M $116,905 < L $119,243 |
| **Remote ratio (non-monotonic)** | 0%: $106,355 · 50%: $80,823 · 100%: $122,457 — hybrid pays least |
| **Cross-border pay gap (the central finding)** | Same-country: $115,558 · Cross-border: $76,758 (8.4% of records) |
| Company location (top) | US $144,055 far exceeds every other country; India lowest among the top 10 at $28,582 |
| Target skew | 1.668 (right-skewed) — log1p-transformed for modeling |

---

## 5. Modeling Results

### Experiment log (validation set)

| Model | Val MAE | Val % within ±20% |
|---|---|---|
| Linear Regression (baseline: experience + cross-border) | $38,772 | 28.57% |
| **Random Forest** | **$23,325** | **47.25%** |
| XGBoost | $25,054 | 46.15% |

Random Forest won on both metrics this run — selected programmatically,
consistent with this series' "let the code decide" principle (see Day
10, where the same pattern first appeared).

### Held-out test set

| Metric | Value |
|---|---|
| Test MAE | $38,184 |
| Test MAPE | 41.89% |
| Test R² | 0.3545 |
| Test % within ±20% | 43.48% |
| N test employees | 92 |

**This accuracy is modest, and that's stated honestly rather than
dressed up.** With only 6 base features (experience, year, company
size/location, remote ratio, job family) and no data on individual
skills, negotiation outcomes, or company-specific pay bands, a large
share of real salary variance is simply not observable to this model.
The R² of 0.35 and 43.5% within-tolerance rate are lower than most other
regression projects in this series (e.g. Day 8's insurance-cost model
hit 69.65% within ±20%) — a genuine reflection of how much harder
individual salary is to predict from limited demographic/role data than,
say, individual healthcare cost.

### What drives the model (feature importance)

Top 5 by importance:

1. `location_group_US` (0.3974) — by far the strongest driver
2. `experience_level_ord` (0.1239)
3. `location_group_IN` (0.0968)
4. `is_cross_border` (0.0739)
5. `location_group_Other` (0.0535)

Notably, `is_cross_border` ranks 4th on its own — a meaningful signal
even alongside the country-location features it's partly correlated
with.

### The equity-flagging business validation (the real deliverable)

On the 92-employee held-out test set, each employee's actual salary was
compared against the model's own experience/location/company-size-
adjusted benchmark for their specific profile (not just a raw group
average):

| Group | Underpaid-flag rate (actual < 80% of benchmark) | N |
|---|---|---|
| Cross-border employees | **28.57%** | 7 |
| Same-country employees | 20.00% | 85 |

This is directionally consistent with the raw EDA finding, and it's a
MORE rigorous check than the raw group averages — it controls for
experience level, company size, and location before flagging a gap,
rather than comparing unadjusted averages across two very different
populations. **However, with only 7 cross-border employees in the test
set, this is explicitly reported as directional, not statistically
conclusive** — the honest caveat that belongs on this finding before any
real compensation decision gets made from it.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Test form — benchmark-only or benchmark + equity check |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/benchmark` | POST | Get a salary benchmark, optionally flag against an actual salary |

**Response (verified, two contrasting real profiles):**
```json
{
  "benchmark_salary_usd": 223363.16,
  "actual_salary_usd": null,
  "flagged_for_equity_review": null,
  "top_factors": ["location_group_US", "experience_level_ord", "location_group_IN"]
}
```
A cross-border mid-level Data Scientist profile (India-based, US
company) with a stated actual salary of $45,000 benchmarked at
**$58,448** and correctly returned `flagged_for_equity_review: true`.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` implements PSI drift checking and an accuracy-drop retrain
trigger, with the same same-snapshot-stand-in limitation as the churn,
attrition, and insurance projects in this series (no genuine second time
period exists in this dataset). **The more important point for THIS
project specifically**: EDA found a real 30% salary increase from 2020
to 2022 — a salary benchmarking model that isn't refreshed regularly
doesn't just get less accurate, it actively misleads compensation
decisions during a period of rising market pay. This is called out
directly in the script's output, not left implicit.

---

## 8. Known Limitations (stated for the handover record)

1. **Small (607 rows) and US-dominated (58%)** — limited signal for
   non-US locations specifically.
2. **2020-2022 only** — needs annual refresh; see Section 7.
3. **Modest accuracy (R²=0.35)** — an honest limitation of a 6-feature
   model for a target with this much unobserved variance, not a
   modeling error.
4. **The equity-flagging comparison is statistically underpowered** — 7
   cross-border employees in the test set is directional evidence, not
   a conclusive audit.
5. **`job_title` grouping into families is a simplification** — a real
   compensation system would benchmark against much more granular role
   taxonomies and internal leveling frameworks.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document source |
| `eda.py` | 6 | Experience/year/size/remote-ratio cuts, the equity finding |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Encoding decisions (each checked against EDA), job-family grouping |
| `split.py` | 7 | Salary-quartile-stratified split |
| `train_model.py` | 8-9 | Model training, programmatic selection, **equity-flagging validation** |
| `app.py` | 10 | FastAPI benchmarking service |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
