# Salary Benchmarking Model — Full Project Code (Real Dataset)

Companion code to Day 15 of the 100-day LinkedIn series. Uses the real
"Data Science Job Salaries" dataset, run end to end — every number in
`PROJECT_DOCUMENTATION.md` came from executing this code.

## Data source

**Data Science Job Salaries** — 607 real, crowd-sourced salary records
for data-field roles (2020-2022), aggregated by ai-jobs.net and widely
mirrored on Kaggle. The CSV is already included at `data/ds_salaries.csv`.

## The real business question this project answers

Most salary-prediction tutorials stop at "predict the number." This
project is built around a sharper question, directly motivated by an
EDA finding: **employees whose residence differs from their employer's
country average $76,758/year, vs. $115,558 for same-country employees.**
That's a real pay-equity signal worth investigating, not just observing.
`train_model.py` doesn't stop at accuracy metrics — it uses the model to
flag employees whose actual pay falls notably below their own
experience/location/company-size-adjusted benchmark, then checks whether
cross-border employees are disproportionately flagged. On the (small)
test set: **28.6% of cross-border employees flagged vs. 20.0% of
same-country employees** — directionally consistent with the raw EDA
finding, though with only 7 cross-border employees in the test set,
this is explicitly reported as directional, not statistically
conclusive.

## Why every script is commented the way it is

- **`experience_level` is ordinal-encoded, `bmi_category` (from the
  insurance project) was one-hot encoded** — a deliberate contrast. Here,
  EDA verified actual average pay rises monotonically EN → MI → SE → EX;
  ordinal encoding is checked against the data, not assumed from the
  label names looking ordered.
- **`remote_ratio` is one-hot encoded despite looking numeric** — EDA
  found hybrid (50%) roles pay LESS than both fully in-office (0%) and
  fully remote (100%) roles, a non-monotonic U-shape a linear/ordinal
  encoding would get wrong.
- **50 job titles are grouped into ~7 families** rather than one-hot
  encoded individually — with only 607 rows total, most individual
  titles have too few examples to support their own category reliably.
- **Model selection is genuinely programmatic** — Random Forest won this
  run (0.4725 vs. XGBoost's 0.4615 on val pct-within-tolerance), the
  same "let the data decide" pattern as Day 10's demand-forecasting
  project.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & inspects the raw CSV
python eda.py                  # Phase 6 — experience/year/size/remote-ratio cuts + the equity finding + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — encoding, job-family grouping, engineered features
python train_model.py          # Phase 8-9 — baseline + final model, the equity-flagging business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
        "experience_level": "MI", "job_title": "Data Scientist",
        "employee_residence": "IN", "company_location": "US",
        "remote_ratio": 100, "company_size": "M",
        "actual_salary_usd": 45000
      }'
```

Expected: a benchmark around **$58,448** and `"flagged_for_equity_review":
true` — the $45K actual falls well below 80% of the model's
location/experience-adjusted benchmark for this profile.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the real dataset, documents its source |
| `eda.py` | 6 | Experience/year/size/remote-ratio cuts, **the cross-border pay-equity finding** |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Ordinal vs. one-hot encoding decisions (each checked against EDA), job-family grouping |
| `split.py` | 7 | Salary-quartile-stratified train/val/test split |
| `train_model.py` | 8-9 | Baseline → Random Forest → XGBoost, **programmatic selection**, equity-flagging business validation |
| `app.py` | 10 | FastAPI benchmarking service with equity-flag output |
| `monitor.py` | 13 | PSI drift check + retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`, `eda_summary.png`, `experiment_log.csv`
- `business_validation.json` — accuracy metrics AND the equity-flag comparison
- `feature_importance.csv`
- `salary_model.joblib`, `feature_columns.joblib`, `model_card.json`

## Known limitations (stated honestly, not hidden)

- **Small dataset (607 rows), US-dominated (58%)** — the model has
  comparatively little to learn from for other countries.
- **2020-2022 only** — EDA found a 30% average salary jump from 2020 to
  2022 alone; this model needs annual refreshing to stay meaningful.
- **Modest predictive accuracy** (test R² = 0.35, 43.5% within ±20%) —
  an honest reflection that 6 features (no specific skills, negotiation,
  or company-level pay bands) leave a lot of real salary variance
  unexplained. This is stated directly rather than picking a looser
  tolerance to make the number look better.
- **The equity-flagging comparison uses only 7 cross-border employees in
  the test set** — directionally consistent with the EDA finding, not a
  statistically powered audit. A real pay-equity review needs a much
  larger, ideally company-specific dataset.
