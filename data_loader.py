"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "Data Science Job Salaries" dataset — real,
crowd-sourced salary records for data-field roles collected 2020-2022,
originally aggregated on ai-jobs.net and widely mirrored on Kaggle:
    https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries

607 records covering experience level, employment type, job title,
salary (in both local currency and standardized USD), employee
residence, remote-work ratio, company location, and company size.

If you're following along on Kaggle: download `ds_salaries.csv` from the
link above and place it at `data/ds_salaries.csv` — the schema is
identical to the file already included in this project.

WHY THIS DATASET
-----------------
- It has a real, standardized dollar target (`salary_in_usd`) already
  converted from each record's local currency, which matters for the
  same reason it mattered in Day 8's insurance project: business
  validation figures can be computed directly from real data rather
  than needing an externally-assumed conversion.
- It has a genuinely ordinal feature (`experience_level`: EN < MI < SE <
  EX) — a useful contrast with Day 8's `bmi_category`, which LOOKED
  ordinal but was one-hot encoded because the smoker interaction wasn't
  linear across categories. Here, EDA confirms the progression really
  is close to monotonic (see eda.py), so ordinal encoding is the
  correct choice this time, not a shortcut.
- It surfaces a genuine pay-equity question directly relevant to the
  "fair, data-backed compensation decisions" framing for this project:
  employees whose residence differs from their employer's location earn
  meaningfully less on average (see eda.py) — worth investigating with
  a benchmarking model, not just predicting salary for its own sake.
"""

import pandas as pd

RAW_DATA_PATH = "data/ds_salaries.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"\nsalary_in_usd summary:")
    print(df["salary_in_usd"].describe())
    print(f"\nsalary_in_usd skew: {df['salary_in_usd'].skew():.3f} (right-skewed — see clean_and_engineer.py)")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
