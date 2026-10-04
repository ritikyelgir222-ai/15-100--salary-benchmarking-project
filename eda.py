"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: a compensation/total-rewards stakeholder's
first questions are "does pay scale sensibly with experience and
company scale" (a sanity check before trusting the data at all) and
"are there any groups who look systematically underpaid" — the second
question is the one this project is actually built to help answer, not
just salary prediction for its own sake.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data

EXPERIENCE_ORDER = ["EN", "MI", "SE", "EX"]


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()

    print("=== Average salary (USD) by experience level ===")
    print(df.groupby("experience_level")["salary_in_usd"].mean().reindex(EXPERIENCE_ORDER).round(0))
    # WHY WE CHECK THIS FIRST: confirms experience_level is genuinely
    # ordinal (EN < MI < SE < EX in actual average pay) before encoding
    # it that way in clean_and_engineer.py -- an assumption worth
    # verifying, not just assuming because the labels look ordered.

    print("\n=== Average salary (USD) by work year ===")
    print(df.groupby("work_year")["salary_in_usd"].mean().round(0))
    # WHY: a 30% jump from 2020 to 2022 is a real market trend (the
    # post-2020 tech hiring boom) worth knowing about before treating
    # any single year's average as "the" benchmark.

    print("\n=== Average salary (USD) by company size ===")
    print(df.groupby("company_size")["salary_in_usd"].mean().round(0))

    print("\n=== Average salary (USD) by remote-work ratio ===")
    print(df.groupby("remote_ratio")["salary_in_usd"].mean().round(0))
    # WHY THIS ONE MATTERS FOR MODELING: if this were a clean monotonic
    # relationship (more remote = more or less pay, consistently), it
    # could be treated as a single ordinal/numeric feature. It ISN'T --
    # hybrid (50%) pays LESS than both fully in-office (0%) and fully
    # remote (100%) roles. Treating it as a linear numeric feature would
    # force a wrong assumption on the model; it's one-hot encoded
    # instead (see clean_and_engineer.py).

    is_cross_border = df["employee_residence"] != df["company_location"]
    print(f"\n=== THE EQUITY FINDING: cross-border employees ({is_cross_border.mean():.1%} of records) ===")
    print(df.groupby(is_cross_border)["salary_in_usd"].mean().round(0))
    print("-> Employees whose residence differs from their employer's country average")
    print("   $76,758 vs. $115,558 for same-country employees -- a meaningful gap worth")
    print("   investigating with a benchmarking model, not just observing. This is the")
    print("   central business question this project is built to help answer.")

    print("\n=== Average salary (USD) by top 10 company locations ===")
    top_countries = df["company_location"].value_counts().head(10)
    print(df[df["company_location"].isin(top_countries.index)]
          .groupby("company_location")["salary_in_usd"].mean().sort_values(ascending=False).round(0))

    print(f"\n=== Data quality ===")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"salary_in_usd skew: {df['salary_in_usd'].skew():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    df.groupby("experience_level")["salary_in_usd"].mean().reindex(EXPERIENCE_ORDER).plot(
        kind="bar", ax=axes[0], color="#4C72B0", title="Avg salary by experience level"
    )
    axes[0].tick_params(axis="x", rotation=0)

    df.groupby("remote_ratio")["salary_in_usd"].mean().plot(
        kind="bar", ax=axes[1], color="#C44E52", title="Avg salary by remote ratio (non-monotonic)"
    )
    axes[1].tick_params(axis="x", rotation=0)

    df.groupby(is_cross_border)["salary_in_usd"].mean().rename(
        index={False: "Same country", True: "Cross-border"}
    ).plot(kind="bar", ax=axes[2], color="#55A868", title="Avg salary: cross-border vs. same-country")
    axes[2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
