"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it. FINAL FEATURE LIST is built at the bottom as
FEATURE_COLUMNS.
"""

import pandas as pd

EXPERIENCE_MAP = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}  # verified genuinely ordinal in eda.py
COMPANY_SIZE_MAP = {"S": 1, "M": 2, "L": 3}
TOP_N_COUNTRIES = 10

ONE_HOT_COLUMNS = ["remote_ratio", "job_family", "location_group"]

JOB_FAMILY_KEYWORDS = [
    ("Data Scientist", "Data Scientist"),
    ("Data Engineer", "Data Engineer"),
    ("Data Analyst", "Data Analyst"),
    ("Machine Learning", "ML Engineer/Scientist"),
    ("ML ", "ML Engineer/Scientist"),
    ("AI ", "ML Engineer/Scientist"),
    ("Research", "Research"),
    ("Manager", "Management"),
    ("Director", "Management"),
    ("Head", "Management"),
    ("Lead", "Management"),
    ("Architect", "Architect"),
    ("Analytics", "Data Analyst"),
]


def _assign_job_family(title: str) -> str:
    """
    WHY GROUP 50 JOB TITLES INTO ~7 FAMILIES rather than one-hot encode
    all 50 directly: with only 607 rows total, one-hot encoding 50
    individual titles would create many columns with just 1-2 examples
    each -- not enough data for the model to learn anything reliable
    about most of them, and a real risk of overfitting to specific-title
    noise. Grouping into broader families (keyword-matched against each
    title, checked to cover the data reasonably) keeps the categories
    large enough to support a real pattern.
    """
    for keyword, family in JOB_FAMILY_KEYWORDS:
        if keyword in title:
            return family
    return "Other"


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # This dataset has zero missing values (verified in data_loader.py).
    # Phase 5's work here is about which columns to DROP as redundant
    # or misleading, not fixing bad values:
    # -----------------------------------------------------------------

    # WHY DROP `salary` and `salary_currency`: `salary` is the raw
    # amount in LOCAL currency -- comparing 70,000 EUR to 260,000 USD
    # directly would be meaningless. `salary_in_usd` already standardizes
    # this using each year's average FX rate, which is the correct
    # single target to model. Keeping the raw local-currency columns as
    # FEATURES would also leak strong target information (local salary
    # is highly correlated with its own USD conversion).
    df = df.drop(columns=["salary", "salary_currency"])

    return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # experience_level: ordinal encoding
    # WHY ORDINAL HERE (unlike bmi_category in the insurance project,
    # which was one-hot encoded despite looking ordinal): EDA verified
    # actual average salary rises monotonically EN -> MI -> SE -> EX
    # ($61.6K -> $88K -> $138.6K -> $199.4K) -- the ordinal assumption is
    # CHECKED against real data here, not just assumed from the label
    # names. When a categorical genuinely behaves ordinally, ordinal
    # encoding is the right, simpler choice; when it doesn't (like
    # remote_ratio below), one-hot is.
    # -----------------------------------------------------------------
    df["experience_level_ord"] = df["experience_level"].map(EXPERIENCE_MAP)

    # -----------------------------------------------------------------
    # company_size: ordinal encoding (same reasoning -- S < M < L in
    # average pay, verified in EDA: $77.6K -> $116.9K -> $119.2K)
    # -----------------------------------------------------------------
    df["company_size_ord"] = df["company_size"].map(COMPANY_SIZE_MAP)

    # -----------------------------------------------------------------
    # employment_type: collapse to a binary flag
    # WHY: 96.9% of records are Full-Time (FT); Part-Time, Contract, and
    # Freelance combined are only 19 rows total -- too few to support
    # separate one-hot categories the model could learn anything
    # reliable from individually. A single is_full_time flag captures
    # the one distinction the data can actually support.
    # -----------------------------------------------------------------
    df["is_full_time"] = (df["employment_type"] == "FT").astype(int)

    # -----------------------------------------------------------------
    # Engineered feature: is_cross_border
    # BUSINESS LOGIC: directly motivated by EDA's central finding --
    # employees whose residence differs from their employer's country
    # average $76,758 vs. $115,558 for same-country employees. This flag
    # lets the model (and the equity-flagging analysis in
    # train_model.py) use this pattern explicitly rather than requiring
    # it to be rediscovered from two separate country columns.
    # -----------------------------------------------------------------
    df["is_cross_border"] = (df["employee_residence"] != df["company_location"]).astype(int)

    # -----------------------------------------------------------------
    # remote_ratio: one-hot, NOT treated as a numeric/ordinal feature
    # WHY: EDA found the relationship is non-monotonic (hybrid/50% pays
    # LESS than both 0% and 100%) -- forcing it into a single numeric
    # feature would tell a linear model "half of 0 and 100 is the
    # expected value at 50," which is factually wrong for this data.
    # One-hot lets the model learn each level's own effect independently.
    # -----------------------------------------------------------------
    df["remote_ratio"] = df["remote_ratio"].astype(str)

    # -----------------------------------------------------------------
    # job_title -> job_family (see _assign_job_family's WHY above)
    # -----------------------------------------------------------------
    df["job_family"] = df["job_title"].apply(_assign_job_family)

    # -----------------------------------------------------------------
    # company_location -> location_group (top 10 countries + "Other")
    # WHY: the top 10 countries cover 88% of records; the remaining ~40
    # countries have too few examples each to support their own
    # category reliably, so they're grouped into "Other" rather than
    # creating dozens of near-empty one-hot columns.
    # -----------------------------------------------------------------
    top_countries = clean_df["company_location"].value_counts().head(TOP_N_COUNTRIES).index
    df["location_group"] = df["company_location"].where(df["company_location"].isin(top_countries), "Other")

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    dummy_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df


NUMERIC_MODEL_FEATURES = ["experience_level_ord", "company_size_ord", "work_year", "is_full_time", "is_cross_border"]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print("\nJob family distribution:")
    print(engineered.filter(like="job_family_").sum().sort_values(ascending=False))
    print(f"\nis_cross_border rate: {engineered['is_cross_border'].mean():.3%}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
