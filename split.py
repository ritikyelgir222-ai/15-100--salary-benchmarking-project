"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY STRATIFY ON A SALARY QUARTILE: salary_in_usd is continuous, and this
dataset is small (607 rows) with a right-skewed target dominated by a
handful of very high earners (EX-level, US-based). A plain random split
could easily leave the test set with an unlucky mix -- too many or too
few high earners -- making evaluation harder or easier than it should
be for the wrong reason. The same reasoning as the house-price and
insurance projects' quartile stratification, applied here.
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(X, y, salary_for_stratify, test_size=0.15, val_size=0.15, random_state=42):
    salary_quartile = pd.qcut(salary_for_stratify, q=4, labels=False, duplicates="drop")

    X_temp, X_test, y_temp, y_test, q_temp, _ = train_test_split(
        X, y, salary_quartile, test_size=test_size, stratify=salary_quartile, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=q_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
