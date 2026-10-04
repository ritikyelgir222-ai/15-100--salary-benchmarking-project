"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that benchmarks expected salary for a role
profile, and flags whether a given actual salary looks like a pay-equity
review candidate relative to that benchmark. Reuses
clean_and_engineer.py so training and serving logic never drift apart.

Run with:  uvicorn app:app --reload
"""

from typing import Optional

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Salary Benchmarking API", version="1.0")

MODEL = joblib.load("outputs/salary_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")
UNDERPAID_THRESHOLD = 0.80


class BenchmarkRequest(BaseModel):
    work_year: int = 2022
    experience_level: str  # "EN" | "MI" | "SE" | "EX"
    employment_type: str = "FT"
    job_title: str
    employee_residence: str  # ISO 3166 country code, e.g. "US"
    remote_ratio: int  # 0 | 50 | 100
    company_location: str  # ISO 3166 country code
    company_size: str  # "S" | "M" | "L"
    actual_salary_usd: Optional[float] = None  # optional -- if given, checks it against the benchmark


class BenchmarkResponse(BaseModel):
    benchmark_salary_usd: float
    actual_salary_usd: Optional[float] = None
    flagged_for_equity_review: Optional[bool] = None
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Salary Benchmarking</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Salary Benchmarking — Test Form</h2>
        <p>Every field maps 1:1 to <code>BenchmarkRequest</code> (app.py).
           Leave "actual salary" blank to just get a benchmark, or fill it
           in to check it against the benchmark. Full API docs at
           <a href="/docs">/docs</a>.</p>
        <form id="benchForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Experience level<br>
                <select name="experience_level" required>
                    <option>EN</option><option selected>MI</option><option>SE</option><option>EX</option>
                </select>
            </label>
            <label>Job title<br><input name="job_title" value="Data Scientist" required></label>
            <label>Employee residence (country code)<br><input name="employee_residence" value="IN" required></label>
            <label>Company location (country code)<br><input name="company_location" value="US" required></label>
            <label>Remote ratio<br>
                <select name="remote_ratio" required>
                    <option value="0">0 (in-office)</option>
                    <option value="50">50 (hybrid)</option>
                    <option value="100" selected>100 (fully remote)</option>
                </select>
            </label>
            <label>Company size<br>
                <select name="company_size" required><option>S</option><option selected>M</option><option>L</option></select>
            </label>
            <label>Actual salary, USD (optional)<br><input name="actual_salary_usd" type="number" placeholder="e.g. 65000"></label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Benchmark</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("benchForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                experience_level: form.get("experience_level"),
                job_title: form.get("job_title"),
                employee_residence: form.get("employee_residence"),
                company_location: form.get("company_location"),
                remote_ratio: parseInt(form.get("remote_ratio"), 10),
                company_size: form.get("company_size"),
                actual_salary_usd: form.get("actual_salary_usd") ? parseFloat(form.get("actual_salary_usd")) : null
            };
            const res = await fetch("/benchmark", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                document.getElementById("result").innerText = "Error " + res.status + ": " + await res.text();
                return;
            }
            const data = await res.json();
            let text = "Benchmark: $" + data.benchmark_salary_usd.toLocaleString();
            if (data.actual_salary_usd !== null && data.actual_salary_usd !== undefined) {
                text += " | Actual: $" + data.actual_salary_usd.toLocaleString();
                text += " | Flagged for equity review: " + data.flagged_for_equity_review;
            }
            text += " | Top factors: " + data.top_factors.join(", ");
            document.getElementById("result").innerText = text;
        });
        </script>
    </body>
    </html>
    """


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark_salary(record: BenchmarkRequest):
    raw = record.model_dump()
    raw_row = {
        "work_year": raw["work_year"],
        "experience_level": raw["experience_level"],
        "employment_type": raw["employment_type"],
        "job_title": raw["job_title"],
        "salary": 0, "salary_currency": "USD",  # placeholders so clean_data()'s drop-column step has something to drop
        "salary_in_usd": 0,  # placeholder; unused for scoring
        "employee_residence": raw["employee_residence"],
        "remote_ratio": raw["remote_ratio"],
        "company_location": raw["company_location"],
        "company_size": raw["company_size"],
    }
    raw_df = pd.DataFrame([raw_row])

    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # NOTE: the model was trained on log1p(salary_in_usd); both
    # RandomForestRegressor and XGBRegressor predict on that log scale,
    # so expm1 inverts it back to a dollar figure here.
    pred_log = MODEL.predict(X)[0]
    benchmark = float(np.expm1(pred_log))

    if hasattr(MODEL, "feature_importances_"):
        importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
        top_factors = importances.sort_values(ascending=False).head(3).index.tolist()
    else:
        top_factors = ["experience_level_ord", "location_group_US", "is_cross_border"]

    flagged = None
    actual = raw["actual_salary_usd"]
    if actual is not None:
        flagged = actual < UNDERPAID_THRESHOLD * benchmark

    return BenchmarkResponse(
        benchmark_salary_usd=round(benchmark, 2),
        actual_salary_usd=actual,
        flagged_for_equity_review=flagged,
        top_factors=top_factors,
    )
