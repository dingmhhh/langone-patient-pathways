#!/usr/bin/env python3
"""Model 1 / Q1: appointment no-shows and patient/late cancellations.

Designed for data/scenario/calibrated/{patients,appointments}.csv.
Synthetic data only; associations are not NYU Langone patient findings.

Run:
  python src/model1_appointment_barriers.py \
      --data-dir data/scenario/calibrated \
      --output-dir results/model1_calibrated

Dependencies: pandas numpy scipy statsmodels scikit-learn matplotlib
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

PURPLE = "#57068C"
ALIASES = {
    "patient_id": ["study_patient_id", "patient_id", "study_id", "pat_id", "person_id"],
    "appointment_id": ["appointment_id", "appt_id", "encounter_id", "pat_enc_csn_id"],
    "status": ["status", "appointment_status", "appt_status", "appt_status_c_name"],
    "appointment_date": ["appointment_date", "appt_date", "visit_date", "scheduled_date",
                         "scheduled_start", "appointment_datetime", "prov_start_time", "contact_date"],
    "booking_date": ["booking_date", "booked_date", "booking_datetime", "booked_at",
                     "created_at", "enc_instant"],
    "lead_days": ["booking_lead_time_days", "booking_lead_days", "lead_time_days",
                  "days_until_appointment", "days_to_appointment", "booking_lead_time"],
    "prior_no_shows": ["prior_no_shows", "prior_noshows", "previous_no_shows", "no_show_history"],
    "age": ["age", "patient_age", "age_years"],
    "mychart": ["mychart_active", "mychart_status", "mychart_activated", "portal_active"],
    "insurance": ["insurance_type", "insurance", "payer_type", "financial_class"],
    "sex": ["sex", "gender"],
    "race": ["race"],
    "ethnicity": ["ethnicity"],
    "language": ["preferred_language", "language"],
    "interpreter": ["interpreter_needed", "needs_interpreter"],
    "sdoh_screened": ["sdoh_screened"],
    "comorbidity": ["comorbidity_count"],
    "visit_type": ["visit_type", "appointment_type", "appt_type"],
    "visit_mode": ["visit_mode", "appointment_mode", "modality"],
    "booking_channel": ["booking_channel", "booking_source", "scheduling_channel"],
    "reminder_sent": ["reminder_sent", "reminder_status"],
    "reminder_opt_in": ["reminder_opt_in", "reminder_opted_in"],
    "own_pcp": ["seeing_own_pcp", "own_pcp", "seen_by_own_pcp"],
}

# These are Q1-relevant SDOH candidates, NOT all possible SDOH columns.
SDOH_KEYWORDS = ("financial", "transport", "housing", "food", "literacy", "social", "isolation")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_name(x) for x in result.columns]
    if result.columns.duplicated().any():
        raise ValueError("Columns collide after normalization: " + str(result.columns[result.columns.duplicated()].tolist()))
    return result


def find_col(df: pd.DataFrame, key: str) -> str | None:
    return next((name for name in ALIASES[key] if name in df.columns), None)


def norm_status(value: object) -> str:
    if pd.isna(value):
        return "missing"
    s = normalize_name(str(value))
    if "no_show" in s or "noshow" in s or "did_not_show" in s or s == "missed":
        return "no_show"
    if "late" in s and ("cancel" in s or "cancell" in s):
        return "late_cancel"
    if ("clinic" in s or "provider" in s or "practice" in s or "office" in s) and "cancel" in s:
        return "clinic_cancel"
    if ("patient" in s or "self" in s) and "cancel" in s:
        return "patient_cancel"
    if "cancel" in s or "cancell" in s:
        return "unspecified_cancel"
    if s in {"completed", "complete", "arrived", "checked_out", "finished", "seen"}:
        return "completed"
    return "other"


def to_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return float(value)
    s = normalize_name(str(value))
    if s in {"1", "true", "yes", "y", "active", "activated", "enabled", "enrolled"}:
        return 1.0
    if s in {"0", "false", "no", "n", "inactive", "inactivated", "disabled", "declined", "not_active"}:
        return 0.0
    return np.nan


def read_data(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    patient_path = data_dir / "patients.csv"
    appt_path = data_dir / "appointments.csv"
    if not patient_path.is_file() or not appt_path.is_file():
        raise FileNotFoundError(f"Expected {patient_path} and {appt_path}. README describes these files, but you must supply the CSVs locally.")
    patients = normalize_columns(pd.read_csv(patient_path, low_memory=False))
    appts = normalize_columns(pd.read_csv(appt_path, low_memory=False))
    pkey, akey = find_col(patients, "patient_id"), find_col(appts, "patient_id")
    status = find_col(appts, "status")
    if not pkey or not akey or not status:
        raise ValueError("Missing essential patient_id or appointment status field.\n"
                         f"patients columns: {patients.columns.tolist()}\n"
                         f"appointments columns: {appts.columns.tolist()}")
    if patients[pkey].duplicated().any():
        raise ValueError("patients.csv must contain exactly one row per patient_id; duplicates found.")
    patients = patients.rename(columns={pkey: "patient_id"})
    appts = appts.rename(columns={akey: "patient_id", status: "raw_status"})
    if appts["patient_id"].isna().any():
        appts = appts.loc[appts["patient_id"].notna()].copy()
    unmatched_patient_rows = int((~appts["patient_id"].isin(patients["patient_id"])).sum())
    # Merge only patient columns absent from appointment table; preserve source fields.
    patient_fields = [c for c in patients.columns if c == "patient_id" or c not in appts.columns]
    df = appts.merge(patients[patient_fields], on="patient_id", how="left", validate="many_to_one")
    df["status_norm"] = df["raw_status"].map(norm_status)
    meta = {"n_appointments": int(len(df)), "n_patients": int(df.patient_id.nunique()),
            "status_counts": {str(k): int(v) for k, v in df.status_norm.value_counts().items()},
            "unmatched_patient_rows": unmatched_patient_rows}
    return df, meta


def historical_no_shows(df: pd.DataFrame, date_col: str) -> pd.Series:
    # Counts no-shows on PRIOR calendar dates only; same-day appointments do not leak.
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    work = pd.DataFrame({"patient_id": df.patient_id, "day": dates,
                         "event": (df.status_norm == "no_show").astype(int)})
    valid = work.dropna(subset=["day"])
    daily = valid.groupby(["patient_id", "day"], as_index=False)["event"].sum()
    daily = daily.sort_values(["patient_id", "day"])
    daily["previous"] = daily.groupby("patient_id")["event"].cumsum() - daily["event"]
    key = pd.MultiIndex.from_frame(work[["patient_id", "day"]])
    mapping = daily.set_index(["patient_id", "day"])["previous"]
    return pd.Series(mapping.reindex(key).to_numpy(), index=df.index, dtype=float)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    result = pd.DataFrame(index=df.index)
    numerical: list[str] = []
    categorical: list[str] = []
    source_map: dict[str, str] = {}

    def add_num(feature: str, values: pd.Series, source: str):
        values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() >= 30 and values.nunique(dropna=True) > 1:
            result[feature] = values
            numerical.append(feature)
            source_map[feature] = source

    def add_cat(feature: str, values: pd.Series, source: str):
        clean = values.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        if clean.notna().sum() < 30 or clean.nunique(dropna=True) <= 1:
            return
        # Rare categories pooled, missing kept explicit (especially unscreened SDOH).
        counts = clean.value_counts()
        frequent = set(counts[counts >= 30].head(8).index)
        clean = clean.map(lambda x: str(x) if pd.notna(x) and x in frequent else ("Other" if pd.notna(x) else "Missing"))
        if clean.nunique() > 1:
            result[feature] = clean.astype(str)
            categorical.append(feature)
            source_map[feature] = source

    age_col = find_col(df, "age")
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce").where(lambda x: x.between(0, 110))
        add_num("age_per_10_years", age / 10, age_col)
        add_num("age_65plus", (age >= 65).astype(float).where(age.notna()), age_col)

    my_col = find_col(df, "mychart")
    if my_col:
        my = df[my_col].map(to_binary)
        add_num("mychart_active", my, my_col)
        if my.isna().any() and my.notna().sum() >= 30:
            add_num("mychart_unknown", my.isna().astype(float), my_col)
        if "age_65plus" in numerical and "mychart_active" in numerical:
            # Interaction only among patients whose MyChart status and age are known.
            interaction = result["age_65plus"] * result["mychart_active"]
            add_num("mychart_active_x_age65plus", interaction, f"{my_col} x {age_col}")

    date_col = find_col(df, "appointment_date")
    lead_col = find_col(df, "lead_days")
    if lead_col:
        lead = pd.to_numeric(df[lead_col], errors="coerce")
    else:
        booked_col = find_col(df, "booking_date")
        if booked_col and date_col:
            lead = (pd.to_datetime(df[date_col], errors="coerce") -
                    pd.to_datetime(df[booked_col], errors="coerce")).dt.total_seconds() / 86400
        else:
            lead = None
    if lead is not None:
        add_num("booking_lead_per_30_days", lead.where(lead.between(0, 730)) / 30,
                lead_col or "appointment_date - booking_date")

    if date_col and pd.to_datetime(df[date_col], errors="coerce").notna().sum() >= 30:
        prior = historical_no_shows(df, date_col)
        add_num("prior_no_shows", prior, f"computed from historical {date_col} and status")
    elif find_col(df, "prior_no_shows"):
        col = find_col(df, "prior_no_shows")
        add_num("prior_no_shows", df[col], col)
        warnings.warn("Using supplied prior_no_shows. Verify it was computed using only earlier appointments.")

    comorb = find_col(df, "comorbidity")
    if comorb:
        add_num("comorbidity_count", df[comorb], comorb)

    for key in ("insurance", "sex", "race", "ethnicity", "language", "interpreter",
                "sdoh_screened", "visit_type", "visit_mode", "booking_channel",
                "reminder_sent", "reminder_opt_in", "own_pcp"):
        col = find_col(df, key)
        if col:
            add_cat(key, df[col], col)

    for col in df.columns:
        if col.startswith("sdoh_") and col != "sdoh_screened" and any(k in col for k in SDOH_KEYWORDS):
            if not any(word in col for word in ("date", "time")):
                add_cat(col, df[col], col)

    if not numerical and not categorical:
        raise ValueError("No usable predictors found. Check CSV column names and ALIASES.")
    return result, numerical, categorical, source_map


def design_matrix(raw: pd.DataFrame, numerical: list[str], categorical: list[str]) -> tuple[pd.DataFrame, dict]:
    pieces = []
    labels = {}
    for col in numerical:
        x = pd.to_numeric(raw[col], errors="coerce")
        if x.isna().all():
            continue
        median = float(x.median())
        pieces.append(x.fillna(median).rename(col).to_frame().astype(float))
        labels[col] = col.replace("_", " ")
        if x.isna().any() and col not in {"mychart_unknown"}:
            missing = f"{col}__missing"
            pieces.append(x.isna().astype(float).rename(missing).to_frame())
            labels[missing] = col.replace("_", " ") + " (missing)"
    for col in categorical:
        s = raw[col].fillna("Missing").astype(str)
        # Baseline is most common observed category, not the alphabetically first.
        observed = s[s != "Missing"]
        baseline = observed.value_counts().index[0] if not observed.empty else s.value_counts().index[0]
        levels = [baseline] + sorted(set(s) - {baseline})
        dummies = pd.get_dummies(pd.Categorical(s, categories=levels), prefix=col,
                                 prefix_sep=" = ", drop_first=True, dtype=float)
        dummies.index = raw.index
        if not dummies.empty:
            pieces.append(dummies)
            for name in dummies.columns:
                labels[name] = f"{name} (ref: {baseline})"
    X = pd.concat(pieces, axis=1)
    # Remove constant and exact duplicate/linearly dependent columns.
    X = X.loc[:, X.nunique(dropna=False) > 1].astype(float)
    if X.empty:
        raise ValueError("Predictor matrix has no varying columns")
    keep, dropped = [], []
    for col in X.columns:
        test = X[keep + [col]].to_numpy(dtype=float)
        if np.linalg.matrix_rank(test) > len(keep):
            keep.append(col)
        else:
            dropped.append(col)
    if dropped:
        warnings.warn(f"Dropped linearly dependent columns: {dropped}")
    return X[keep], {k: labels[k] for k in keep}


def forest_plot(table: pd.DataFrame, title: str, path: Path):
    plot = table.loc[table.term != "const"].copy()
    plot = plot[np.isfinite(plot["or"]) & np.isfinite(plot["ci_low"]) & np.isfinite(plot["ci_high"])]
    # Avoid making a misleading chart when CI spans huge orders of magnitude.
    plot = plot.sort_values("or").tail(24)
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot) + 1.7)))
    y = np.arange(len(plot))
    ax.errorbar(plot["or"], y,
                xerr=[np.maximum(plot["or"] - plot["ci_low"], 0),
                      np.maximum(plot["ci_high"] - plot["or"], 0)],
                fmt="o", color=PURPLE, ecolor=PURPLE, capsize=2, markersize=4)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_yticks(y, plot["label"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Adjusted odds ratio (95% CI, log scale)")
    ax.set_title(title + "\nCalibrated synthetic data — not patient findings", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_mychart_age_interaction(glm, out_dir: Path) -> dict:
    """Compare adjusted MyChart Active vs Inactive ORs in age <65 and >=65.

    Compute Wald confidence intervals for the two linear contrasts using
    the full *patient-cluster robust* covariance matrix.  In particular,
    variance(beta_main + beta_interaction) includes their covariance.
    This is a conditional odds-ratio contrast, NOT an absolute risk plot.
    """
    main = "mychart_active"
    interaction = "mychart_active_x_age65plus"
    needed = [main, interaction]
    missing = [term for term in needed if term not in glm.params.index]
    if missing:
        msg = ("Cannot draw MyChart x age plot: missing model terms "
               + ", ".join(missing)
               + ". Check whether these columns vary and survive design_matrix().")
        warnings.warn(msg)
        return {"status": "skipped", "reason": msg}

    beta = glm.params.loc[needed].to_numpy(dtype=float)
    cov = glm.cov_params().loc[needed, needed].to_numpy(dtype=float)
    contrasts = [
        ("Under 65", np.array([1.0, 0.0])),
        ("65+", np.array([1.0, 1.0])),
    ]
    rows = []
    for group, weights in contrasts:
        log_or = float(weights @ beta)
        variance = float(weights @ cov @ weights)
        if not np.isfinite(variance) or variance < -1e-10:
            warnings.warn(f"Invalid estimated variance for {group}; skipping interaction plot")
            return {"status": "skipped", "reason": f"Invalid variance for {group}"}
        se = float(np.sqrt(max(0.0, variance)))
        rows.append({
            "age_group": group,
            "comparison": "MyChart Active vs Inactive",
            "log_odds_ratio": log_or,
            "std_error": se,
            "odds_ratio": float(np.exp(log_or)),
            "ci_lower": float(np.exp(log_or - 1.96 * se)),
            "ci_upper": float(np.exp(log_or + 1.96 * se)),
        })

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "no_show_mychart_age_interaction_or.csv"
    png_path = out_dir / "no_show_mychart_age_interaction_or.png"
    summary.to_csv(csv_path, index=False)

    # NYU-style bars + confidence intervals + no-association reference.
    fig, ax = plt.subplots(figsize=(8, 5.4))
    x = np.arange(len(summary))
    odds = summary["odds_ratio"].to_numpy(dtype=float)
    lower = summary["ci_lower"].to_numpy(dtype=float)
    upper = summary["ci_upper"].to_numpy(dtype=float)
    ax.bar(x, odds, width=0.48, color=PURPLE, alpha=0.82)
    ax.errorbar(x, odds, yerr=[odds - lower, upper - odds],
                fmt="none", ecolor="#252525", elinewidth=1.7,
                capsize=5, capthick=1.7, zorder=4)
    ax.axhline(1.0, ls="--", lw=1.1, color="#777777", label="OR = 1")
    ax.set_xticks(x, summary["age_group"].tolist())
    ax.set_ylabel("Adjusted odds ratio (95% CI)")
    ax.set_title("MyChart Active vs Inactive — No-show Odds\n"
                 "Calibrated synthetic data — not patient findings",
                 fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ymax = max(1.22, float(upper.max()) * 1.18)
    ax.set_ylim(0, ymax)
    for i, value in enumerate(odds):
        ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11)
    fig.text(0.5, 0.015,
             "Adjusted conditional ORs; 65+ CI uses main/interaction covariance.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved MyChart x Age interaction chart: {png_path}")
    print(f"Saved MyChart x Age interaction table: {csv_path}")
    return {"status": "ok", "figure": str(png_path), "table": str(csv_path),
            "groups": rows}




def plot_adjusted_probability_comparisons(glm, X: pd.DataFrame, out_dir: Path) -> dict:
    """Marginal-standardized No-show probabilities from the fitted GLM.

    For each comparison, use the SAME analysis rows for both levels, changing
    only the specified design-matrix terms. Wald 95% CIs use the fitted model's
    patient-clustered covariance and the delta method. This is an associational
    contrast, not a causal intervention estimate.
    """
    required = set(glm.params.index) - {"const"}
    if not required.issubset(X.columns):
        raise ValueError(f"Missing model predictors: {sorted(required - set(X.columns))}")

    beta = glm.params.to_numpy(dtype=float)
    cov = glm.cov_params().to_numpy(dtype=float)
    columns = glm.params.index.tolist()
    rows = []

    def evaluate(M: pd.DataFrame) -> tuple[float, np.ndarray]:
        mat = sm.add_constant(M, has_constant="add").reindex(columns=columns)
        a = mat.to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-np.clip(a @ beta, -35, 35)))
        mean_p = float(p.mean())
        gradient = (a * (p * (1.0 - p))[:, None]).mean(axis=0)
        return mean_p, gradient

    def compare(label: str, population: pd.DataFrame, lo_label: str,
                hi_label: str, modify):
        if len(population) < 30:
            warnings.warn(f"Skipped {label}: not enough observations")
            return
        M0 = population.copy()
        M1 = population.copy()
        modify(M0, M1)
        p0, g0 = evaluate(M0)
        p1, g1 = evaluate(M1)
        delta = p1 - p0
        grad_delta = g1 - g0
        var_delta = float(grad_delta @ cov @ grad_delta)
        se_delta = np.sqrt(max(0.0, var_delta))
        for level, p, grad in ((lo_label, p0, g0), (hi_label, p1, g1)):
            var_p = float(grad @ cov @ grad)
            se_p = np.sqrt(max(0.0, var_p))
            rows.append({
                "comparison": label, "level": level, "n_standardized_over": len(population),
                "adjusted_probability": p,
                "ci_lower": max(0.0, p - 1.96 * se_p),
                "ci_upper": min(1.0, p + 1.96 * se_p),
                "difference_pp_high_minus_low": 100.0 * delta,
                "difference_ci_lower_pp": 100.0 * (delta - 1.96 * se_delta),
                "difference_ci_upper_pp": 100.0 * (delta + 1.96 * se_delta),
            })

    # Restrict SDOH comparison to screened patients, because unscreened != no barrier.
    transport_col = "sdoh_transportation = 1.0"
    screened_missing = "sdoh_screened = False"
    if transport_col in X:
        population = X.loc[X[screened_missing].eq(0)] if screened_missing in X else X
        def set_transport(M0, M1):
            M0[transport_col] = 0.0
            M1[transport_col] = 1.0
        compare("Transportation barrier (screened only)", population,
                "No barrier", "Barrier", set_transport)

    if "prior_no_shows" in X:
        def set_prior(M0, M1):
            M0["prior_no_shows"] = 0.0
            M1["prior_no_shows"] = 1.0
        compare("Prior No-show History", X, "0 prior", "1 prior", set_prior)

    # Age-specific MyChart comparisons: hold age-group membership fixed and
    # set the interaction consistently with the altered MyChart status.
    age = "age_65plus"
    mychart = "mychart_active"
    interaction = "mychart_active_x_age65plus"
    if all(c in X.columns for c in (age, mychart, interaction)):
        for age_value, label in ((0, "MyChart: Under 65"), (1, "MyChart: 65+")):
            population = X.loc[X[age].eq(age_value)]
            def set_mychart(M0, M1, value=age_value):
                M0[mychart] = 0.0
                M1[mychart] = 1.0
                M0[interaction] = 0.0
                M1[interaction] = float(value)
                if "mychart_unknown" in M0.columns:
                    M0["mychart_unknown"] = 0.0
                    M1["mychart_unknown"] = 0.0
                if "mychart_active__missing" in M0.columns:
                    M0["mychart_active__missing"] = 0.0
                    M1["mychart_active__missing"] = 0.0
                if "mychart_active_x_age65plus__missing" in M0.columns:
                    M0["mychart_active_x_age65plus__missing"] = 0.0
                    M1["mychart_active_x_age65plus__missing"] = 0.0
            compare(label, population, "Inactive", "Active", set_mychart)

    if not rows:
        warnings.warn("No supported adjusted probability comparisons available")
        return {"status": "skipped"}

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "no_show_adjusted_probabilities.csv"
    png_path = out_dir / "no_show_adjusted_probability_comparison.png"
    summary.to_csv(csv_path, index=False)

    comparisons = summary["comparison"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(11, max(5, 1.05 * len(comparisons) + 2.0)))
    y_base = np.arange(len(comparisons)) * 1.5
    palette = ["#A8A2B8", PURPLE]
    for j, level_id in enumerate((0, 1)):
        chunk = summary.groupby("comparison", sort=False).nth(level_id)
        x = 100 * chunk["adjusted_probability"].to_numpy(dtype=float)
        low = 100 * chunk["ci_lower"].to_numpy(dtype=float)
        high = 100 * chunk["ci_upper"].to_numpy(dtype=float)
        ypos = y_base + (-0.25 if j == 0 else 0.25)
        ax.barh(ypos, x, height=0.44, color=palette[j], alpha=0.95)
        ax.errorbar(x, ypos, xerr=[np.maximum(0, x-low), np.maximum(0, high-x)],
                    fmt="none", ecolor="#292929", capsize=3, lw=1.3)
        for k, (value, lev) in enumerate(zip(x, chunk["level"])):
            ax.annotate(f"{lev}: {value:.1f}%",
                        (100 * float(summary["ci_upper"].max()) * 1.08, ypos[k]),
                        xytext=(0, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(y_base, comparisons)
    ax.invert_yaxis()
    ax.set_xlabel("Model-adjusted predicted No-show probability (%)")
    ax.set_title("Model 1 — Adjusted Predicted Probability Comparison\n"
                 "Calibrated synthetic data — not patient findings", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ymax = max(1.0, 100 * float(summary["ci_upper"].max()))
    ax.set_xlim(0, ymax * 1.55)
    fig.text(0.5, 0.015,
             "Average standardized predictions within each comparison population; 95% CIs use patient-clustered covariance.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved adjusted probability chart: {png_path}")
    print(f"Saved adjusted probability table: {csv_path}")
    return {"status": "ok", "figure": str(png_path), "table": str(csv_path)}



# ==============================================================
# Q1 SPONSOR-FACING FIGURES (standardized predicted probabilities)
# ==============================================================


def _standardized_predict(glm, X: pd.DataFrame) -> tuple[float, float, float]:
    """Population-averaged fitted risk and delta-method 95% CI.

    Inference uses the cluster-robust covariance matrix fitted in analyze().
    Results are modeled associations on calibrated SYNTHETIC data.
    """
    terms = glm.params.index.tolist()
    M = sm.add_constant(X, has_constant="add").reindex(columns=terms)
    if M.isna().any().any():
        raise ValueError("Counterfactual design matrix is missing fitted GLM terms")
    A = M.to_numpy(dtype=float)
    beta = glm.params.to_numpy(dtype=float)
    p = expit(A @ beta)
    point = float(p.mean())
    gradient = (A * (p * (1 - p))[:, None]).mean(axis=0)
    covariance = glm.cov_params().to_numpy(dtype=float)
    se = float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))
    return point, max(0.0, point - 1.96 * se), min(1.0, point + 1.96 * se)


def _save_probability_plot(points: pd.DataFrame, title: str, path: Path,
                           ylabel: str = "Adjusted predicted probability (%)") -> None:
    """Simple sponsor-ready interval plot for a series of discrete scenarios."""
    fig, ax = plt.subplots(figsize=(9, 5.4))
    xp = np.arange(len(points))
    prob = points["adjusted_probability"].to_numpy(dtype=float) * 100
    low = points["ci_lower"].to_numpy(dtype=float) * 100
    high = points["ci_upper"].to_numpy(dtype=float) * 100
    ax.bar(xp, prob, color=PURPLE, width=0.58, alpha=0.83)
    ax.errorbar(xp, prob, yerr=[np.maximum(0, prob-low), np.maximum(0, high-prob)],
                fmt="none", color="#2b2b2b", capsize=4, lw=1.4)
    ax.set_xticks(xp, points["group"].astype(str), rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\nCalibrated synthetic data — not patient findings", weight="bold")
    ax.set_ylim(0, max(1, high.max() * 1.27))
    ax.grid(axis="y", alpha=0.13)
    ax.set_axisbelow(True)
    for i, v in enumerate(prob):
        ax.annotate(f"{v:.1f}%", (i, high[i]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_status_distribution(df: pd.DataFrame, out_dir: Path) -> dict:
    """Figure 1: ALL scheduled appointment outcomes; denominator is all records."""
    labels = [
        ("completed", "Completed"), ("no_show", "No-show"),
        ("patient_cancel", "Patient canceled"), ("late_cancel", "Late canceled"),
        ("clinic_cancel", "Clinic canceled"),
    ]
    counts = df["status_norm"].value_counts()
    rows = [{"status": label, "count": int(counts.get(key, 0)),
             "share_all_appointments": float(counts.get(key, 0) / len(df))}
            for key, label in labels]
    rows.extend({"status": label, "count": int(count), "share_all_appointments": float(count / len(df))}
                for label, count in counts.items() if label not in dict(labels))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "fig1_appointment_outcomes.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(table))
    colors = ["#70A69A", PURPLE, "#A586C7", "#D39B55", "#9DA2A7"] + ["#777777"] * max(0, len(table)-5)
    bars = ax.bar(x, 100 * table["share_all_appointments"], color=colors[:len(table)], width=.67)
    ax.set_xticks(x, table["status"], rotation=15, ha="right")
    ax.set_ylabel("Share of all appointments (%)")
    ax.set_title("Figure 1 — Appointment Outcome Distribution\nCalibrated synthetic data — not patient findings", weight="bold")
    ax.set_ylim(0, max(1, 100 * table["share_all_appointments"].max() * 1.16))
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f"{row['share_all_appointments']:.1%}\n(n={row['count']:,})", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=.13)
    ax.set_axisbelow(True)
    fig.tight_layout()
    dest = out_dir / "fig1_appointment_outcomes.png"
    fig.savefig(dest, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"status":"ok", "figure":str(dest), "table":str(out_dir / "fig1_appointment_outcomes.csv")}


def plot_lead_time_probability(glm, X: pd.DataFrame, out_dir: Path) -> dict:
    """Figure 3: average standardized probabilities over a prespecified 0–180 day range."""
    key = "booking_lead_per_30_days"
    if key not in X:
        return {"status": "skipped", "reason": f"Missing predictor: {key}"}
    days_grid = np.arange(0, 181, 15)
    rows = []
    for days in days_grid:
        modified = X.copy()
        modified[key] = days / 30
        missing_key = key + "__missing"
        if missing_key in modified:
            modified[missing_key] = 0
        point, lower, upper = _standardized_predict(glm, modified)
        rows.append({"lead_time_days":int(days), "adjusted_probability":point,
                     "ci_lower":lower, "ci_upper":upper, "n_standardized_over":len(X)})
    table = pd.DataFrame(rows)
    csv = out_dir / "fig3_lead_time_adjusted_probability.csv"
    png = out_dir / "fig3_lead_time_adjusted_probability.png"
    table.to_csv(csv, index=False)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(table.lead_time_days, 100*table.adjusted_probability, color=PURPLE, lw=2.5)
    ax.fill_between(table.lead_time_days, 100*table.ci_lower, 100*table.ci_upper,
                    color=PURPLE, alpha=0.15, label="95% CI")
    ax.set(xlabel="Booking lead time (days)", ylabel="Adjusted predicted No-show probability (%)",
           xlim=(0, 180))
    ax.set_title("Figure 3 — Booking Lead Time and No-show Risk\nCalibrated synthetic data — not patient findings", weight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=.13)
    fig.tight_layout()
    fig.savefig(png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"status":"ok", "figure":str(png), "table":str(csv)}


def _set_categorical_level(Xnew: pd.DataFrame, Xoriginal: pd.DataFrame, raw: pd.DataFrame,
                           feature: str, level: str):
    """Change a category using SAME encoding/baseline as design_matrix()."""
    dummy_prefix = feature + " = "
    dummies = [c for c in Xoriginal.columns if c.startswith(dummy_prefix)]
    observed = raw[feature].astype(str)
    observed = observed[observed != "Missing"]
    if observed.empty:
        raise ValueError(f"No observed {feature} categories")
    baseline = observed.value_counts().index[0]
    valid_levels = set(observed) | {"Missing"}
    if level not in valid_levels:
        raise ValueError(f"Category {level} is not found in {feature}")
    if level != baseline and (dummy_prefix + level) not in Xoriginal:
        raise ValueError(f"Category {level} was not estimable in {feature} model")
    Xnew.loc[:, dummies] = 0.0
    dummy = dummy_prefix + level
    if dummy in Xnew:
        Xnew[dummy] = 1.0


def plot_visit_type_probability(glm, X: pd.DataFrame, raw: pd.DataFrame,
                                out_dir: Path) -> dict:
    """Figure 4: covariate-standardized risk for each estimable visit type."""
    if "visit_type" not in raw:
        return {"status":"skipped", "reason":"visit_type unavailable"}
    preferred = ["New Patient", "Follow-up", "Annual Physical"]
    available = list(raw["visit_type"].astype(str).value_counts().index)
    levels = [x for x in preferred if x in available] + [x for x in available if x not in preferred and x != "Missing"]
    rows = []
    for level in levels[:6]:
        modified = X.copy()
        try:
            _set_categorical_level(modified, X, raw, "visit_type", level)
        except ValueError:
            continue
        point, lower, upper = _standardized_predict(glm, modified)
        rows.append({"group":level,"adjusted_probability":point,"ci_lower":lower,
                     "ci_upper":upper,"n_standardized_over":len(X)})
    if len(rows) < 2:
        return {"status":"skipped", "reason":"fewer than two estimable visit types"}
    table = pd.DataFrame(rows)
    csv = out_dir / "fig4_visit_type_adjusted_probability.csv"
    png = out_dir / "fig4_visit_type_adjusted_probability.png"
    table.to_csv(csv,index=False)
    _save_probability_plot(table, "Figure 4 — Adjusted No-show Risk by Visit Type", png)
    return {"status":"ok", "figure":str(png), "table":str(csv)}


def plot_mychart_group_probabilities(glm, X: pd.DataFrame, out_dir: Path) -> dict:
    """Figure 5: four adjusted risks, standardized WITHIN valid age strata.

    Age group is never changed counterfactually. Changing MyChart also sets
    the interaction term consistently, avoiding impossible age/interaction pairs.
    """
    required = ("age_65plus", "mychart_active", "mychart_active_x_age65plus")
    if any(c not in X for c in required):
        return {"status":"skipped", "reason":"MyChart x Age terms not estimable"}
    rows = []
    for age_value, age_label in ((0.0, "Under 65"), (1.0, "65+")):
        group = X.loc[X["age_65plus"].eq(age_value)]
        if len(group) < 30:
            continue
        for mychart_value, mychart_label in ((0.0,"Inactive"),(1.0,"Active")):
            changed = group.copy()
            changed["mychart_active"] = mychart_value
            changed["mychart_active_x_age65plus"] = mychart_value * age_value
            for feature in ("mychart_active", "mychart_active_x_age65plus"):
                missing = feature + "__missing"
                if missing in changed:
                    changed[missing] = 0.0
            if "mychart_unknown" in changed:
                changed["mychart_unknown"] = 0.0
            point, lower, upper = _standardized_predict(glm, changed)
            rows.append({"age_group":age_label,"mychart":mychart_label,
                         "adjusted_probability":point,"ci_lower":lower,"ci_upper":upper,
                         "n_standardized_over":len(group)})
    if len(rows) != 4:
        return {"status":"skipped", "reason":"Insufficient observations in an age stratum"}
    table = pd.DataFrame(rows)
    csv = out_dir / "fig5_mychart_age_adjusted_probability.csv"
    png = out_dir / "fig5_mychart_age_adjusted_probability.png"
    table.to_csv(csv,index=False)
    fig, ax = plt.subplots(figsize=(9,5.5))
    xp = np.arange(2)
    width = 0.32
    for j, (level, color) in enumerate((("Inactive", "#B8A8C8"),("Active",PURPLE))):
        chunk = table.loc[table.mychart.eq(level)].set_index("age_group").loc[["Under 65", "65+"]]
        vals = 100 * chunk.adjusted_probability.to_numpy()
        lows = 100 * chunk.ci_lower.to_numpy()
        highs = 100 * chunk.ci_upper.to_numpy()
        xx = xp + (j-.5)*width
        bars = ax.bar(xx, vals, width, label=f"MyChart {level}", color=color)
        ax.errorbar(xx, vals, yerr=[vals-lows, highs-vals], fmt="none", color="#292929", capsize=4)
        for xx0, high0, value in zip(xx, highs, vals):
            ax.annotate(f"{value:.1f}%", (xx0,high0), xytext=(0,6), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(xp, ["Under 65", "65+"])
    ax.set_ylabel("Adjusted predicted No-show probability (%)")
    ax.set_title("Figure 5 — MyChart × Age Interaction\nCalibrated synthetic data — not patient findings", weight="bold")
    ax.set_ylim(0,max(1,100*table.ci_upper.max()*1.25))
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.13)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(png,dpi=250,bbox_inches="tight")
    plt.close(fig)
    return {"status":"ok", "figure":str(png), "table":str(csv)}


def plot_cancellation_risk_and_timing(glm, X: pd.DataFrame, df: pd.DataFrame,
                                     out_dir: Path) -> dict:
    """Figure 6: cancellation probability vs lead time + late-cancel composition.

    Data README does not promise cancellation timestamps. The right panel is
    the late-vs-regular split among patient cancellations, NOT true hours-to-cancel.
    """
    key = "booking_lead_per_30_days"
    if key not in X:
        return {"status":"skipped", "reason":"Booking lead time predictor unavailable"}
    days_grid = [0, 15, 30, 60, 90, 120, 150, 180]
    rows = []
    for days in days_grid:
        changed = X.copy()
        changed[key] = days / 30.0
        missing = key + "__missing"
        if missing in changed:
            changed[missing] = 0.0
        p, lo, hi = _standardized_predict(glm, changed)
        rows.append({"lead_time_days":days,"adjusted_cancellation_probability":p,
                     "ci_lower":lo,"ci_upper":hi,"n_standardized_over":len(X)})
    table = pd.DataFrame(rows)
    csv = out_dir / "fig6_cancellation_risk_by_lead_time.csv"
    png = out_dir / "fig6_cancellation_risk_and_late_share.png"
    table.to_csv(csv,index=False)
    total_patient_cancel = int(df.status_norm.isin(["patient_cancel","late_cancel"]).sum())
    late_count = int(df.status_norm.eq("late_cancel").sum())
    regular_count = int(df.status_norm.eq("patient_cancel").sum())
    fig, axes = plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={"width_ratios":[1.35,1]})
    ax=axes[0]
    ax.plot(table.lead_time_days,100*table.adjusted_cancellation_probability,color=PURPLE,lw=2.5)
    ax.fill_between(table.lead_time_days,100*table.ci_lower,100*table.ci_upper,color=PURPLE,alpha=.15)
    ax.set(xlabel="Booking lead time (days)",ylabel="Adjusted patient/late cancellation probability (%)",xlim=(0,180))
    ax.set_title("Adjusted cancellation risk")
    ax.grid(alpha=.12)
    ax2=axes[1]
    if total_patient_cancel:
        ax2.bar(["Patient canceled","Late canceled"],
                [100*regular_count/total_patient_cancel,100*late_count/total_patient_cancel],
                color=["#B8A8C8",PURPLE],width=.57)
        for i,v in enumerate((regular_count,late_count)):
            ax2.text(i,100*v/total_patient_cancel+1,f"{v:,} ({v/total_patient_cancel:.1%})",ha="center",fontsize=9)
        ax2.set_ylim(0,115)
    else:
        ax2.text(.5,.5,"No patient cancellations",ha="center",va="center",transform=ax2.transAxes)
    ax2.set(ylabel="Share of patient-initiated cancellations (%)",title="Late vs other patient cancellations")
    ax2.grid(axis="y",alpha=.12)
    fig.suptitle("Figure 6 — Patient Cancellation Risk and Late-cancel Composition\n"
                 "Calibrated synthetic data — not patient findings",fontsize=13,fontweight="bold")
    fig.text(.5,.005,"Late cancellation status is NOT a cancellation timestamp; no hours-before-visit distribution is inferred.",
             ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=[0,.035,1,.88])
    fig.savefig(png,dpi=250,bbox_inches="tight")
    plt.close(fig)
    return {"status":"ok", "figure":str(png), "table":str(csv),
            "late_cancel_share_within_patient_cancellations": (late_count/total_patient_cancel if total_patient_cancel else None)}

def analyze(name: str, df: pd.DataFrame, raw_features: pd.DataFrame,
            numerical: list[str], categorical: list[str], out_dir: Path, test_size: float) -> dict:
    if name == "no_show":
        eligible = df.status_norm.isin(["no_show", "completed"])
        positive = "no_show"
    else:
        eligible = df.status_norm.isin(["patient_cancel", "late_cancel", "completed"])
        positive = "patient_cancel_or_late_cancel"
    subset = df.loc[eligible].copy()
    if name == "no_show":
        y = subset.status_norm.eq("no_show").astype(int)
    else:
        y = subset.status_norm.isin(["patient_cancel", "late_cancel"]).astype(int)
    if len(subset) < 100 or y.sum() < 20 or (1 - y).sum() < 20:
        return {"outcome": positive, "status": "skipped: insufficient events", "n": len(subset), "events": int(y.sum())}

    X, labels = design_matrix(raw_features.loc[eligible], numerical, categorical)
    # Full-sample association estimates, with patient-clustered standard errors.
    exog = sm.add_constant(X, has_constant="add")
    try:
        glm = sm.GLM(y, exog, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": subset.patient_id}, maxiter=200)
        cluster_note = "Patient-clustered robust standard errors"
    except Exception as exc:
        return {"outcome": positive, "status": f"inference fit failed: {exc}",
                "n": len(subset), "events": int(y.sum())}
    ci = glm.conf_int()
    table = pd.DataFrame({
        "term": glm.params.index,
        "coefficient_log_odds": glm.params.values,
        "se": glm.bse.values,
        "p_value": glm.pvalues.values,
        "or": np.exp(np.clip(glm.params.values, -700, 700)),
        "ci_low": np.exp(np.clip(ci.iloc[:, 0].values, -700, 700)),
        "ci_high": np.exp(np.clip(ci.iloc[:, 1].values, -700, 700)),
    })
    table["label"] = table.term.map(lambda k: labels.get(k, "Intercept"))
    table.to_csv(out_dir / f"{name}_odds_ratios.csv", index=False)
    forest_plot(table, f"Model 1 — {positive.replace('_', ' ').title()}", out_dir / f"{name}_forest_plot.png")
    interaction_summary = plot_mychart_age_interaction(glm, out_dir) if name == "no_show" else None
    adjusted_summary = plot_adjusted_probability_comparisons(glm, X, out_dir) if name == "no_show" else None
    figures_q1 = {}
    if name == "no_show":
        figures_q1["fig2_patient_factors"] = adjusted_summary
        figures_q1["fig3_booking_lead_time"] = plot_lead_time_probability(glm, X, out_dir)
        figures_q1["fig4_visit_type"] = plot_visit_type_probability(
            glm, X, raw_features.loc[eligible], out_dir)
        figures_q1["fig5_mychart_age"] = plot_mychart_group_probabilities(glm, X, out_dir)
    else:
        figures_q1["fig6_cancellation"] = plot_cancellation_risk_and_timing(glm, X, df, out_dir)

    # Patient-level held-out evaluation. Feature encoding is outcome-independent;
    # model parameters are learned on training patients only.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    tr, te = next(splitter.split(X, y, groups=subset.patient_id))
    y_train, y_test = y.iloc[tr], y.iloc[te]
    evaluation = {"split": "held-out patients", "test_patients": int(subset.patient_id.iloc[te].nunique()),
                  "train_n": int(len(tr)), "test_n": int(len(te)),
                  "test_events": int(y_test.sum()), "test_event_rate": float(y_test.mean())}
    if y_train.nunique() == 2 and y_test.nunique() == 2:
        clf = LogisticRegression(max_iter=3000, class_weight=None)
        clf.fit(X.iloc[tr], y_train)
        probabilities = clf.predict_proba(X.iloc[te])[:, 1]
        auc = float(roc_auc_score(y_test, probabilities))
        evaluation.update({"heldout_roc_auc": auc,
                           "heldout_average_precision": float(average_precision_score(y_test, probabilities)),
                           "heldout_brier_score": float(brier_score_loss(y_test, probabilities))})
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=PURPLE, lw=2, label=f"Held-out AUROC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", label="Random = 0.500")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1),
               title=f"Model 1 — {positive.replace('_', ' ').title()}\nCalibrated synthetic data")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_heldout_roc.png", dpi=200)
        plt.close(fig)
        pd.DataFrame({"patient_id": subset.patient_id.iloc[te].to_numpy(),
                      "observed": y_test.to_numpy(), "predicted_probability": probabilities}).to_csv(
                          out_dir / f"{name}_heldout_predictions.csv", index=False)
    else:
        evaluation["evaluation_note"] = "Held-out set has only one outcome class; AUROC unavailable."

    return {"outcome": positive, "status": "ok", "n": int(len(subset)), "events": int(y.sum()),
            "event_rate": float(y.mean()), "n_patients": int(subset.patient_id.nunique()),
            "n_predictors": int(X.shape[1]), "se_method": cluster_note,
            "events_per_parameter": round(float(y.sum() / max(1, X.shape[1])), 2),
            "caution": "Sparse events per parameter can make odds ratios unstable" if y.sum() < 10 * X.shape[1] else None,
            **({"mychart_age_interaction": interaction_summary,
                "adjusted_probability_comparisons": adjusted_summary} if name == "no_show" else {}),
            "sponsor_figures": figures_q1,
            **evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scenario/calibrated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model1_calibrated"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, metadata = read_data(args.data_dir)
    features, numerical, categorical, source_map = prepare_features(df)
    metadata.update({"data_dir": str(args.data_dir), "data_type": "calibrated synthetic",
                     "features": source_map,
                     "notes": [
                         "Associations, not causal effects or actual NYU Langone findings.",
                         "Unspecified cancellations excluded from both outcomes; clinic cancellations excluded.",
                         "SDOH missing values kept as Missing; do not equate missing with no barrier.",
                         "Prior no-shows, when derived, use only earlier appointment calendar dates.",
                         "Full-sample ORs and held-out patient-level AUROC answer different questions.",
                         "A MyChart x age65+ interaction can be unstable in small subgroups.",
                     ]})
    results = {}
    for model in ("no_show", "cancellation"):
        print(f"\nRunning {model} ...")
        results[model] = analyze(model, df, features, numerical, categorical, args.output_dir, args.test_size)
        print(json.dumps(results[model], indent=2, ensure_ascii=False))
    metadata["models"] = results
    metadata["sponsor_figure_1"] = plot_status_distribution(df, args.output_dir)
    metadata["sponsor_figure_2"] = "no_show_adjusted_probability_comparison.png (existing figure)"
    metadata["cancellation_timing_note"] = (
        "Late-cancel share is computed from appointment status; no cancellation timestamps are available in the README."
    )
    (args.output_dir / "model1_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    df.status_norm.value_counts().rename_axis("status").reset_index(name="n").to_csv(
        args.output_dir / "appointment_status_counts.csv", index=False)
    print(f"\nSaved results to: {args.output_dir.resolve()}")
    print("NOTE: calibrated SYNTHETIC data only. Do not report coefficients as real NYU Langone effects.")


if __name__ == "__main__":
    main()
