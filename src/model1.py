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
    (args.output_dir / "model1_summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    df.status_norm.value_counts().rename_axis("status").reset_index(name="n").to_csv(
        args.output_dir / "appointment_status_counts.csv", index=False)
    print(f"\nSaved results to: {args.output_dir.resolve()}")
    print("NOTE: calibrated SYNTHETIC data only. Do not report coefficients as real NYU Langone effects.")


if __name__ == "__main__":
    main()
