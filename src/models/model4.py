#!/usr/bin/env python3
"""Model 4 / Q7: ED, urgent care and hospital use after missed care.

Designed for data/scenario/calibrated/ (patients, appointments, referrals,
hospital_visits). Synthetic data only; associations are not NYU Langone findings.

What it answers
  Q7.1  After a missed PCP visit (no-show or late cancel), are patients more
        likely to use the ED / urgent care / hospital within 30, 90, 180 days?
        Logistic regression, shown unadjusted and adjusted, because sicker
        patients both miss more visits and use the ED more (confounding).
  Q7.1  Same question after a referral that was not completed by day 60.
  Q7.2  Time from ED to next PCP visit: NOT estimated here. The synthetic
        data has no 'return to PCP after ED' mechanism (see Model 2, D14).

Run:
  python src/models/model4.py \
      --data-dir data/scenario/calibrated \
      --output-dir results/model4_calibrated

Dependencies: pandas numpy statsmodels matplotlib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

PURPLE = "#57068C"
LIGHT_PURPLE = "#C9B6DE"
GRAY = "#8A8A8A"
SUBTITLE = "Calibrated synthetic data — not patient findings"

WINDOWS = [30, 90, 180]
MISSED = ["No-show", "Late canceled"]

COVARIATES = ["recent_missed_90d", "comorbidity_count", "age65", "no_mychart",
              "C(insurance_type, Treatment('Commercial'))"]
LABELS = {
    "missed_visit": "Missed this visit (no-show / late cancel)",
    "recent_missed_90d": "Missed another visit in the prior 90 days",
    "comorbidity_count": "Chronic conditions (per condition)",
    "age65": "Age 65+",
    "no_mychart": "No active MyChart",
    "incomplete_referral": "Referral not completed by day 60",
}


# ============================================================
# 0. LOAD
# ============================================================

def to_flag(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.lower()
             .map({"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}))


def read_data(data_dir: Path) -> dict:
    need = ["patients", "appointments", "referrals", "hospital_visits"]
    missing = [n for n in need if not (data_dir / f"{n}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing in {data_dir}: {', '.join(m + '.csv' for m in missing)}")
    d = {n: pd.read_csv(data_dir / f"{n}.csv", low_memory=False) for n in need}

    p = d["patients"]
    p["mychart_active"] = to_flag(p["mychart_active"])
    p["no_mychart"] = 1 - p["mychart_active"]
    p["age65"] = (p["age"] >= 65).astype(int)

    a = d["appointments"]
    a["appointment_date"] = pd.to_datetime(a["appointment_date"])
    d["appointments"] = a.sort_values(["study_patient_id", "appointment_date"]).reset_index(drop=True)
    d["referrals"]["referral_date"] = pd.to_datetime(d["referrals"]["referral_date"])
    d["hospital_visits"]["admission_date"] = pd.to_datetime(d["hospital_visits"]["admission_date"])
    d["window_end"] = max(a["appointment_date"].max(), d["hospital_visits"]["admission_date"].max())
    return d


def add_patient_columns(df: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    cols = ["study_patient_id", "comorbidity_count", "age65", "no_mychart", "insurance_type"]
    return df.merge(p[cols], on="study_patient_id", how="left", validate="many_to_one")


def any_visit_between(index: pd.DataFrame, hv: pd.DataFrame, start_col: str, lo: int, hi: int) -> pd.Series:
    """1 if the patient has an ED/UC/inpatient visit in (start + lo, start + hi] days."""
    h = hv[["study_patient_id", "admission_date"]].sort_values("admission_date")
    left = index.assign(_key=index[start_col] + pd.Timedelta(days=lo + 1), _row=np.arange(len(index))).sort_values("_key")
    out = pd.merge_asof(left, h, left_on="_key", right_on="admission_date",
                        by="study_patient_id", direction="forward").sort_values("_row")
    hit = out["admission_date"].notna() & (out["admission_date"] <= out[start_col] + pd.Timedelta(days=hi))
    return pd.Series(hit.astype(int).to_numpy(), index=index.index)


# ============================================================
# 1. BUILD ANALYSIS TABLES
# ============================================================

def visit_index(d: dict) -> pd.DataFrame:
    """One row per attended or missed PCP appointment (the 'index' visit)."""
    a = d["appointments"]
    idx = a[a["appointment_status"].isin(["Completed"] + MISSED)].copy()
    idx["missed_visit"] = idx["appointment_status"].isin(MISSED).astype(int)
    # Carry-over: a visit soon after an earlier missed visit is still inside that visit's risk window
    prev = (a[a["appointment_status"].isin(MISSED)][["study_patient_id", "appointment_date"]]
            .rename(columns={"appointment_date": "prev_missed"}).sort_values("prev_missed"))
    idx = pd.merge_asof(idx.sort_values("appointment_date"), prev, left_on="appointment_date",
                        right_on="prev_missed", by="study_patient_id",
                        direction="backward", allow_exact_matches=False)
    idx["recent_missed_90d"] = ((idx["appointment_date"] - idx["prev_missed"]).dt.days <= 90).astype(int)
    idx = add_patient_columns(idx.drop(columns="prev_missed"), d["patients"]).reset_index(drop=True)
    for w in WINDOWS:
        col = f"ed_{w}d"
        idx[col] = any_visit_between(idx, d["hospital_visits"], "appointment_date", 0, w)
        idx.loc[idx["appointment_date"] > d["window_end"] - pd.Timedelta(days=w), col] = np.nan
    return idx


def referral_index(d: dict) -> pd.DataFrame:
    """One row per referral; exposure = not completed by day 60; outcome window days 60-150."""
    r = d["referrals"]
    r = r[r["referral_date"] <= d["window_end"] - pd.Timedelta(days=150)].copy()
    r["incomplete_referral"] = (~((r["referral_status"] == "Completed") & (r["days_to_status"] <= 60))).astype(int)
    r = add_patient_columns(r, d["patients"]).reset_index(drop=True)
    r["ed_60_150d"] = any_visit_between(r, d["hospital_visits"], "referral_date", 60, 150)
    return r


# ============================================================
# 2. MODELS
# ============================================================

def fit_logit(formula: str, df: pd.DataFrame):
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["study_patient_id"])[0]})


def tidy(res, **info) -> pd.DataFrame:
    ci = res.conf_int()
    t = pd.DataFrame({"term": res.params.index, "coefficient_log_odds": res.params.values,
                      "p_value": res.pvalues.values, "or": np.exp(res.params.values),
                      "ci_low": np.exp(ci.iloc[:, 0].values), "ci_high": np.exp(ci.iloc[:, 1].values)})
    t = t[t["term"] != "Intercept"].copy()
    t["label"] = (t["term"].map(LABELS)
                  .fillna(t["term"].str.replace(r"C\(insurance_type.*\)\[T\.(.*)\]", r"Insurance: \1 (vs Commercial)", regex=True)))
    for k, v in info.items():
        t.insert(0, k, v)
    t["n"] = int(res.nobs)
    return t


def run_visit_models(idx: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for w in WINDOWS:
        y = f"ed_{w}d"
        df = idx.dropna(subset=[y, "no_mychart"])
        un = fit_logit(f"{y} ~ missed_visit", df)
        adj = fit_logit(f"{y} ~ missed_visit + " + " + ".join(COVARIATES), df)
        tables.append(tidy(un, window_days=w, model="Unadjusted", exposure="Missed PCP visit"))
        tables.append(tidy(adj, window_days=w, model="Adjusted", exposure="Missed PCP visit"))
    return pd.concat(tables, ignore_index=True)


def run_referral_models(r: pd.DataFrame) -> pd.DataFrame:
    df = r.dropna(subset=["no_mychart"])
    un = fit_logit("ed_60_150d ~ incomplete_referral", df)
    adj = fit_logit("ed_60_150d ~ incomplete_referral + comorbidity_count + age65 + no_mychart"
                    " + C(insurance_type, Treatment('Commercial'))", df)
    return pd.concat([tidy(un, window_days="60-150", model="Unadjusted", exposure="Incomplete referral"),
                      tidy(adj, window_days="60-150", model="Adjusted", exposure="Incomplete referral")],
                     ignore_index=True)


def observed_rates(idx: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        y = f"ed_{w}d"
        for flag, name in [(0, "Attended visit"), (1, "Missed visit")]:
            sub = idx.loc[(idx["missed_visit"] == flag) & idx[y].notna(), y]
            rows.append({"exposure": "PCP visit", "group": name, "window": f"{w} days",
                         "n": len(sub), "share_with_ed_uc_inpatient": sub.mean()})
    for flag, name in [(0, "Referral completed by day 60"), (1, "Referral not completed by day 60")]:
        sub = r.loc[r["incomplete_referral"] == flag, "ed_60_150d"]
        rows.append({"exposure": "Referral", "group": name, "window": "days 60-150",
                     "n": len(sub), "share_with_ed_uc_inpatient": sub.mean()})
    return pd.DataFrame(rows)


# ============================================================
# 3. FIGURES
# ============================================================

def plot_confounding(t: pd.DataFrame, path: Path):
    """Missed-visit OR, unadjusted vs adjusted, for each window."""
    m = t[t["term"] == "missed_visit"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    offsets = {"Unadjusted": -0.12, "Adjusted": 0.12}
    colors = {"Unadjusted": GRAY, "Adjusted": PURPLE}
    for model in ["Unadjusted", "Adjusted"]:
        sub = m[m["model"] == model]
        x = np.arange(len(sub)) + offsets[model]
        ax.errorbar(x, sub["or"], yerr=[sub["or"] - sub["ci_low"], sub["ci_high"] - sub["or"]],
                    fmt="o", color=colors[model], ecolor=colors[model], capsize=4, markersize=7,
                    label=f"{model}" + (" (+ chronic conditions, age, insurance, MyChart, recent missed visit)"
                                        if model == "Adjusted" else " (missed visit only)"))
        for xi, v in zip(x, sub["or"]):
            ax.text(xi + 0.05, v, f"{v:.2f}", va="center", fontsize=9, color=colors[model])
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(WINDOWS)), [f"Within {w} days" for w in WINDOWS])
    ax.set_ylabel("Odds ratio: missed vs attended visit (95% CI)")
    ax.set_xlim(-0.5, len(WINDOWS) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    ax.set_title("Model 4 — ED / urgent care / hospital use after a missed PCP visit (Q7)\n" + SUBTITLE,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_forest(t: pd.DataFrame, title: str, path: Path):
    plot = t.sort_values("or").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.45 * len(plot) + 1.6)))
    y = np.arange(len(plot))
    colors = [PURPLE if term in ("missed_visit", "incomplete_referral") else GRAY for term in plot["term"]]
    for yi, (_, r), c in zip(y, plot.iterrows(), colors):
        ax.errorbar(r["or"], yi, xerr=[[r["or"] - r["ci_low"]], [r["ci_high"] - r["or"]]],
                    fmt="o", color=c, ecolor=c, capsize=3, markersize=6)
        ax.text(r["ci_high"] * 1.02, yi, f"{r['or']:.2f}", va="center", fontsize=8)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_yticks(y, plot["label"], fontsize=9)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.set_xlim(plot["ci_low"].min() * 0.9, plot["ci_high"].max() * 1.15)
    ax.set_xlabel("Adjusted odds ratio (95% CI, log scale)")
    ax.set_title(title + "\n" + SUBTITLE, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rates(rates: pd.DataFrame, path: Path):
    v = rates[rates["exposure"] == "PCP visit"]
    rr = rates[rates["exposure"] == "Referral"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    x = np.arange(len(WINDOWS))
    for k, (grp, c) in enumerate([("Attended visit", LIGHT_PURPLE), ("Missed visit", PURPLE)]):
        vals = v[v["group"] == grp]["share_with_ed_uc_inpatient"].to_numpy() * 100
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.36, color=c, label=grp)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.2, f"{val:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"Within {w} days" for w in WINDOWS])
    ax.set_ylabel("% with an ED / UC / inpatient visit")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("After a PCP visit", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    vals = rr["share_with_ed_uc_inpatient"].to_numpy() * 100
    bars = ax.bar([0, 1], vals, color=[LIGHT_PURPLE, PURPLE], width=0.6)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.2, f"{val:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["Completed\nby day 60", "Not completed\nby day 60"], fontsize=9)
    ax.set_title("After a referral (days 60–150)", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model 4 — Observed ED / urgent care / hospital use (unadjusted)\n" + SUBTITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scenario/calibrated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model4_calibrated"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    d = read_data(args.data_dir)

    print("Building index visits and referrals ...")
    idx = visit_index(d)
    ref = referral_index(d)

    print("Fitting logistic regressions ...")
    vt = run_visit_models(idx)
    rt = run_referral_models(ref)
    pd.concat([vt, rt], ignore_index=True).to_csv(out / "ed_odds_ratios.csv", index=False)
    rates = observed_rates(idx, ref)
    rates.to_csv(out / "ed_observed_rates.csv", index=False)

    plot_confounding(vt, out / "missed_visit_or_unadjusted_vs_adjusted.png")
    plot_forest(vt[(vt["window_days"] == 90) & (vt["model"] == "Adjusted")],
                "Model 4 — ED / UC / hospital use within 90 days of a PCP visit (adjusted)",
                out / "ed_90d_forest_plot.png")
    plot_forest(rt[rt["model"] == "Adjusted"],
                "Model 4 — ED / UC / hospital use 60–150 days after a referral (adjusted)",
                out / "referral_ed_forest_plot.png")
    plot_rates(rates, out / "ed_observed_rates.png")

    mv = vt[vt["term"] == "missed_visit"].set_index(["window_days", "model"])["or"]
    ir = rt[rt["term"] == "incomplete_referral"].set_index("model")["or"]
    summary = {
        "data_dir": str(args.data_dir), "data_type": "calibrated synthetic",
        "outcome": "any ED, urgent care or inpatient visit (hospital_visits.csv)",
        "index_visits": int(len(idx)), "index_patients": int(idx["study_patient_id"].nunique()),
        "referrals": int(len(ref)),
        "missed_visit_or": {f"{w}d": {"unadjusted": round(float(mv[(w, "Unadjusted")]), 3),
                                      "adjusted": round(float(mv[(w, "Adjusted")]), 3)} for w in WINDOWS},
        "incomplete_referral_or": {"unadjusted": round(float(ir["Unadjusted"]), 3),
                                   "adjusted": round(float(ir["Adjusted"]), 3)},
        "se_method": "Patient-clustered robust standard errors",
        "notes": [
            "Associations, not causal effects or NYU Langone findings.",
            "Unadjusted ORs are expected to be higher: chronic conditions drive both missed visits and ED use.",
            "Adjusted model also controls for a missed visit in the prior 90 days, so overlapping risk windows are not double counted.",
            "Visits in the last 30/90/180 days of the data window are excluded from that window's model (not enough follow-up).",
            "Q7.2 (time from ED to next PCP visit) is not estimated: the synthetic data has no return-after-ED mechanism.",
        ],
    }
    (out / "model4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["missed_visit_or", "incomplete_referral_or"]}, indent=2))
    print(f"\nSaved results to: {out.resolve()}")
    print("NOTE: calibrated SYNTHETIC data only. Do not report as real NYU Langone results.")


if __name__ == "__main__":
    main()
