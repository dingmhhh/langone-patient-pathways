"""
NYU Langone Capstone - do the models recover the planted effects?

Runs every model in src/scenario_models.py on each folder in
data/scenario/ and compares the estimates with the values
planted in that folder's scenario_params.json.

Usage (from the project root):
    python src/validate_scenarios.py                    # all scenarios
    python src/validate_scenarios.py --scenario calibrated

Outputs (results/scenario_validation/):
    recovery_table.csv          planted vs estimated, every term
    pathway_metrics.csv         Q6 drop-off metrics + Q9
    q4_km_by_prior_auth.csv     referral completion by day 30/60/90
    q8_top_decile_profile.csv   who sits in the top 10% risk
    <scenario>_recovery.png     planted vs estimated chart
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import scenario_models as sm_  # noqa: E402

SCENARIO_ROOT = PROJECT_ROOT / "data" / "scenario"
OUT_DIR = PROJECT_ROOT / "results" / "scenario_validation"
NYU_PURPLE = "#57068C"

INS = "C(insurance_type, Treatment('Commercial'))"
MODE = "C(visit_mode, Treatment('Office'))"
VTYPE1 = "C(visit_type, Treatment('Annual Physical'))"
VTYPE3 = "C(visit_type, Treatment('New Patient'))"
SPEC = "C(specialty, Treatment('Cardiology'))"


def planted_lookup(P):
    """(model, term) -> (planted value, exact_by_design)."""
    q1, q2, q3 = P["q1_attendance"], P["q2_followup"], P["q3_access"]
    q4, q5, q7 = P["q4_referrals"], P["q5_messages"], P["q7_utilization"]
    hs = q4["hr_specialty"]
    t = {}
    m = "No-show logistic"
    t.update({(m, "lead_time_30"): q1["or_lead_time_per_30d"], (m, "no_mychart"): q1["or_no_mychart"],
              (m, "age65"): 1.0, (m, "no_mychart:age65"): q1["or_age65_x_no_mychart"],
              (m, "prior_no_show"): q1["or_prior_no_show"],
              (m, f"{INS}[T.Medicaid]"): q1["or_medicaid"], (m, f"{INS}[T.Medicare]"): 1.0,
              (m, f"{INS}[T.Other]"): 1.0, (m, "sdoh_transportation"): q1["or_transportation"],
              (m, "comorbidity_count"): q1["or_comorbidity_per_condition"], (m, "female"): q1["or_female"],
              (m, f"{MODE}[T.Phone]"): q1["or_phone"], (m, f"{MODE}[T.Video]"): q1["or_video"],
              (m, f"{VTYPE1}[T.Follow-up]"): 1.0, (m, f"{VTYPE1}[T.New Patient]"): 1.0})
    m = "Follow-up on time: patient factors"
    t.update({(m, "asymptomatic_chronic"): q2["or_asymptomatic_chronic"],
              (m, "sdoh_social_isolation"): q2["or_social_isolation"],
              (m, "sdoh_low_health_literacy"): q2["or_low_health_literacy"],
              (m, "medicaid"): q2["or_medicaid"]})
    m = "Preventive care on time"
    t.update({(m, "sdoh_social_isolation"): q2["or_preventive_social_isolation"],
              (m, "sdoh_low_health_literacy"): q2["or_preventive_low_health_literacy"],
              (m, "no_mychart"): q2["or_preventive_no_mychart"]})
    m = "Tests completed in 60 days"
    t.update({(m, "prior_auth_required"): q2["or_order_prior_auth"],
              (m, "sdoh_low_health_literacy"): q2["or_order_low_health_literacy"]})
    m = "Wait time Gamma GLM"
    t.update({(m, f"{VTYPE3}[T.Follow-up]"): q3["mult_follow_up"],
              (m, f"{VTYPE3}[T.Annual Physical]"): q3["mult_annual_physical"],
              (m, "large_panel"): q3["mult_large_panel"], (m, "mychart_booking"): q3["mult_mychart_booking"],
              (m, "age65"): q3["mult_age65"]})
    m = "Referral completion Cox PH"
    t.update({(m, "prior_auth_required"): q4["hr_prior_auth"], (m, "urgent"): q4["hr_urgent"],
              (m, "medicaid"): q4["hr_medicaid"]})
    for s in hs:
        if s != "Cardiology":
            t[(m, f"{SPEC}[T.{s}]")] = hs[s] / hs["Cardiology"]
    m = "Message response (median hours)"
    t.update({(m, "weekend_extra_hours"): q5["weekend_delay_hours"], (m, "pac_extra_hours"): q5["pac_delay_hours"]})
    for m in ["ED/UC/inpatient 90d: unadjusted", "ED/UC/inpatient 90d: adjusted"]:
        t[(m, "missed_visit")] = q7["or_missed_visit"]
    m = "ED/UC/inpatient 90d: adjusted"
    t.update({(m, "comorbidity_count"): q7["or_comorbidity_per_condition"], (m, "no_mychart"): q7["or_no_mychart"]})
    m = "ED/UC/inpatient after incomplete referral"
    t.update({(m, "incomplete_referral"): q7["or_incomplete_referral"],
              (m, "comorbidity_count"): q7["or_comorbidity_per_condition"]})
    return t


# Where the simulation does not plant the value one-to-one
NOTES = {
    "Follow-up on time: patient factors": "approximate: planted on the return step, outcome also needs attendance",
    "ED/UC/inpatient 90d: unadjusted": "expected to be biased upward: comorbidity confounds",
    "ED/UC/inpatient 90d: adjusted": "approximate: planted as a rate ratio, estimated as an odds ratio",
    "ED/UC/inpatient after incomplete referral": "approximate: planted as a rate ratio, estimated as an odds ratio",
}


def run_scenario(folder):
    P = json.loads((folder / "scenario_params.json").read_text())["params"]
    d = sm_.load_data(folder)
    parts = []
    parts.append(sm_.q1_no_show(d)[0])
    parts.append(sm_.q2_followup(d)[0])
    parts.append(sm_.q2_preventive(d)[0])
    parts.append(sm_.q2_orders(d)[0])
    parts.append(sm_.q3_wait_time(d)[0])
    parts.append(sm_.q4_referral_cox(d)[0])
    parts.append(sm_.q5_messages(d))
    parts.append(sm_.q7_utilization(d)[0])
    q8, profile = sm_.q8_risk(d)
    parts.append(q8)
    table = pd.concat(parts, ignore_index=True)

    truth = planted_lookup(P)
    table["planted"] = [truth.get((m, t), np.nan) for m, t in zip(table["model"], table["term"])]
    table["covered"] = np.where(table["planted"].notna() & table["ci_low"].notna(),
                                (table["ci_low"] <= table["planted"]) & (table["planted"] <= table["ci_high"]), np.nan)
    table["note"] = table["model"].map(NOTES).fillna("")
    table.insert(0, "scenario", folder.name)

    pathway = sm_.q6_q9_pathway_metrics(d)
    pathway["planted"] = np.nan
    pathway.loc[pathway["drop_off"] == "D10", "planted"] = P["q2_followup"]["p_documented"]
    pathway.loc[pathway["drop_off"] == "D11", "planted"] = P["q2_followup"]["p_same_day_if_documented"]
    pathway.insert(0, "scenario", folder.name)
    km = sm_.q4_km_table(d).assign(scenario=folder.name)
    profile.insert(0, "scenario", folder.name)
    return table, pathway, km, profile


def plot_recovery(table, scenario, path):
    df = table[table["planted"].notna() & table["ci_low"].notna()
               & (table["kind"] != "hours")].copy()
    short = {"Follow-up on time: patient factors": "follow-up", "Preventive care on time": "preventive",
             "Tests completed in 60 days": "tests", "ED/UC/inpatient 90d: unadjusted": "ED 90d unadj.",
             "ED/UC/inpatient 90d: adjusted": "ED 90d adj.", "ED/UC/inpatient after incomplete referral": "ED after referral"}
    df["label"] = (df["question"] + " " + df["model"].map(short).fillna("").str.cat(["  "] * len(df))
                   + df["term"].str.replace(r"C\((\w+).*\)\[T\.(.*)\]", r"\1: \2", regex=True))
    df["label"] = df["label"].str.replace(r"\s+", " ", regex=True)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.errorbar(df["estimate"], y, xerr=[df["estimate"] - df["ci_low"], df["ci_high"] - df["estimate"]],
                fmt="o", color=NYU_PURPLE, ecolor=NYU_PURPLE, capsize=3, markersize=4, label="Estimate (95% CI)")
    ax.scatter(df["planted"], y, marker="|", s=180, color="#D9480F", zorder=3, label="Planted value")
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Ratio (OR, HR or ratio of means), log scale")
    ax.set_title(f"Planted vs estimated effects: '{scenario}' scenario\nSynthetic data, pipeline validation only",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario")
    args = parser.parse_args()
    folders = sorted(p for p in SCENARIO_ROOT.iterdir() if (p / "scenario_params.json").exists())
    if args.scenario:
        folders = [f for f in folders if f.name == args.scenario]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables, paths, kms, profiles = [], [], [], []
    for folder in folders:
        table, pathway, km, profile = run_scenario(folder)
        plot_recovery(table, folder.name, OUT_DIR / f"{folder.name}_recovery.png")
        tables.append(table); paths.append(pathway); kms.append(km); profiles.append(profile)
        chk = table[table["covered"].notna()]
        exact = chk[chk["note"] == ""]
        print(f"\n[{folder.name}] planted value inside 95% CI: "
              f"{int(exact['covered'].sum())}/{len(exact)} exact-by-design terms, "
              f"{int(chk['covered'].sum())}/{len(chk)} all terms")
        for _, r in table[table["planted"].notna()].iterrows():
            flag = "" if r["covered"] in (True, 1.0) or pd.isna(r["covered"]) else "  <-- outside CI"
            ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]" if pd.notna(r["ci_low"]) else ""
            print(f"  {r['question']} {r['model'][:34]:<34} {r['term'][:46]:<46} planted {r['planted']:>6.2f}  est {r['estimate']:>6.2f} {ci}{flag}")

    pd.concat(tables).to_csv(OUT_DIR / "recovery_table.csv", index=False)
    pd.concat(paths).to_csv(OUT_DIR / "pathway_metrics.csv", index=False)
    pd.concat(kms).to_csv(OUT_DIR / "q4_km_by_prior_auth.csv", index=False)
    pd.concat(profiles).to_csv(OUT_DIR / "q8_top_decile_profile.csv", index=False)
    print(f"\nSaved to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
