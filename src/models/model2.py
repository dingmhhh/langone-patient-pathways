#!/usr/bin/env python3
"""Model 2 / Q3 + Q6 + Q9: patient pathway, access, and care gaps.

Designed for data/scenario/calibrated/ (patients, appointments, referrals,
hospital_visits). Synthetic data only; numbers are not NYU Langone findings.

What it answers
  Q6  Where do patients drop off?  One metric per drop-off point (D1-D14)
      on Paola's pathway flowchart, split new vs established (Q6.4), and
      how much of the delay is patient-driven vs system-driven (Q6.5).
  Q3  How long do patients wait, and what makes the wait longer?
      Gamma regression on days from booking to visit.
  Q9  Which patients go 12+ months with no PCP care, and do they use
      the ED / urgent care more during that gap?

Run:
  python src/models/model2.py \
      --data-dir data/scenario/calibrated \
      --output-dir results/model2_calibrated

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

MISSED = ["No-show", "Late canceled"]
PATIENT_CANCEL = ["Patient canceled"]
CLINIC_CANCEL = ["Clinic canceled"]
FOLLOWUP_GRACE_DAYS = 30
GAP_DAYS = 365

# Referral close reasons: who caused the delay (Q6.5)
PATIENT_SIDE_REASONS = {"No longer needed", "Care elsewhere", "Issue resolved"}
SYSTEM_SIDE_REASONS = {"Insurance", "No appointment available"}


# ============================================================
# 0. LOAD
# ============================================================

def to_flag(s: pd.Series) -> pd.Series:
    """True/False/1/0/blank -> 1.0 / 0.0 / NaN."""
    return (s.astype("string").str.strip().str.lower()
             .map({"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}))


def read_data(data_dir: Path) -> dict:
    need = ["patients", "appointments", "referrals", "hospital_visits"]
    missing = [n for n in need if not (data_dir / f"{n}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing in {data_dir}: {', '.join(m + '.csv' for m in missing)}")
    d = {n: pd.read_csv(data_dir / f"{n}.csv", low_memory=False) for n in need}
    if (data_dir / "orders.csv").is_file():
        d["orders"] = pd.read_csv(data_dir / "orders.csv", low_memory=False)

    p = d["patients"]
    p["mychart_active"] = to_flag(p["mychart_active"])
    p["age65"] = (p["age"] >= 65).astype(int)
    p["panel_join_date"] = pd.to_datetime(p["panel_join_date"])

    a = d["appointments"]
    a["booking_date"] = pd.to_datetime(a["booking_date"])
    a["appointment_date"] = pd.to_datetime(a["appointment_date"])
    for col in ["booked_at_checkout", "followup_booked_same_day"]:
        a[col] = to_flag(a[col]).fillna(0).astype(int)
    a = a.sort_values(["study_patient_id", "appointment_date", "booking_date"]).reset_index(drop=True)
    # New vs established patient (Q6.4): patient's first visit is a New Patient visit
    first_type = a.groupby("study_patient_id")["visit_type"].transform("first")
    a["patient_type"] = np.where(first_type == "New Patient", "New", "Established")
    d["appointments"] = a

    r = d["referrals"]
    r["referral_date"] = pd.to_datetime(r["referral_date"])
    r["prior_auth_required"] = to_flag(r["prior_auth_required"]).fillna(0).astype(int)

    d["hospital_visits"]["admission_date"] = pd.to_datetime(d["hospital_visits"]["admission_date"])
    d["window_end"] = a["appointment_date"].max()
    return d


# ============================================================
# 1. Q6: DROP-OFF METRICS (one per D point)
# ============================================================

def followup_outcomes(a: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Completed visits with a documented return interval, and whether the
    next completed visit happened within interval + grace days."""
    src = a[(a["appointment_status"] == "Completed") & a["return_interval_days"].notna()].copy()
    src["due_by"] = src["appointment_date"] + pd.to_timedelta(src["return_interval_days"] + FOLLOWUP_GRACE_DAYS, unit="D")
    src = src[src["due_by"] <= end]
    done = (a[a["appointment_status"] == "Completed"][["study_patient_id", "appointment_date"]]
            .rename(columns={"appointment_date": "next_completed"}).sort_values("next_completed"))
    src = pd.merge_asof(src.assign(_k=src["appointment_date"] + pd.Timedelta(days=1)).sort_values("_k"),
                        done, left_on="_k", right_on="next_completed",
                        by="study_patient_id", direction="forward").drop(columns="_k")
    src["on_time"] = (src["next_completed"].notna() & (src["next_completed"] <= src["due_by"])).astype(int)
    return src


def dropoff_metrics(d: dict) -> pd.DataFrame:
    a, r, hv, end = d["appointments"], d["referrals"], d["hospital_visits"], d["window_end"]
    s = a["appointment_status"]
    done = a[s == "Completed"]
    rows = []

    def add(dp, name, phase, metric, value, n, driver, note=""):
        rows.append({"drop_off": dp, "name": name, "phase": phase, "metric": metric,
                     "value": value, "denominator_n": n, "driver": driver, "note": note})

    pending = "Not in synthetic data; needs Epic field"
    add("D1", "Waits it out", "1 Need arises", "Days from preventive care due date to next PCP visit",
        np.nan, 0, "Patient", pending)

    # D2: ED/UC visits with no completed PCP visit in the prior 12 months
    hv_s = hv.sort_values("admission_date")
    prev = pd.merge_asof(hv_s, done[["study_patient_id", "appointment_date"]].sort_values("appointment_date"),
                         left_on="admission_date", right_on="appointment_date",
                         by="study_patient_id", direction="backward")
    no_pcp = prev["appointment_date"].isna() | ((prev["admission_date"] - prev["appointment_date"]).dt.days > GAP_DAYS)
    add("D2", "Goes to urgent care or ED first", "1 Need arises",
        "Share of ED/UC/inpatient visits with no PCP visit in prior 12 months", no_pcp.mean(), len(hv), "Patient")

    for dp, name, metric in [("D3", "No MyChart or no PCP listed", "MyChart status at booking"),
                             ("D4", "Not bookable online", "Share of slots open to online booking"),
                             ("D5", "Cannot get through by phone", "PAC hold time / abandoned calls"),
                             ("D6", "PAC message takes days", "Days from PAC message to practice action")]:
        add(dp, name, "2 Getting in", metric, np.nan, 0, "System", pending if dp != "D5" else "Not in Epic (phone system)")

    # D7: wait for a slot (exclude follow-ups booked at checkout: those are planned, not access waits)
    access = a[a["booked_at_checkout"] == 0]
    newp = access[access["visit_type"] == "New Patient"]
    add("D7", "No slot in the needed window", "2 Getting in",
        "Median days from booking to visit", access["lead_time_days"].median(), len(access), "System")
    add("D7", "No slot in the needed window", "2 Getting in",
        "Share of new patients seen within 10 days", (newp["lead_time_days"] <= 10).mean(), len(newp), "System",
        "Lisa's September figure: 43%")

    add("D8", "Reminder not received", "3 Before visit", "Share of visits with a reminder sent",
        np.nan, 0, "System", pending)

    # D9: missed visits and rebooking
    attended_or_missed = s.isin(["Completed"] + MISSED)
    add("D9", "No-show or late cancel", "3 Before visit",
        "No-show + late cancel rate", s.isin(MISSED).sum() / attended_or_missed.sum(), int(attended_or_missed.sum()), "Patient")
    nxt_booking = a.groupby("study_patient_id")["booking_date"].shift(-1)
    missed = a[s.isin(MISSED)]
    days_to_rebook = (nxt_booking[missed.index] - missed["appointment_date"]).dt.days
    add("D9", "No-show or late cancel", "3 Before visit",
        "Share rebooked within 60 days after a missed visit", (days_to_rebook <= 60).mean(), len(missed), "Patient")

    # D10 / D11: next step set at the visit
    add("D10", "Return interval not documented", "4 The visit",
        "Share of completed visits with return interval documented", done["return_interval_days"].notna().mean(),
        len(done), "System", "Documentation gap, not a patient loss: blank = not needed OR not entered. Dr. Rein: under half")
    doc = done[done["return_interval_days"].notna()]
    add("D11", "Follow-up not booked before leaving", "4 The visit",
        "Share of documented follow-ups booked on the visit day", doc["followup_booked_same_day"].mean(), len(doc), "Both")

    # D12 / D13: referrals
    r90 = r[r["referral_date"] <= end - pd.Timedelta(days=90)]
    add("D12", "Referral stalls", "5 After visit", "Median days referral to completed specialist visit",
        r.loc[r["referral_status"] == "Completed", "days_to_status"].median(),
        int((r["referral_status"] == "Completed").sum()), "System")
    add("D12", "Referral stalls", "5 After visit", "Share of referrals needing prior auth",
        r["prior_auth_required"].mean(), len(r), "System")
    add("D13", "Specialist visit missed or outside NYU", "5 After visit",
        "Share of referrals not completed within 90 days",
        1 - ((r90["referral_status"] == "Completed") & (r90["days_to_status"] <= 90)).mean(), len(r90), "Both")

    # D14: back to PCP after ED/UC
    nxt = pd.merge_asof(hv_s.assign(_k=hv_s["admission_date"] + pd.Timedelta(days=1)).sort_values("_k"),
                        done[["study_patient_id", "appointment_date"]].sort_values("appointment_date"),
                        left_on="_k", right_on="appointment_date", by="study_patient_id", direction="forward")
    within = (nxt["appointment_date"] - nxt["admission_date"]).dt.days
    hv_ok = hv_s["admission_date"] <= end - pd.Timedelta(days=30)
    for t in [7, 14, 30]:
        add("D14", "No PCP follow-up after urgent care or ED", "Downstream",
            f"Share with a PCP visit within {t} days after ED/UC/inpatient",
            (within[hv_ok.to_numpy()] <= t).mean(), int(hv_ok.sum()), "Both",
            "Synthetic data has no 'return after ED' mechanism, so this value is a placeholder")
    return pd.DataFrame(rows)


def dropoff_by_patient_type(d: dict) -> pd.DataFrame:
    """Q6.4: do new and established patients drop off at different points?"""
    a = d["appointments"]
    rows = []
    for ptype, sub in a.groupby("patient_type"):
        s = sub["appointment_status"]
        access = sub[sub["booked_at_checkout"] == 0]
        done = sub[s == "Completed"]
        rows.append({
            "patient_type": ptype,
            "patients": sub["study_patient_id"].nunique(),
            "D7 median days booking to visit": access["lead_time_days"].median(),
            "D9 no-show + late cancel rate": s.isin(MISSED).sum() / s.isin(["Completed"] + MISSED).sum(),
            "D10 return interval documented": done["return_interval_days"].notna().mean(),
            "D11 follow-up booked same day (if documented)": done.loc[done["return_interval_days"].notna(),
                                                                        "followup_booked_same_day"].mean(),
        })
    return pd.DataFrame(rows)


def delay_attribution(d: dict) -> pd.DataFrame:
    """Q6.5: of the appointments and referrals that did not happen as planned,
    what share was patient-driven vs system-driven?"""
    a, r = d["appointments"], d["referrals"]
    s = a["appointment_status"]
    closed = r[r["referral_status"] == "Closed"]
    rows = [
        {"step": "Appointments not attended", "driver": "Patient-driven",
         "detail": "No-show, late cancel, patient cancel", "n": int(s.isin(MISSED + PATIENT_CANCEL).sum())},
        {"step": "Appointments not attended", "driver": "System-driven",
         "detail": "Clinic canceled", "n": int(s.isin(CLINIC_CANCEL).sum())},
        {"step": "Referrals closed without a visit", "driver": "Patient-driven",
         "detail": ", ".join(sorted(PATIENT_SIDE_REASONS)),
         "n": int(closed["close_reason"].isin(PATIENT_SIDE_REASONS).sum())},
        {"step": "Referrals closed without a visit", "driver": "System-driven",
         "detail": ", ".join(sorted(SYSTEM_SIDE_REASONS)),
         "n": int(closed["close_reason"].isin(SYSTEM_SIDE_REASONS).sum())},
    ]
    out = pd.DataFrame(rows)
    out["share_within_step"] = out["n"] / out.groupby("step")["n"].transform("sum")
    return out


def pathway_stages(d: dict) -> pd.DataFrame:
    """
    Follows the structure of Paola's pathway flowchart.

    Phase 2-3  Before the visit: booked -> not canceled by clinic -> attended (D9)
    Phase 4    End of visit (B5): what next step was recorded?
               - follow-up interval recorded
               - referral or test only
               - no next step recorded: either not needed ("loop closed") or
                 needed but not documented (D10). Epic cannot tell these
                 apart, so this group is NOT counted as a drop-off.
    Phase 5    Follow-up lane, visits with a recorded interval only:
               booked before leaving (D11), and completed on time.
    """
    a, end = d["appointments"], d["window_end"]
    booked = a[a["appointment_date"] <= end - pd.Timedelta(days=455)]   # leave time to observe the follow-up
    s = booked["appointment_status"]
    rows = []

    def add(phase, step, n, denom, note=""):
        rows.append({"phase": phase, "step": step, "n": int(n), "denominator": int(denom),
                     "share": n / denom if denom else np.nan, "note": note})

    n0 = len(booked)
    add("Before the visit", "Appointment booked", n0, n0)
    add("Before the visit", "Not canceled by clinic", (~s.isin(CLINIC_CANCEL)).sum(), n0, "system-driven loss")
    attended = booked[s == "Completed"]
    add("Before the visit", "Patient attended", len(attended), n0, "D9: no-show / cancel")

    # B5: next step recorded at the end of the visit
    ref_src = set(d["referrals"]["source_encounter_id"])
    if "orders" in d:
        ref_src |= set(d["orders"]["source_encounter_id"])
    has_fu = attended["return_interval_days"].notna()
    has_ref = attended["encounter_id"].isin(ref_src)
    n_att = len(attended)
    add("End of visit (B5)", "Follow-up interval recorded", has_fu.sum(), n_att)
    add("End of visit (B5)", "Referral or test only", (~has_fu & has_ref).sum(), n_att)
    add("End of visit (B5)", "No next step recorded", (~has_fu & ~has_ref).sum(), n_att,
        "not needed (loop closed) OR not documented (D10); not a drop-off")

    # Follow-up lane
    fu = followup_outcomes(a, end)
    fu = fu[fu["encounter_id"].isin(attended["encounter_id"])]
    n_fu = len(fu)
    same = fu["followup_booked_same_day"] == 1
    add("Follow-up lane", "Booked before leaving", same.sum(), n_fu, "D11 if not")
    add("Follow-up lane", "Completed on time: booked before leaving", fu.loc[same, "on_time"].sum(), same.sum())
    add("Follow-up lane", "Completed on time: booked later", fu.loc[~same, "on_time"].sum(), (~same).sum())
    add("Follow-up lane", "Completed on time: all recorded follow-ups", fu["on_time"].sum(), n_fu)
    return pd.DataFrame(rows)


# ============================================================
# 2. Q3: WAIT TIME REGRESSION
# ============================================================

def wait_time_model(d: dict):
    a, p, end = d["appointments"], d["patients"], d["window_end"]
    df = a[(a["booked_at_checkout"] == 0) & (a["booking_date"] <= end - pd.Timedelta(days=180))].copy()
    df = df.merge(p[["study_patient_id", "age65", "insurance_type", "panel_size"]],
                  on="study_patient_id", how="left", validate="many_to_one")
    panel_median = p.groupby("pcp_id")["panel_size"].first().median()   # median across PCPs, not patients
    df["large_panel"] = (df["panel_size"] > panel_median).astype(int)
    formula = ("lead_time_days ~ C(visit_type, Treatment('New Patient'))"
               " + C(booking_channel, Treatment('PAC')) + large_panel + age65"
               " + C(insurance_type, Treatment('Commercial'))")
    res = smf.glm(formula, data=df, family=sm.families.Gamma(link=sm.families.links.Log())).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["study_patient_id"])[0]})
    ci = res.conf_int()
    table = pd.DataFrame({"term": res.params.index, "coefficient_log": res.params.values,
                          "p_value": res.pvalues.values, "ratio": np.exp(res.params.values),
                          "ci_low": np.exp(ci.iloc[:, 0].values), "ci_high": np.exp(ci.iloc[:, 1].values)})
    table["label"] = (table["term"]
                      .str.replace(r"C\(visit_type.*\)\[T\.(.*)\]", r"Visit type: \1 (vs New Patient)", regex=True)
                      .str.replace(r"C\(booking_channel.*\)\[T\.(.*)\]", r"Booked via \1 (vs PAC)", regex=True)
                      .str.replace(r"C\(insurance_type.*\)\[T\.(.*)\]", r"Insurance: \1 (vs Commercial)", regex=True)
                      .replace({"large_panel": "PCP panel above median size", "age65": "Age 65+"}))
    return table[table["term"] != "Intercept"].reset_index(drop=True), res, df


def access_by_pcp(d: dict) -> pd.DataFrame:
    """Q3.1 / Q3.3 / Q3.4: capacity and access per PCP panel."""
    a, p = d["appointments"], d["patients"]
    access = a[a["booked_at_checkout"] == 0]
    out = pd.DataFrame({
        "panel_patients": p.groupby("pcp_id").size(),
        "share_visits_new_patient": a.groupby("pcp_id")["visit_type"].apply(lambda x: (x == "New Patient").mean()),
        "median_wait_days": access.groupby("pcp_id")["lead_time_days"].median(),
        "clinic_cancel_rate": a.groupby("pcp_id")["appointment_status"].apply(lambda x: x.isin(CLINIC_CANCEL).mean()),
    }).reset_index().rename(columns={"index": "pcp_id"})
    return out


# ============================================================
# 3. Q9: 12-MONTH GAPS IN PCP CARE
# ============================================================

def snapshot_gaps(d: dict, months: int = 12) -> pd.DataFrame:
    """
    Q9.1 at fixed dates (data plan definition): share of panel patients with
    no completed PCP visit in the previous N months AND nothing scheduled.
    Only patients first seen at least N months before the snapshot count.
    """
    a, end = d["appointments"], d["window_end"]
    window = pd.Timedelta(days=round(months * 30.44))
    first = a.groupby("study_patient_id")["booking_date"].min()
    done = a[a["appointment_status"] == "Completed"]
    rows = []
    snaps = [pd.Timestamp(f"{y}-10-01") for y in range(2022, end.year + 1)] + [end]
    for snap in sorted(set(x for x in snaps if x <= end)):
        eligible = first[first <= snap - window].index
        if len(eligible) < 100:          # too few patients followed long enough
            continue
        recent = done[(done["appointment_date"] > snap - window) & (done["appointment_date"] <= snap)]["study_patient_id"]
        pending = a[(a["booking_date"] <= snap) & (a["appointment_date"] > snap)]["study_patient_id"]
        covered = set(recent) | set(pending)
        n_gap = sum(pid not in covered for pid in eligible)
        rows.append({"snapshot_date": snap.date(), "months": months, "eligible_patients": len(eligible),
                     "patients_in_gap": n_gap, "share_in_gap": n_gap / len(eligible)})
    return pd.DataFrame(rows)


def care_gaps(d: dict, gap_days: int = GAP_DAYS):
    """
    Q9.2: ED / urgent care use during gap time vs time in PCP care.

    A patient is 'in care' on a day if they completed a PCP visit in the
    previous 12 months or have an appointment booked for the future
    (data plan definition). Otherwise they are 'in a gap'.
    """
    a, p, hv, end = d["appointments"], d["patients"], d["hospital_visits"], d["window_end"]
    gaps, person = [], []
    for pid, sub in a.groupby("study_patient_id"):
        comp = sub.loc[sub["appointment_status"] == "Completed", "appointment_date"]
        iv = list(zip(comp, comp + pd.Timedelta(days=gap_days)))     # 12 months after each completed visit
        iv += list(zip(sub["booking_date"], sub["appointment_date"]))   # booked, waiting for the visit
        iv.sort()
        start = sub["booking_date"].min() + pd.Timedelta(days=gap_days)
        if start >= end:
            continue
        cur = start
        for lo, hi in iv:
            if hi <= cur:
                continue
            if lo > cur:
                gaps.append((pid, cur, min(lo, end)))
            cur = max(cur, hi)
            if cur >= end:
                break
        if cur < end:
            gaps.append((pid, cur, end))
        person.append((pid, start, (end - start).days))
    g = pd.DataFrame(gaps, columns=["study_patient_id", "gap_start", "gap_end"])
    g["gap_days"] = (g["gap_end"] - g["gap_start"]).dt.days
    g = g[g["gap_days"] > 0]
    followed = pd.DataFrame(person, columns=["study_patient_id", "start", "followup_days"])

    hv_f = hv.merge(followed[["study_patient_id", "start"]], on="study_patient_id")
    hv_f = hv_f[hv_f["admission_date"] > hv_f["start"]]
    in_gap = hv_f.merge(g, on="study_patient_id")
    in_gap = in_gap[(in_gap["admission_date"] > in_gap["gap_start"]) & (in_gap["admission_date"] <= in_gap["gap_end"])]
    gap_years = g["gap_days"].sum() / 365.25
    total_years = followed["followup_days"].sum() / 365.25
    ed_gap = in_gap["hospital_encounter_id"].nunique()
    rate_gap = ed_gap / gap_years
    rate_care = (len(hv_f) - ed_gap) / (total_years - gap_years)

    snap12, snap24 = snapshot_gaps(d, 12), snapshot_gaps(d, 24)
    latest12, latest24 = snap12.iloc[-1], snap24.iloc[-1]
    in_gap_now = set()
    a_end = d["window_end"]
    recent = a[(a["appointment_status"] == "Completed") & (a["appointment_date"] > a_end - pd.Timedelta(days=gap_days))]
    first = a.groupby("study_patient_id")["booking_date"].min()
    eligible = first[first <= a_end - pd.Timedelta(days=gap_days)].index
    in_gap_now = pd.Series(~eligible.isin(set(recent["study_patient_id"])), index=eligible).astype(int)
    profile = (p.set_index("study_patient_id").loc[eligible].assign(in_gap=in_gap_now.values)
               .groupby("in_gap")
               .agg(patients=("age", "size"), mean_age=("age", "mean"), share_age65=("age65", "mean"),
                    share_no_mychart=("mychart_active", lambda x: 1 - x.mean()),
                    mean_comorbidities=("comorbidity_count", "mean"))
               .rename(index={0: "Seen in last 12 months", 1: "No PCP visit in last 12 months"}))

    summary = pd.DataFrame([
        {"metric": f"Share of panel patients with no PCP visit in 12 months and nothing scheduled (as of {latest12['snapshot_date']})",
         "value": latest12["share_in_gap"]},
        {"metric": f"Average of yearly snapshots (12 months)", "value": snap12["share_in_gap"].mean()},
        {"metric": f"Share with no PCP visit in 24 months and nothing scheduled (as of {latest24['snapshot_date']})",
         "value": latest24["share_in_gap"]},
        {"metric": "ED/UC/inpatient visits per patient-year during gaps", "value": rate_gap},
        {"metric": "ED/UC/inpatient visits per patient-year while in PCP care", "value": rate_care},
        {"metric": "Rate ratio (gap vs in care)", "value": rate_gap / rate_care},
    ])
    snaps = pd.concat([snap12, snap24], ignore_index=True)
    return summary, profile.reset_index(), snaps


# ============================================================
# 4. FIGURES
# ============================================================

def plot_dropoffs(m: pd.DataFrame, path: Path):
    """Every bar = share of patients LOST or STALLED at that drop-off point (higher = worse)."""
    def val(text):
        return float(m.loc[m["metric"].str.contains(text, regex=False), "value"].iloc[0])
    rows = [
        ("D2", "ED/UC visits by patients with no PCP visit in prior 12 months", val("no PCP visit in prior 12 months"), "Patient"),
        ("D7", "New patients NOT seen within 10 days", 1 - val("within 10 days"), "System"),
        ("D9", "Booked visits missed (no-show or late cancel)", val("No-show + late cancel rate"), "Patient"),
        ("D9", "Missed visits NOT rebooked within 60 days", 1 - val("rebooked within 60 days"), "Patient"),
        ("D11", "Documented follow-ups NOT booked before leaving", 1 - val("booked on the visit day"), "Both"),
        ("D13", "Referrals NOT completed within 90 days", val("not completed within 90 days"), "Both"),
    ]
    df = pd.DataFrame(rows, columns=["dp", "label", "value", "driver"]).iloc[::-1]
    colors = {"Patient": LIGHT_PURPLE, "System": PURPLE, "Both": GRAY}
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(df) + 1.9))
    y = np.arange(len(df))
    ax.barh(y, df["value"] * 100, color=[colors[x] for x in df["driver"]])
    for yi, v in zip(y, df["value"]):
        ax.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=9)
    ax.set_yticks(y, [f"{dp}  {lb}" for dp, lb in zip(df["dp"], df["label"])], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% lost or stalled at this point (higher = worse)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, ["Patient-driven", "System-driven", "Both"], loc="lower right", fontsize=8)
    ax.set_title("Model 2 — Where patients drop off on the pathway (Q6)\n" + SUBTITLE, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.text(0, -0.13, "Not shown: D10 (documentation gap, not a loss; see pathway chart); D1, D3-D6, D8 need Epic fields; D14 not simulated.",
            transform=ax.transAxes, fontsize=7.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pathway(st: pd.DataFrame, path: Path):
    """Three panels in the order of the pathway flowchart."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), gridspec_kw={"height_ratios": [3, 1.3, 2.6]})

    # Panel 1: before the visit
    ax = axes[0]
    p1 = st[st["phase"] == "Before the visit"].reset_index(drop=True)
    y = np.arange(len(p1))[::-1]
    for yi, (_, r) in zip(y, p1.iterrows()):
        ax.barh(yi, r["share"] * 100, color=PURPLE, height=0.6)
        ax.text(r["share"] * 100 + 1, yi, f"{r['step']}  {r['share']:.0%}", va="center", fontsize=10, fontweight="bold")
        if r["note"]:
            lost = p1["share"].iloc[_ - 1] - r["share"] if _ > 0 else 0
            ax.text(r["share"] * 100 + 1, yi - 0.32, f"−{lost:.0%} of booked: {r['note']}", va="center", fontsize=8, color=GRAY)
    ax.set_xlim(0, 140); ax.set_ylim(-0.7, len(p1) - 0.4); ax.axis("off")
    ax.set_title("Phases 2–3 · Before the visit  (% of booked appointments)", loc="left", fontsize=10, fontweight="bold")

    # Panel 2: end of visit, stacked
    ax = axes[1]
    p2 = st[st["phase"] == "End of visit (B5)"].reset_index(drop=True)
    colors = [PURPLE, "#8E63B5", "#D9D9D9"]
    left = 0
    for (_, r), c in zip(p2.iterrows(), colors):
        ax.barh(0, r["share"] * 100, left=left, color=c, height=0.55)
        ax.text(left + r["share"] * 50, 0, f"{r['share']:.0%}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white" if c != "#D9D9D9" else "black")
        ax.text(left + r["share"] * 50, -0.55, r["step"], ha="center", va="center", fontsize=8.5)
        left += r["share"] * 100
    ax.text(100, 0.55, "Grey = not needed (loop closed) or not documented (D10). Epic cannot tell them apart; not counted as drop-off.",
            ha="right", fontsize=7.5, color=GRAY)
    ax.set_xlim(0, 100); ax.set_ylim(-0.9, 0.8); ax.axis("off")
    ax.set_title("Phase 4 · Next step recorded at the end of the visit (B5)  (% of attended visits)",
                 loc="left", fontsize=10, fontweight="bold")

    # Panel 3: follow-up lane
    ax = axes[2]
    p3 = st[st["phase"] == "Follow-up lane"].set_index("step")
    booked = p3.loc["Booked before leaving", "share"]
    ot_same = p3.loc["Completed on time: booked before leaving", "share"]
    ot_later = p3.loc["Completed on time: booked later", "share"]
    ot_all = p3.loc["Completed on time: all recorded follow-ups", "share"]
    labels = ["Booked before leaving", "Booked later (D11)"]
    shares = [booked, 1 - booked]
    ontime = [ot_same, ot_later]
    y = np.array([1, 0])
    ax.barh(y, [x * 100 for x in shares], color=LIGHT_PURPLE, height=0.55)
    ax.barh(y, [x * o * 100 for x, o in zip(shares, ontime)], color=PURPLE, height=0.55)
    for yi, lb, sh, ot in zip(y, labels, shares, ontime):
        ax.text(-1, yi, f"{lb}\n{sh:.0%} of follow-ups", ha="right", va="center", fontsize=9)
        ax.text(sh * 100 + 1, yi, f"{ot:.0%} completed on time", va="center", fontsize=9, fontweight="bold")
    ax.text(0, -0.75, f"Dark = completed within interval + {FOLLOWUP_GRACE_DAYS} days.  All recorded follow-ups: {ot_all:.0%} on time.",
            fontsize=8, color=GRAY)
    ax.set_xlim(-38, 100); ax.set_ylim(-1, 1.6); ax.axis("off")
    ax.set_title("Phase 5 · Follow-up lane  (visits with a follow-up interval recorded)", loc="left", fontsize=10, fontweight="bold")

    fig.suptitle("Model 2 — Patient pathway: before, during and after the visit\n" + SUBTITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_attribution(att: pd.DataFrame, path: Path):
    piv = att.pivot(index="step", columns="driver", values="share_within_step")[["Patient-driven", "System-driven"]]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    left = np.zeros(len(piv))
    for col, color in [("Patient-driven", LIGHT_PURPLE), ("System-driven", PURPLE)]:
        ax.barh(piv.index, piv[col] * 100, left=left, color=color, label=col)
        for i, (l, v) in enumerate(zip(left, piv[col])):
            ax.text(l + v * 50, i, f"{v:.0%}", ha="center", va="center",
                    color="black" if color == LIGHT_PURPLE else "white", fontsize=10, fontweight="bold")
        left += piv[col].to_numpy() * 100
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of events that did not happen as planned")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    ax.set_title("Model 2 — Patient-driven vs system-driven (Q6.5)\n" + SUBTITLE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_wait_forest(t: pd.DataFrame, path: Path):
    plot = t.sort_values("ratio").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(plot) + 1.6)))
    y = np.arange(len(plot))
    ax.errorbar(plot["ratio"], y, xerr=[plot["ratio"] - plot["ci_low"], plot["ci_high"] - plot["ratio"]],
                fmt="o", color=PURPLE, ecolor=PURPLE, capsize=3, markersize=5)
    for yi, (_, r) in zip(y, plot.iterrows()):
        ax.text(r["ci_high"] * 1.015, yi, f"{(r['ratio'] - 1):+.0%}", va="center", fontsize=8)
    ax.set_xlim(plot["ci_low"].min() * 0.95, plot["ci_high"].max() * 1.12)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_yticks(y, plot["label"], fontsize=9)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.set_xlabel("Ratio of mean wait (95% CI); label = % longer or shorter wait")
    ax.set_title("Model 2 — What makes patients wait longer for a PCP visit (Q3)\n" + SUBTITLE, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gap_rates(summary: pd.DataFrame, path: Path):
    v = summary.set_index("metric")["value"]
    gap = v["ED/UC/inpatient visits per patient-year during gaps"]
    cov = v["ED/UC/inpatient visits per patient-year while in PCP care"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["In PCP care", "During a 12-month+ gap"], [cov, gap], color=[LIGHT_PURPLE, PURPLE], width=0.55)
    for b, val in zip(bars, [cov, gap]):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("ED / urgent care / inpatient visits\nper patient-year")
    ax.set_title(f"Model 2 — ED use during gaps in PCP care (Q9)\nRate ratio {gap / cov:.1f}× · " + SUBTITLE,
                 fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scenario/calibrated"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model2_calibrated"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    d = read_data(args.data_dir)

    print("Q6 drop-off metrics ...")
    m = dropoff_metrics(d)
    m.to_csv(out / "dropoff_metrics.csv", index=False)
    plot_dropoffs(m, out / "dropoff_metrics.png")
    dropoff_by_patient_type(d).to_csv(out / "dropoff_by_patient_type.csv", index=False)
    att = delay_attribution(d)
    att.to_csv(out / "delay_attribution.csv", index=False)
    plot_attribution(att, out / "delay_attribution.png")
    st = pathway_stages(d)
    st.to_csv(out / "pathway_stages.csv", index=False)
    plot_pathway(st, out / "pathway_stages.png")

    print("Q3 wait time regression ...")
    wt, res, wdf = wait_time_model(d)
    wt.to_csv(out / "wait_time_ratios.csv", index=False)
    plot_wait_forest(wt, out / "wait_time_forest_plot.png")
    access_by_pcp(d).to_csv(out / "access_by_pcp.csv", index=False)

    print("Q9 care gaps ...")
    gs, gp, snaps = care_gaps(d)
    gs.to_csv(out / "care_gap_summary.csv", index=False)
    snaps.to_csv(out / "care_gap_snapshots.csv", index=False)
    gp.to_csv(out / "care_gap_patient_profile.csv", index=False)
    plot_gap_rates(gs, out / "care_gap_ed_rate.png")

    summary = {
        "data_dir": str(args.data_dir), "data_type": "calibrated synthetic",
        "questions": {"Q6": "drop-off metrics D1-D14, new vs established, patient vs system",
                      "Q3": "wait time Gamma GLM (log link), patient-clustered SE",
                      "Q9": "no completed PCP visit in 12 (24) months and nothing scheduled, at yearly snapshots"},
        "wait_time_model": {"n": int(res.nobs), "patients": int(wdf["study_patient_id"].nunique()),
                            "excluded": "follow-ups booked at checkout (planned, not access waits); bookings in last 180 days"},
        "headline": {
            "no_show_plus_late_cancel_rate": float(m.loc[m.metric.str.startswith("No-show"), "value"].iloc[0]),
            "new_patients_seen_within_10_days": float(m.loc[m.metric.str.contains("within 10 days"), "value"].iloc[0]),
            "share_with_12_month_gap_latest_snapshot": float(gs["value"].iloc[0]),
            "ed_rate_ratio_gap_vs_in_care": float(gs["value"].iloc[-1]),
        },
        "notes": [
            "Descriptive metrics and associations, not causal effects or NYU Langone findings.",
            "D1, D3-D6, D8 need Epic fields that the synthetic data does not have.",
            "D14 is a placeholder: the synthetic data has no 'return to PCP after ED' mechanism.",
            "D10 is a documentation gap, not a drop-off: a blank return interval can mean no follow-up needed or not entered.",
            "Synthetic data simplification: every completed visit has a true next visit; 'no follow-up needed' is not simulated.",
            "Shares in the drop-off chart use different denominators; see denominator_n in the CSV.",
        ],
    }
    (out / "model2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["headline"], indent=2))
    print(f"\nSaved results to: {out.resolve()}")
    print("NOTE: calibrated SYNTHETIC data only. Do not report as real NYU Langone results.")


if __name__ == "__main__":
    main()
