"""
NYU Langone Capstone - scenario-based synthetic data

Builds one simulated patient population per scenario in
config/scenarios.yaml and writes it to data/scenario/<name>/.

Unlike generate_mock_data.py (independent random columns),
records are simulated patient by patient in time order, with
the effects in the config planted on purpose, so the
regression models have something known to recover.

Usage (from the project root):
    python src/generate_scenario_data.py                    # all scenarios
    python src/generate_scenario_data.py --scenario calibrated

Outputs per scenario:
    patients.csv, appointments.csv, referrals.csv,
    hospital_visits.csv, orders.csv, preventive_care.csv,
    messages.csv, scenario_params.json (the planted truth)

IMPORTANT: all values are assumptions for testing the
pipeline, not findings about NYU Langone patients.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "scenarios.yaml"

DAY = pd.Timedelta(days=1)


# ============================================================
# 0. HELPERS
# ============================================================

def deep_merge(base, override):
    """Return base updated with override (nested dicts merged)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scenarios = {
        name: deep_merge(cfg["base"], override)
        for name, override in cfg["scenarios"].items()
    }
    return cfg["settings"], scenarios


def lg(x):
    return float(np.log(x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return float(np.log(p / (1 - p)))


def solve_intercept(linear_part, target):
    """Find b so that mean(sigmoid(b + linear_part)) == target."""
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(mid + linear_part).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def pick(rng, probs_dict, size=None):
    keys = list(probs_dict.keys())
    p = np.array(list(probs_dict.values()), dtype=float)
    return rng.choice(keys, size=size, p=p / p.sum())


def masked(values, observed):
    """0/1 column that is blank where the item was not observed."""
    arr = pd.array(values.astype(int), dtype="Int64")
    arr[~observed] = pd.NA
    return arr


# ============================================================
# 1. PATIENTS
# ============================================================

def make_patients(rng, P, n, start, end):
    pp = P["patients"]
    sd = pp["sdoh"]

    age = rng.integers(18, 91, n)
    age65 = age >= 65
    sex = pick(rng, pp["sex_probs"], n)
    race = pick(rng, pp["race_probs"], n)
    ethnicity = np.where(
        rng.random(n) < pp["hispanic_share"],
        "Hispanic or Latino", "Not Hispanic or Latino"
    )
    language = pick(rng, pp["language_probs"], n)
    interpreter = (language != "English") & (
        rng.random(n) < pp["interpreter_share_non_english"]
    )
    pcp = pick(rng, pp["pcp_probs"], n)
    large_panel = np.isin(pcp, pp["large_panels"])

    ins_young = pick(rng, pp["insurance_probs_under65"], n)
    ins_old = pick(rng, pp["insurance_probs_65plus"], n)
    insurance = np.where(age65, ins_old, ins_young)
    medicaid = insurance == "Medicaid"

    p_mc = np.where(age65, pp["mychart_rate_65plus"], pp["mychart_rate_under65"])
    p_mc = np.where(interpreter, p_mc * pp["mychart_interpreter_mult"], p_mc)
    mychart = rng.random(n) < p_mc

    # True social drivers (used by the simulation)
    transport = rng.random(n) < np.where(medicaid, sd["transportation_medicaid"], sd["transportation"])
    financial = rng.random(n) < np.where(medicaid, sd["financial_strain_medicaid"], sd["financial_strain"])
    housing = rng.random(n) < sd["housing_instability"]
    isolation = rng.random(n) < np.where(age65, sd["social_isolation_65plus"], sd["social_isolation"])
    low_lit = rng.random(n) < np.where(interpreter, sd["low_health_literacy_interpreter"], sd["low_health_literacy"])
    screened = rng.random(n) < pp["sdoh_screening_rate"]

    comorb = np.minimum(
        rng.poisson(pp["comorbidity_base"] + pp["comorbidity_per_year"] * (age - 18)), 5
    )
    asymptomatic = (comorb > 0) & (rng.random(n) < pp["asymptomatic_share"])

    is_new = rng.random(n) < pp["new_patient_share"]
    span = (end - start).days - 90
    join = start + pd.to_timedelta(np.where(is_new, rng.integers(0, span, n), 0), unit="D")

    patients = pd.DataFrame({
        "study_patient_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "age": age,
        "sex": sex,
        "race": race,
        "ethnicity": ethnicity,
        "preferred_language": language,
        "interpreter_needed": interpreter,
        "pcp_id": pcp,
        "insurance_type": insurance,
        "mychart_active": mychart,
        "sdoh_screened": screened,
        # Screening answers are only visible for screened patients
        "sdoh_transportation": masked(transport, screened),
        "sdoh_financial_strain": masked(financial, screened),
        "sdoh_housing_instability": masked(housing, screened),
        "sdoh_social_isolation": masked(isolation, screened),
        "sdoh_low_health_literacy": masked(low_lit, screened),
        "comorbidity_count": comorb,
        "asymptomatic_chronic": asymptomatic,
        "new_patient_at_start": is_new,
        "panel_join_date": join.date,
    })
    patients["panel_size"] = patients.groupby("pcp_id")["study_patient_id"].transform("size")

    truth = {
        "age65": age65, "female": sex == "Female", "medicaid": medicaid,
        "mychart": mychart, "transport": transport, "isolation": isolation,
        "low_lit": low_lit, "comorb": comorb, "asymptomatic": asymptomatic,
        "large_panel": large_panel, "is_new": is_new, "join": join,
    }
    return patients, truth


# ============================================================
# 2. APPOINTMENTS (simulated in time order per patient)
# ============================================================

def expected_lead(P, visit_type, large_panel, channel, age65):
    q3 = P["q3_access"]
    m = q3["base_mean_lead_days_new"]
    if visit_type == "Follow-up":
        m *= q3["mult_follow_up"]
    elif visit_type == "Annual Physical":
        m *= q3["mult_annual_physical"]
    if large_panel:
        m *= q3["mult_large_panel"]
    if channel == "MyChart":
        m *= q3["mult_mychart_booking"]
    if age65:
        m *= q3["mult_age65"]
    return m


def simulate_appointments(seed, patients, T, P, start, end, b):
    """b = intercepts {'ns','pc','lc','q2'}."""
    rng = np.random.default_rng(seed)
    q1, q2, q3, q9 = P["q1_attendance"], P["q2_followup"], P["q3_access"], P["q9_care_gaps"]
    cap = q1["lead_time_cap_days"]

    # Fixed log-odds pieces
    L = {k: lg(v) for k, v in q1.items() if k.startswith("or_")}
    q2_lin = (
        lg(q2["or_asymptomatic_chronic"]) * T["asymptomatic"]
        + lg(q2["or_social_isolation"]) * T["isolation"]
        + lg(q2["or_low_health_literacy"]) * T["low_lit"]
        + lg(q2["or_medicaid"]) * T["medicaid"]
    )
    modes = list(q1["visit_mode_probs"].keys())
    mode_p = np.array(list(q1["visit_mode_probs"].values()))
    ids = patients["study_patient_id"].to_numpy()
    rows = []

    for i in range(len(patients)):
        mc, a65 = T["mychart"][i], T["age65"][i]
        lp = T["large_panel"][i]
        static_ns = (
            L["or_no_mychart"] * (not mc)
            + L["or_age65_x_no_mychart"] * (a65 and not mc)
            + L["or_medicaid"] * T["medicaid"][i]
            + L["or_transportation"] * T["transport"][i]
            + L["or_comorbidity_per_condition"] * T["comorb"][i]
            + L["or_female"] * T["female"][i]
        )
        p_ontime = sigmoid(b["q2"] + q2_lin[i])
        chronic = T["comorb"][i] > 0

        if T["is_new"][i]:
            booking = T["join"][i]
            vtype = "New Patient"
        else:
            booking = start + int(rng.integers(0, 365)) * DAY
            vtype = "Annual Physical" if rng.random() < 0.6 else "Follow-up"
        checkout, planned = False, None
        prior_ns = 0

        for _ in range(80):
            if booking > end:
                break
            if checkout:
                appt = planned
                channel = "Checkout"
                lead = (appt - booking).days
            else:
                u = rng.random()
                if mc and u < q3["mychart_booking_share_if_active"]:
                    channel = "MyChart"
                elif rng.random() < q3["front_desk_share"]:
                    channel = "Front desk"
                else:
                    channel = "PAC"
                mean = expected_lead(P, vtype, lp, channel, a65)
                lead = max(1, int(round(rng.gamma(q3["gamma_shape"], mean / q3["gamma_shape"]))))
                appt = booking + lead * DAY
            if appt > end:
                break
            mode = modes[rng.choice(len(modes), p=mode_p)]
            rec = {
                "study_patient_id": ids[i], "booking_date": booking, "appointment_date": appt,
                "visit_type": vtype, "visit_mode": mode, "booking_channel": channel,
                "booked_at_checkout": checkout, "lead_time_days": lead,
                "return_interval_days": np.nan, "followup_booked_same_day": False,
            }

            # System side: clinic cancels
            if rng.random() < q1["clinic_cancel_rate"]:
                rec["appointment_status"] = "Clinic canceled"
                rows.append(rec)
                back = int(rng.integers(1, max(2, min(lead, 14))))
                booking = max(booking, appt - back * DAY)
                checkout = False
                continue

            # Patient side: completed / no-show / patient cancel / late cancel
            l30 = min(lead, cap) / 30
            eta_ns = (b["ns"] + static_ns + L["or_lead_time_per_30d"] * l30
                      + L["or_prior_no_show"] * (prior_ns > 0)
                      + L["or_video"] * (mode == "Video") + L["or_phone"] * (mode == "Phone"))
            eta_pc = b["pc"] + L["or_patient_cancel_lead_per_30d"] * l30
            eta_lc = b["lc"] + L["or_late_cancel_transportation"] * T["transport"][i]
            e = np.exp([0.0, eta_ns, eta_pc, eta_lc])
            status = ["Completed", "No-show", "Patient canceled", "Late canceled"][
                rng.choice(4, p=e / e.sum())]
            rec["appointment_status"] = status

            if status == "No-show":
                prior_ns += 1
            if status in ("No-show", "Late canceled"):
                rows.append(rec)
                if rng.random() < q9["rebook_after_missed"]:
                    booking = appt + (1 + int(rng.exponential(q9["rebook_delay_mean_days"]))) * DAY
                else:
                    booking = appt + (q9["no_rebook_delay_min_days"]
                                      + int(rng.exponential(q9["no_rebook_delay_mean_days"]))) * DAY
                checkout = False
                continue
            if status == "Patient canceled":
                rows.append(rec)
                back = int(rng.integers(2, max(3, min(lead, 21))))
                cancel_day = max(booking, appt - back * DAY)
                if rng.random() < q9["rebook_after_cancel"]:
                    booking = cancel_day + int(rng.exponential(q9["rebook_after_cancel_delay_mean_days"])) * DAY
                else:
                    booking = appt + (q9["no_rebook_delay_min_days"]
                                      + int(rng.exponential(q9["no_rebook_delay_mean_days"]))) * DAY
                checkout = False
                continue

            # Completed visit: PCP sets the next step
            if vtype == "Annual Physical":
                R = 365
            elif chronic:
                R = 90 if rng.random() < 0.5 else 180
            else:
                R = 180 if rng.random() < 0.3 else 365
            documented = rng.random() < q2["p_documented"]
            p_sd = q2["p_same_day_if_documented"] if documented else q2["p_same_day_if_undocumented"]
            same_day = rng.random() < p_sd
            rec["return_interval_days"] = R if documented else np.nan
            rec["followup_booked_same_day"] = same_day
            rows.append(rec)
            next_type = "Annual Physical" if R == 365 else "Follow-up"

            if same_day:
                booking = appt
                planned = appt + (R + int(rng.integers(-7, 8))) * DAY
                checkout = True
            else:
                checkout = False
                if rng.random() < p_ontime:
                    lead_guess = expected_lead(P, next_type, lp, "PAC", a65)
                    booking = appt + max(1, int(R - lead_guess - 20 + rng.integers(-10, 11))) * DAY
                elif rng.random() < q9["dropout_prob"]:
                    booking = appt + (R + int(rng.integers(q9["dropout_gap_min_days"],
                                                           q9["dropout_gap_max_days"]))) * DAY
                else:
                    booking = appt + (R + q2["grace_days"] + 1
                                      + int(rng.exponential(q9["late_return_mean_days"]))) * DAY
            vtype = next_type

    appts = pd.DataFrame(rows)
    appts = appts.sort_values(["study_patient_id", "appointment_date", "booking_date"]).reset_index(drop=True)
    appts.insert(0, "encounter_id", [f"E{i:06d}" for i in range(1, len(appts) + 1)])
    pcp_map = patients.set_index("study_patient_id")["pcp_id"]
    appts["pcp_id"] = appts["study_patient_id"].map(pcp_map)
    return appts


def calibrate_attendance(seed, patients, T, P, start, end):
    """Two passes so the overall no-show / cancel rates hit their targets."""
    q1, q2 = P["q1_attendance"], P["q2_followup"]
    q2_lin = (lg(q2["or_asymptomatic_chronic"]) * T["asymptomatic"]
              + lg(q2["or_social_isolation"]) * T["isolation"]
              + lg(q2["or_low_health_literacy"]) * T["low_lit"]
              + lg(q2["or_medicaid"]) * T["medicaid"])
    b = {
        "ns": logit(q1["target_no_show_rate"]) - 0.8,
        "pc": logit(q1["target_patient_cancel_rate"]),
        "lc": logit(q1["target_late_cancel_rate"]),
        "q2": solve_intercept(q2_lin, q2["target_on_time_return"]),
    }
    for _ in range(3):
        appts = simulate_appointments(seed, patients, T, P, start, end, b)
        s = appts["appointment_status"]
        n_pat = s.isin(["Completed", "No-show", "Patient canceled", "Late canceled"]).sum()
        ns = (s == "No-show").sum() / s.isin(["Completed", "No-show"]).sum()
        pc = (s == "Patient canceled").sum() / n_pat
        lc = (s == "Late canceled").sum() / n_pat
        b["ns"] += logit(q1["target_no_show_rate"]) - logit(ns)
        b["pc"] += logit(q1["target_patient_cancel_rate"]) - logit(pc)
        b["lc"] += logit(q1["target_late_cancel_rate"]) - logit(lc)
    appts = simulate_appointments(seed, patients, T, P, start, end, b)
    return appts, b


# ============================================================
# 3. REFERRALS (Q4)
# ============================================================

def make_referrals(rng, appts, patients, T, P, end):
    q4 = P["q4_referrals"]
    pidx = {pid: k for k, pid in enumerate(patients["study_patient_id"])}
    done = appts[appts["appointment_status"] == "Completed"].copy()
    k = done["study_patient_id"].map(pidx).to_numpy()
    p_ref = sigmoid(logit(q4["base_referral_rate"]) + lg(q4["or_referral_per_condition"]) * T["comorb"][k])
    done = done[rng.random(len(done)) < p_ref]
    k = done["study_patient_id"].map(pidx).to_numpy()
    n = len(done)

    specialty = pick(rng, q4["specialty_probs"], n)
    auth_p = np.array([q4["prior_auth_share"][s] for s in specialty])
    prior_auth = rng.random(n) < auth_p
    urgent = rng.random(n) < q4["urgent_share"]
    leaves = rng.random(n) < q4["leaves_nyu_share"]
    hr_spec = np.array([q4["hr_specialty"][s] for s in specialty])

    lam_c = (np.log(2) / q4["base_median_days_to_completion"]) * hr_spec \
        * np.where(prior_auth, q4["hr_prior_auth"], 1.0) \
        * np.where(urgent, q4["hr_urgent"], 1.0) \
        * np.where(T["medicaid"][k], q4["hr_medicaid"], 1.0)
    lam_x = np.log(2) / q4["median_days_to_closure"]
    t_c = rng.exponential(1 / lam_c)
    t_x = rng.exponential(1 / lam_x, n)
    t_x = np.where(leaves, rng.uniform(1, 10, n), t_x)
    t_c = np.where(leaves, np.inf, t_c)

    ref_date = pd.to_datetime(done["appointment_date"].to_numpy())
    follow = (end - ref_date).days.to_numpy()
    t_event = np.minimum(t_c, t_x)
    status = np.where(t_event > follow, "Pending", np.where(t_c <= t_x, "Completed", "Closed"))
    days = np.ceil(np.minimum(t_event, follow)).astype(int)

    reasons = pick(rng, q4["close_reason_probs"], n)
    reasons = np.where(leaves, "Care elsewhere", reasons)

    refs = pd.DataFrame({
        "referral_id": [f"R{i:06d}" for i in range(1, n + 1)],
        "study_patient_id": done["study_patient_id"].to_numpy(),
        "source_encounter_id": done["encounter_id"].to_numpy(),
        "referral_date": ref_date.date,
        "specialty": specialty,
        "priority": np.where(urgent, "Urgent", "Routine"),
        "prior_auth_required": prior_auth,
        "referred_outside_nyu": leaves,
        "referral_status": status,
        "status_date": (ref_date + pd.to_timedelta(days, unit="D")).date,
        "days_to_status": days,
        "close_reason": np.where(status == "Closed", reasons, None),
    })
    refs["completion_date"] = np.where(status == "Completed", refs["status_date"], None)
    return refs


# ============================================================
# 4. ED / URGENT CARE / INPATIENT (Q7, Q9, D2, D14)
# ============================================================

def make_hospital_visits(rng, appts, refs, patients, T, P, end):
    """
    ED / urgent care / inpatient visits from a continuous-time rate:

        rate(t) = background * or_comorbidity^count * or_no_mychart^(no MyChart)
                  * or_missed_visit        if t is within 90 days after a missed visit
                  * or_incomplete_referral if t is 60-150 days after a referral not done by day 60
                  * mult_ed_in_gap         if t falls in a long gap (ed_gap_threshold_days) between PCP appointments

    Simulated by thinning: draw candidates at the maximum rate, keep each
    with probability rate(t) / max rate.
    """
    q7, q9 = P["q7_utilization"], P["q9_care_gaps"]
    ids = patients["study_patient_id"].to_numpy()
    base = (q7["background_ed_rate_per_year"] / 365.25
            * q7["or_comorbidity_per_condition"] ** T["comorb"]
            * np.where(T["mychart"], 1.0, q7["or_no_mychart"]))
    m_miss, m_ref, m_gap = q7["or_missed_visit"], q7["or_incomplete_referral"], q9["mult_ed_in_gap"]
    max_mult = max(m_miss, 1) * max(m_ref, 1) * max(m_gap, 1)

    span = (end - T["join"]).days.to_numpy()
    n_cand = rng.poisson(base * max_mult * span)
    k = np.repeat(np.arange(len(patients)), n_cand)
    t = pd.to_datetime(T["join"][k]) + pd.to_timedelta((rng.random(len(k)) * span[k]).astype(int) + 1, unit="D")
    cand = pd.DataFrame({"study_patient_id": ids[k], "admission_date": t, "k": k}).sort_values("admission_date")

    def last_before(df, events, col):
        ev = events.rename(columns={col: "_ev"}).sort_values("_ev")
        return pd.merge_asof(df, ev, left_on="admission_date", right_on="_ev",
                             by="study_patient_id", direction="backward", allow_exact_matches=False)["_ev"].to_numpy()

    a = appts.copy()
    a["appointment_date"] = pd.to_datetime(a["appointment_date"])
    missed = a[a["appointment_status"].isin(["No-show", "Late canceled"])][["study_patient_id", "appointment_date"]]
    last_miss = last_before(cand, missed, "appointment_date")
    in_miss = (cand["admission_date"].to_numpy() - last_miss) <= np.timedelta64(90, "D")

    r = refs.copy()
    r["referral_date"] = pd.to_datetime(r["referral_date"])
    bad = r[~((r["referral_status"] == "Completed") & (r["days_to_status"] <= 60))]
    bad = pd.DataFrame({"study_patient_id": bad["study_patient_id"], "start": bad["referral_date"] + pd.Timedelta(days=60)})
    last_ref = last_before(cand, bad, "start")
    in_ref = (cand["admission_date"].to_numpy() - last_ref) <= np.timedelta64(90, "D")

    a = a.sort_values(["study_patient_id", "appointment_date"])
    a["next_date"] = a.groupby("study_patient_id")["appointment_date"].shift(-1)
    prev = pd.merge_asof(cand, a[["study_patient_id", "appointment_date", "next_date"]].sort_values("appointment_date"),
                         left_on="admission_date", right_on="appointment_date", by="study_patient_id", direction="backward")
    in_gap = ((prev["next_date"] - prev["appointment_date"]).dt.days > q9["ed_gap_threshold_days"]).to_numpy()

    mult = np.where(in_miss, m_miss, 1.0) * np.where(in_ref, m_ref, 1.0) * np.where(in_gap, m_gap, 1.0)
    keep = rng.random(len(cand)) < mult / max_mult
    hv = cand[keep][["study_patient_id", "admission_date"]]
    hv = hv[hv["admission_date"] <= end].sort_values(["study_patient_id", "admission_date"]).reset_index(drop=True)
    hv["visit_category"] = pick(rng, q7["category_probs"], len(hv))
    stay = np.where(hv["visit_category"] == "Inpatient", rng.integers(1, 8, len(hv)), 0)
    hv["discharge_date"] = (hv["admission_date"] + pd.to_timedelta(stay, unit="D")).dt.date
    hv["admission_date"] = hv["admission_date"].dt.date
    hv.insert(0, "hospital_encounter_id", [f"H{i:06d}" for i in range(1, len(hv) + 1)])
    return hv


# ============================================================
# 5. ORDERS, PREVENTIVE CARE, MESSAGES (Q2.2, Q2.3, Q5)
# ============================================================

def make_orders(rng, appts, patients, T, P, end):
    q2 = P["q2_followup"]
    pidx = {pid: k for k, pid in enumerate(patients["study_patient_id"])}
    done = appts[appts["appointment_status"] == "Completed"]
    done = done[rng.random(len(done)) < q2["order_rate_per_visit"]]
    n = len(done)
    k = done["study_patient_id"].map(pidx).to_numpy()
    otype = rng.choice(["Mammogram", "Bone density", "Lab panel", "CT/MRI"], n, p=[0.3, 0.15, 0.35, 0.2])
    auth = rng.random(n) < np.where(otype == "CT/MRI", 0.7, 0.05)
    lin = lg(q2["or_order_prior_auth"]) * auth + lg(q2["or_order_low_health_literacy"]) * T["low_lit"][k]
    b0 = solve_intercept(lin - lg(q2["or_order_prior_auth"]) * auth, q2["target_order_completion"])
    ok = rng.random(n) < sigmoid(b0 + lin)
    odate = pd.to_datetime(done["appointment_date"].to_numpy())
    cdate = odate + pd.to_timedelta(rng.integers(1, 61, n), unit="D")
    late = (~ok) & (rng.random(n) < 0.3)
    cdate = np.where(late, odate + pd.to_timedelta(rng.integers(61, 240, n), unit="D"), cdate)
    completed = ok | late
    cdate = pd.to_datetime(np.where(completed, cdate, pd.NaT))
    cdate = pd.Series(cdate).where(pd.Series(cdate) <= end)
    return pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(1, n + 1)],
        "study_patient_id": done["study_patient_id"].to_numpy(),
        "source_encounter_id": done["encounter_id"].to_numpy(),
        "order_type": otype,
        "prior_auth_required": auth,
        "order_date": odate.date,
        "completed_date": cdate.dt.date.to_numpy(),
    })


def make_preventive(rng, patients, T, P, start, end):
    q2 = P["q2_followup"]
    items = []
    ages = patients["age"].to_numpy()
    rules = [  # name, min age, max age, interval days
        ("Mammogram", 40, 74, 730), ("Bone density", 65, 120, 730),
        ("Colorectal screening", 45, 75, 3650), ("Flu vaccine", 18, 120, 365),
    ]
    for name, lo, hi, every in rules:
        elig = np.where((ages >= lo) & (ages <= hi))[0]
        for k in elig:
            due = start + int(rng.integers(0, min(every, 365))) * DAY
            while due <= end - 90 * DAY:
                items.append((k, name, due))
                due = due + every * DAY
    k = np.array([x[0] for x in items])
    n = len(items)
    lin = (lg(q2["or_preventive_social_isolation"]) * T["isolation"][k]
           + lg(q2["or_preventive_low_health_literacy"]) * T["low_lit"][k]
           + lg(q2["or_preventive_no_mychart"]) * (~T["mychart"][k]))
    b0 = solve_intercept(lin, q2["target_preventive_on_time"])
    on_time = rng.random(n) < sigmoid(b0 + lin)
    due = pd.to_datetime([x[2] for x in items])
    late = (~on_time) & (rng.random(n) < 0.35)
    postponed = (~on_time) & (~late) & (rng.random(n) < 0.3)
    comp = np.where(on_time, due + pd.to_timedelta(rng.integers(0, 91, n), unit="D"),
                    np.where(late, due + pd.to_timedelta(rng.integers(91, 365, n), unit="D"), pd.NaT))
    comp = pd.Series(pd.to_datetime(comp)).where(lambda s: s <= end)
    status = np.where(on_time, "Completed on time",
                      np.where(late, "Completed late", np.where(postponed, "Postponed", "Overdue")))
    reason = np.where(postponed, rng.choice(["Patient declined", "Deferred by provider"], n), None)
    return pd.DataFrame({
        "preventive_item_id": [f"HM{i:06d}" for i in range(1, n + 1)],
        "study_patient_id": patients["study_patient_id"].to_numpy()[k],
        "item": [x[1] for x in items],
        "due_date": due.date,
        "completed_date": comp.dt.date.to_numpy(),
        "status": status,
        "postpone_reason": reason,
    })


def make_messages(rng, patients, T, P, start, end):
    q5 = P["q5_messages"]
    n_pat = len(patients)
    volume = rng.gamma(q5["gamma_shape_patient_volume"],
                       q5["mean_messages_per_patient"] / q5["gamma_shape_patient_volume"], n_pat)
    counts = rng.poisson(volume)
    k = np.repeat(np.arange(n_pat), counts)
    n = len(k)
    span_h = (end - start).days * 24
    sent = start + pd.to_timedelta(rng.integers(0, span_h, n), unit="h")
    high = rng.random(n) < q5["high_priority_share"]
    pac = rng.random(n) < q5["pac_route_share"]
    weekend = sent.dayofweek >= 5
    hours = rng.lognormal(np.log(q5["median_response_hours"]), q5["response_sigma"], n)
    hours = np.where(high, hours * q5["high_priority_mult"], hours)
    hours = hours + weekend * q5["weekend_delay_hours"] + pac * q5["pac_delay_hours"]
    first = sent + pd.to_timedelta(hours, unit="h")
    out = pd.DataFrame({
        "message_id": [f"M{i:07d}" for i in range(1, n + 1)],
        "study_patient_id": patients["study_patient_id"].to_numpy()[k],
        "sent_datetime": sent,
        "priority": np.where(high, "High", "Routine"),
        "route": np.where(pac, "PAC", "Direct"),
        "first_response_datetime": first,
    })
    return out.sort_values(["study_patient_id", "sent_datetime"]).reset_index(drop=True)


# ============================================================
# 6. MAIN
# ============================================================

def build_scenario(name, P, settings, out_root):
    seed = settings["seed"]
    start = pd.Timestamp(settings["window_start"])
    end = pd.Timestamp(settings["window_end"])
    rng = np.random.default_rng(seed)

    patients, T = make_patients(rng, P, settings["n_patients"], start, end)
    appts, b = calibrate_attendance(seed + 1, patients, T, P, start, end)
    rng2 = np.random.default_rng(seed + 2)
    refs = make_referrals(rng2, appts, patients, T, P, end)
    hv = make_hospital_visits(rng2, appts, refs, patients, T, P, end)
    orders = make_orders(rng2, appts, patients, T, P, end)
    prev = make_preventive(rng2, patients, T, P, start, end)
    msgs = make_messages(rng2, patients, T, P, start, end)

    for col in ["booking_date", "appointment_date"]:
        appts[col] = pd.to_datetime(appts[col]).dt.date

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    tables = {"patients": patients, "appointments": appts, "referrals": refs,
              "hospital_visits": hv, "orders": orders, "preventive_care": prev,
              "messages": msgs}
    for tname, df in tables.items():
        df.to_csv(out / f"{tname}.csv", index=False)

    params = {"scenario": name, "settings": settings, "params": P,
              "calibrated_intercepts": b}
    with open(out / "scenario_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=str)

    s = appts["appointment_status"]
    print(f"\n[{name}] -> {out.relative_to(PROJECT_ROOT)}")
    for tname, df in tables.items():
        print(f"  {tname:<16}{len(df):>8,} rows")
    print(f"  no-show rate (vs completed): {(s == 'No-show').sum() / s.isin(['Completed', 'No-show']).sum():.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="Build only this scenario (default: all)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    settings, scenarios = load_config(args.config)
    out_root = PROJECT_ROOT / settings["output_root"]
    names = [args.scenario] if args.scenario else list(scenarios)
    for name in names:
        if name not in scenarios:
            raise SystemExit(f"Unknown scenario '{name}'. Options: {', '.join(scenarios)}")
        build_scenario(name, scenarios[name], settings, out_root)


if __name__ == "__main__":
    main()
