"""
NYU Langone Capstone - regression models for Q1-Q9

Every function takes a data folder (one of data/scenario/<name>/,
or later the cleaned Epic extract with the same columns) and
returns a tidy table of estimates.

    from src.scenario_models import load_data, q1_no_show
    d = load_data("data/scenario/calibrated")
    table, result = q1_no_show(d)

Model list
    Q1   no-show logistic (with MyChart x 65+ interaction)
    Q2   follow-up on time / preventive care / tests completed
    Q3   wait time (Gamma GLM, log link)
    Q4   referral time-to-completion (Cox PH)
    Q5   message response time (descriptive)
    Q6   pathway drop-off metrics D2-D14 (descriptive)
    Q7   ED / urgent care / inpatient within 90 days (logistic)
    Q8   patient-level care-gap risk (logistic + random forest)
    Q9   12-month gaps in PCP care (descriptive)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

LEAD_TIME_CAP_DAYS = 180
FOLLOWUP_GRACE_DAYS = 30


# ============================================================
# 0. LOAD
# ============================================================

def _bool(s):
    """Turn True/False/1/0/blank into 1.0 / 0.0 / NaN."""
    return (s.astype("string").str.strip().str.lower()
             .map({"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}))


def load_data(data_dir):
    data_dir = Path(data_dir)
    d = {}
    for name in ["patients", "appointments", "referrals", "hospital_visits",
                 "orders", "preventive_care", "messages"]:
        path = data_dir / f"{name}.csv"
        if path.exists():
            d[name] = pd.read_csv(path)

    p = d["patients"]
    for col in ["mychart_active", "interpreter_needed", "asymptomatic_chronic",
                "sdoh_transportation", "sdoh_financial_strain", "sdoh_housing_instability",
                "sdoh_social_isolation", "sdoh_low_health_literacy"]:
        if col in p:
            p[col] = _bool(p[col])
    p["age65"] = (p["age"] >= 65).astype(int)
    p["no_mychart"] = 1 - p["mychart_active"]
    p["female"] = (p["sex"] == "Female").astype(int)
    p["medicaid"] = (p["insurance_type"] == "Medicaid").astype(int)

    a = d["appointments"]
    for col in ["booking_date", "appointment_date"]:
        a[col] = pd.to_datetime(a[col])
    a = a.sort_values(["study_patient_id", "appointment_date", "booking_date"]).reset_index(drop=True)
    is_ns = (a["appointment_status"] == "No-show").astype(int)
    a["prior_no_show_count"] = is_ns.groupby(a["study_patient_id"]).cumsum() - is_ns
    a["prior_no_show"] = (a["prior_no_show_count"] > 0).astype(int)
    for col in ["booked_at_checkout", "followup_booked_same_day"]:
        if col in a:
            a[col] = _bool(a[col]).fillna(0).astype(int)
    d["appointments"] = a

    if "referrals" in d:
        r = d["referrals"]
        r["referral_date"] = pd.to_datetime(r["referral_date"])
        for col in ["prior_auth_required", "referred_outside_nyu"]:
            r[col] = _bool(r[col]).astype(int)
    if "hospital_visits" in d:
        d["hospital_visits"]["admission_date"] = pd.to_datetime(d["hospital_visits"]["admission_date"])
    if "orders" in d:
        o = d["orders"]
        o["order_date"] = pd.to_datetime(o["order_date"])
        o["completed_date"] = pd.to_datetime(o["completed_date"])
        o["prior_auth_required"] = _bool(o["prior_auth_required"]).astype(int)
    if "preventive_care" in d:
        d["preventive_care"]["due_date"] = pd.to_datetime(d["preventive_care"]["due_date"])
    if "messages" in d:
        m = d["messages"]
        m["sent_datetime"] = pd.to_datetime(m["sent_datetime"])
        m["first_response_datetime"] = pd.to_datetime(m["first_response_datetime"])

    d["window_end"] = a["appointment_date"].max()
    d["window_start"] = a["booking_date"].min()
    return d


def _with_patient(df, d, cols):
    return df.merge(d["patients"][["study_patient_id"] + cols], on="study_patient_id",
                    how="left", validate="many_to_one")


def _tidy(result, question, model, exponentiate=True, kind="OR"):
    conf = result.conf_int()
    f = np.exp if exponentiate else (lambda x: x)
    out = pd.DataFrame({
        "question": question, "model": model, "term": result.params.index,
        "kind": kind, "estimate": f(result.params.values),
        "ci_low": f(conf.iloc[:, 0].values), "ci_high": f(conf.iloc[:, 1].values),
        "p_value": result.pvalues.values,
    })
    out["n_obs"] = int(result.nobs)
    return out[out["term"] != "Intercept"].reset_index(drop=True)


def _logit(formula, data, cluster=None):
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    if cluster is not None:
        return model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0]})
    return model.fit()


# ============================================================
# Q1. NO-SHOW
# ============================================================

def q1_no_show(d):
    a = _with_patient(d["appointments"], d,
                      ["age65", "no_mychart", "female", "insurance_type",
                       "sdoh_transportation", "comorbidity_count"])
    df = a[a["appointment_status"].isin(["Completed", "No-show"])].copy()
    df["is_no_show"] = (df["appointment_status"] == "No-show").astype(int)
    df["lead_time_30"] = df["lead_time_days"].clip(0, LEAD_TIME_CAP_DAYS) / 30
    df = df.dropna(subset=["sdoh_transportation", "no_mychart"])   # screened patients only
    formula = ("is_no_show ~ lead_time_30 + no_mychart + age65 + no_mychart:age65"
               " + prior_no_show + C(insurance_type, Treatment('Commercial'))"
               " + sdoh_transportation + comorbidity_count + female"
               " + C(visit_mode, Treatment('Office'))"
               " + C(visit_type, Treatment('Annual Physical'))")
    res = _logit(formula, df, cluster="study_patient_id")
    return _tidy(res, "Q1", "No-show logistic"), res


# ============================================================
# Q2. FOLLOW-THROUGH
# ============================================================

def _next_completed_after(src, appts):
    """For each source visit, date of the next completed visit of the same patient."""
    done = appts[appts["appointment_status"] == "Completed"][["study_patient_id", "appointment_date"]]
    done = done.rename(columns={"appointment_date": "next_completed_date"}).sort_values("next_completed_date")
    left = src.assign(_key=src["appointment_date"] + pd.Timedelta(days=1)).sort_values("_key")
    out = pd.merge_asof(left, done, left_on="_key", right_on="next_completed_date",
                        by="study_patient_id", direction="forward")
    return out.drop(columns="_key")


def q2_followup(d):
    a = d["appointments"]
    end = d["window_end"]
    src = a[(a["appointment_status"] == "Completed") & a["return_interval_days"].notna()].copy()
    src = src[src["appointment_date"] + pd.to_timedelta(src["return_interval_days"] + FOLLOWUP_GRACE_DAYS, unit="D") <= end]
    src = _next_completed_after(src, a)
    limit = src["appointment_date"] + pd.to_timedelta(src["return_interval_days"] + FOLLOWUP_GRACE_DAYS, unit="D")
    src["on_time"] = (src["next_completed_date"].notna() & (src["next_completed_date"] <= limit)).astype(int)
    src = _with_patient(src, d, ["asymptomatic_chronic", "sdoh_social_isolation",
                                 "sdoh_low_health_literacy", "medicaid", "comorbidity_count"])

    # Q2.4: same-day booking vs later (no fixed planted OR; built into the mechanism)
    res_sd = _logit("on_time ~ followup_booked_same_day + comorbidity_count", src, cluster="study_patient_id")

    # Patient factors among follow-ups NOT booked at checkout
    sub = src[src["followup_booked_same_day"] == 0].dropna(subset=["sdoh_social_isolation"])
    res_pf = _logit("on_time ~ asymptomatic_chronic + sdoh_social_isolation + sdoh_low_health_literacy"
                    " + medicaid + comorbidity_count", sub, cluster="study_patient_id")
    out = pd.concat([_tidy(res_sd, "Q2", "Follow-up on time: same-day booking"),
                     _tidy(res_pf, "Q2", "Follow-up on time: patient factors")], ignore_index=True)
    return out, {"same_day": res_sd, "patient_factors": res_pf}


def q2_preventive(d):
    pc = d["preventive_care"]
    pc = pc.assign(on_time=(pc["status"] == "Completed on time").astype(int))
    pc = _with_patient(pc, d, ["sdoh_social_isolation", "sdoh_low_health_literacy", "no_mychart"])
    pc = pc.dropna(subset=["sdoh_social_isolation"])
    res = _logit("on_time ~ sdoh_social_isolation + sdoh_low_health_literacy + no_mychart",
                 pc, cluster="study_patient_id")
    return _tidy(res, "Q2", "Preventive care on time"), res


def q2_orders(d):
    o = d["orders"]
    o = o[o["order_date"] <= d["window_end"] - pd.Timedelta(days=60)].copy()
    o["done_60d"] = ((o["completed_date"] - o["order_date"]).dt.days <= 60).astype(int)
    o = _with_patient(o, d, ["sdoh_low_health_literacy"]).dropna(subset=["sdoh_low_health_literacy"])
    res = _logit("done_60d ~ prior_auth_required + sdoh_low_health_literacy", o, cluster="study_patient_id")
    return _tidy(res, "Q2", "Tests completed in 60 days"), res


# ============================================================
# Q3. WAIT TIME
# ============================================================

def q3_wait_time(d):
    a = _with_patient(d["appointments"], d, ["age65", "pcp_id"] if "pcp_id" not in d["appointments"] else ["age65"])
    df = a[(a["booked_at_checkout"] == 0)
           & (a["booking_date"] <= d["window_end"] - pd.Timedelta(days=LEAD_TIME_CAP_DAYS))].copy()
    panel = d["patients"]["pcp_id"].value_counts()
    df["large_panel"] = df["pcp_id"].map(panel >= panel.median() * 1.3).astype(int)
    df["mychart_booking"] = (df["booking_channel"] == "MyChart").astype(int)
    res = smf.glm("lead_time_days ~ C(visit_type, Treatment('New Patient')) + large_panel"
                  " + mychart_booking + age65", data=df,
                  family=sm.families.Gamma(link=sm.families.links.Log())).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["study_patient_id"])[0]})
    return _tidy(res, "Q3", "Wait time Gamma GLM", kind="ratio of means"), res


# ============================================================
# Q4. REFERRALS
# ============================================================

def q4_referral_cox(d):
    r = _with_patient(d["referrals"], d, ["medicaid"])
    r["completed"] = (r["referral_status"] == "Completed").astype(int)
    r["urgent"] = (r["priority"] == "Urgent").astype(int)
    r["days"] = r["days_to_status"].clip(lower=0.5)
    model = smf.phreg("days ~ prior_auth_required + urgent + C(specialty, Treatment('Cardiology')) + medicaid",
                      data=r, status=r["completed"].to_numpy(), ties="efron")
    res = model.fit()
    conf = res.conf_int()
    names = model.exog_names
    out = pd.DataFrame({
        "question": "Q4", "model": "Referral completion Cox PH", "term": names, "kind": "HR",
        "estimate": np.exp(res.params), "ci_low": np.exp(conf[:, 0]), "ci_high": np.exp(conf[:, 1]),
        "p_value": res.pvalues, "n_obs": len(r),
    })
    return out, res


def q4_km_table(d, by="prior_auth_required"):
    """Kaplan-Meier: share of referrals completed by day 30/60/90, per group."""
    r = d["referrals"]
    rows = []
    for g, sub in r.groupby(by):
        sf = sm.SurvfuncRight(sub["days_to_status"].clip(lower=0.5), (sub["referral_status"] == "Completed").astype(int))
        for t in [30, 60, 90]:
            idx = np.searchsorted(sf.surv_times, t, side="right") - 1
            surv = 1.0 if idx < 0 else sf.surv_prob[idx]
            rows.append({by: g, "day": t, "share_completed": 1 - surv})
    return pd.DataFrame(rows)


# ============================================================
# Q5. MESSAGES
# ============================================================

def q5_messages(d):
    m = d["messages"].copy()
    m["hours"] = (m["first_response_datetime"] - m["sent_datetime"]).dt.total_seconds() / 3600
    m["weekend"] = m["sent_datetime"].dt.dayofweek >= 5
    routine = m[m["priority"] == "Routine"]
    base = routine[(routine["route"] == "Direct") & ~routine["weekend"]]["hours"].median()
    wk = routine[(routine["route"] == "Direct") & routine["weekend"]]["hours"].median()
    pac = routine[(routine["route"] == "PAC") & ~routine["weekend"]]["hours"].median()
    per_pt = m["study_patient_id"].value_counts()
    top20 = per_pt.head(int(len(d["patients"]) * 0.2)).sum() / len(m)
    return pd.DataFrame([
        {"question": "Q5", "model": "Message response (median hours)", "term": "weekend_extra_hours",
         "kind": "hours", "estimate": wk - base},
        {"question": "Q5", "model": "Message response (median hours)", "term": "pac_extra_hours",
         "kind": "hours", "estimate": pac - base},
        {"question": "Q5", "model": "Message volume", "term": "share_from_top20pct_patients",
         "kind": "share", "estimate": top20},
    ])


# ============================================================
# Q7. DOWNSTREAM UTILIZATION
# ============================================================

def _any_admission_between(index, hv, start_col, lo_days, hi_days):
    """1 if the patient has an ED/UC/inpatient visit in (start+lo, start+hi]."""
    h = hv[["study_patient_id", "admission_date"]].sort_values("admission_date")
    left = index.assign(_key=index[start_col] + pd.Timedelta(days=lo_days + 1)).sort_values("_key")
    out = pd.merge_asof(left, h, left_on="_key", right_on="admission_date",
                        by="study_patient_id", direction="forward")
    hit = out["admission_date"].notna() & (out["admission_date"] <= out[start_col] + pd.Timedelta(days=hi_days))
    return out.drop(columns=["_key", "admission_date"]).assign(outcome=hit.astype(int))


def q7_utilization(d, days=90):
    a = d["appointments"]
    idx = a[a["appointment_status"].isin(["Completed", "No-show", "Late canceled"])]
    idx = idx[idx["appointment_date"] <= d["window_end"] - pd.Timedelta(days=days)].copy()
    idx["missed_visit"] = idx["appointment_status"].isin(["No-show", "Late canceled"]).astype(int)
    # Carry-over: a visit soon after an earlier missed visit is still inside that visit's risk window
    miss = a[a["appointment_status"].isin(["No-show", "Late canceled"])][["study_patient_id", "appointment_date"]]
    miss = miss.rename(columns={"appointment_date": "_prev_miss"}).sort_values("_prev_miss")
    idx = pd.merge_asof(idx.sort_values("appointment_date"), miss, left_on="appointment_date", right_on="_prev_miss",
                        by="study_patient_id", direction="backward", allow_exact_matches=False)
    idx["recent_missed_90d"] = ((idx["appointment_date"] - idx["_prev_miss"]).dt.days <= days).astype(int)
    idx = idx.drop(columns="_prev_miss")
    idx = _any_admission_between(idx, d["hospital_visits"], "appointment_date", 0, days)
    idx = _with_patient(idx, d, ["comorbidity_count", "no_mychart"])
    res_u = _logit("outcome ~ missed_visit", idx, cluster="study_patient_id")
    res_a = _logit("outcome ~ missed_visit + recent_missed_90d + comorbidity_count + no_mychart",
                   idx, cluster="study_patient_id")

    r = d["referrals"]
    r = r[r["referral_date"] <= d["window_end"] - pd.Timedelta(days=150)].copy()
    r["incomplete_referral"] = (~((r["referral_status"] == "Completed") & (r["days_to_status"] <= 60))).astype(int)
    r = _any_admission_between(r, d["hospital_visits"], "referral_date", 60, 150)
    r = _with_patient(r, d, ["comorbidity_count"])
    res_r = _logit("outcome ~ incomplete_referral + comorbidity_count", r, cluster="study_patient_id")

    out = pd.concat([
        _tidy(res_u, "Q7", f"ED/UC/inpatient {days}d: unadjusted"),
        _tidy(res_a, "Q7", f"ED/UC/inpatient {days}d: adjusted"),
        _tidy(res_r, "Q7", "ED/UC/inpatient after incomplete referral"),
    ], ignore_index=True)
    return out, {"unadjusted": res_u, "adjusted": res_a, "referral": res_r}


# ============================================================
# Q6 / Q9. PATHWAY METRICS (descriptive)
# ============================================================

def q6_q9_pathway_metrics(d):
    a, hv, r = d["appointments"], d["hospital_visits"], d["referrals"]
    end = d["window_end"]
    done = a[a["appointment_status"] == "Completed"]
    rows = []

    def add(dp, metric, value):
        rows.append({"question": "Q6", "drop_off": dp, "term": metric, "estimate": value})

    # D2: ED/UC visits by patients with no completed PCP visit in prior 12 months
    hv_s = hv.sort_values("admission_date")
    prev = pd.merge_asof(hv_s, done[["study_patient_id", "appointment_date"]].sort_values("appointment_date"),
                         left_on="admission_date", right_on="appointment_date",
                         by="study_patient_id", direction="backward")
    no_pcp = prev["appointment_date"].isna() | ((prev["admission_date"] - prev["appointment_date"]).dt.days > 365)
    add("D2", "ED/UC visits with no PCP visit in prior 12 months", no_pcp.mean())
    # D7
    newp = a[(a["visit_type"] == "New Patient") & (a["booked_at_checkout"] == 0)]
    add("D7", "New patients seen within 10 days", (newp["lead_time_days"] <= 10).mean())
    add("D7", "Median days booking to visit (not booked at checkout)", a[a["booked_at_checkout"] == 0]["lead_time_days"].median())
    # D9
    s = a["appointment_status"]
    add("D9", "No-show rate (vs completed)", (s == "No-show").sum() / s.isin(["Completed", "No-show"]).sum())
    add("D9", "Late cancel rate", (s == "Late canceled").mean())
    nxt = a.groupby("study_patient_id")["booking_date"].shift(-1)
    missed = a[s.isin(["No-show", "Late canceled"])]
    gap = (nxt[missed.index] - missed["appointment_date"]).dt.days
    add("D9", "Rebooked within 60 days after a missed visit", (gap <= 60).mean())
    # D10 / D11
    add("D10", "Completed visits with return interval documented", done["return_interval_days"].notna().mean())
    doc = done[done["return_interval_days"].notna()]
    add("D11", "Documented follow-ups booked on the visit day", doc["followup_booked_same_day"].mean())
    # D12 / D13
    add("D12", "Median days referral to completed visit", r[r["referral_status"] == "Completed"]["days_to_status"].median())
    add("D12", "Referrals needing prior auth", r["prior_auth_required"].mean())
    add("D13", "Referrals with no completed visit", (r["referral_status"] != "Completed").mean())
    # D14: completed PCP visit within 30 days after ED/UC
    nxt_pcp = pd.merge_asof(hv_s.assign(_k=hv_s["admission_date"] + pd.Timedelta(days=1)).sort_values("_k"),
                            done[["study_patient_id", "appointment_date"]].sort_values("appointment_date"),
                            left_on="_k", right_on="appointment_date", by="study_patient_id", direction="forward")
    within = (nxt_pcp["appointment_date"] - nxt_pcp["admission_date"]).dt.days
    for t in [7, 14, 30]:
        add("D14", f"PCP visit within {t} days after ED/UC/inpatient", (within <= t).mean())

    # Q9: any 12-month gap with no PCP appointment
    a_sorted = a.sort_values(["study_patient_id", "appointment_date"])
    dates = a_sorted.groupby("study_patient_id")["appointment_date"]
    gaps = dates.diff().dt.days
    tail = (end - dates.transform("max")).dt.days
    has_gap = (gaps > 365).groupby(a_sorted["study_patient_id"]).any() | (tail.groupby(a_sorted["study_patient_id"]).max() > 365)
    rows.append({"question": "Q9", "drop_off": "", "term": "Patients with any 12-month gap in PCP appointments",
                 "estimate": has_gap.reindex(d["patients"]["study_patient_id"]).fillna(True).mean()})
    return pd.DataFrame(rows)


# ============================================================
# Q8. HIGHEST-RISK PATIENTS
# ============================================================

def q8_risk(d, seed=0, top_share=0.25):
    """
    Patient-level risk of missing care.

    Missed-care rate = missed / due, counting:
      appointments (no-show or late cancel), documented follow-ups
      not completed on time, preventive items not done on time,
      and referrals not completed within 90 days.
    High risk = rate in the top 25% (patients with 5+ items due).
    """
    p = d["patients"].copy()
    a, end = d["appointments"], d["window_end"]
    parts = []

    ap = a[a["appointment_status"].isin(["Completed", "No-show", "Late canceled"])]
    parts.append(pd.DataFrame({"study_patient_id": ap["study_patient_id"],
                               "missed": ap["appointment_status"].ne("Completed").astype(int)}))

    src = a[(a["appointment_status"] == "Completed") & a["return_interval_days"].notna()].copy()
    limit = src["appointment_date"] + pd.to_timedelta(src["return_interval_days"] + FOLLOWUP_GRACE_DAYS, unit="D")
    src = src[limit <= end]
    src = _next_completed_after(src, a)
    limit = src["appointment_date"] + pd.to_timedelta(src["return_interval_days"] + FOLLOWUP_GRACE_DAYS, unit="D")
    late = src["next_completed_date"].isna() | (src["next_completed_date"] > limit)
    parts.append(pd.DataFrame({"study_patient_id": src["study_patient_id"], "missed": late.astype(int)}))

    if "preventive_care" in d:
        pc = d["preventive_care"]
        parts.append(pd.DataFrame({"study_patient_id": pc["study_patient_id"],
                                   "missed": pc["status"].ne("Completed on time").astype(int)}))
    r = d["referrals"]
    r = r[r["referral_date"] <= end - pd.Timedelta(days=90)]
    done90 = (r["referral_status"] == "Completed") & (r["days_to_status"] <= 90)
    parts.append(pd.DataFrame({"study_patient_id": r["study_patient_id"], "missed": (~done90).astype(int)}))

    agg = pd.concat(parts).groupby("study_patient_id")["missed"].agg(["sum", "count"])
    agg = agg[agg["count"] >= 5]
    agg["missed_care_rate"] = agg["sum"] / agg["count"]
    cut = agg["missed_care_rate"].quantile(1 - top_share)
    agg["high_risk"] = (agg["missed_care_rate"] >= cut).astype(int)
    p = p.merge(agg[["missed_care_rate", "high_risk"]], left_on="study_patient_id", right_index=True)

    feats = ["age65", "medicaid", "no_mychart", "interpreter_needed", "comorbidity_count",
             "sdoh_transportation", "sdoh_financial_strain", "sdoh_housing_instability",
             "sdoh_social_isolation", "sdoh_low_health_literacy"]
    df = p.dropna(subset=feats).copy()
    res = _logit("high_risk ~ " + " + ".join(feats), df)

    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(split.split(df, groups=df["study_patient_id"]))
    X, y = df[feats].astype(float).to_numpy(), df["high_risk"].to_numpy()
    lr = sm.GLM(y[tr], sm.add_constant(X[tr]), family=sm.families.Binomial()).fit()
    auc_lr = roc_auc_score(y[te], lr.predict(sm.add_constant(X[te])))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=seed).fit(X[tr], y[tr])
    auc_rf = roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1])

    df["risk"] = res.predict(df)
    top = df[df["risk"] >= df["risk"].quantile(0.9)]
    profile = pd.DataFrame({"feature": feats,
                            "top_10pct_mean": top[feats].mean().values,
                            "all_patients_mean": df[feats].mean().values})
    summary = pd.DataFrame([
        {"question": "Q8", "model": "Care-gap risk", "term": "test_auroc_logistic", "kind": "AUROC", "estimate": auc_lr},
        {"question": "Q8", "model": "Care-gap risk", "term": "test_auroc_random_forest", "kind": "AUROC", "estimate": auc_rf},
        {"question": "Q8", "model": "Care-gap risk", "term": "median_missed_care_rate", "kind": "share",
         "estimate": agg["missed_care_rate"].median()},
    ])
    return pd.concat([_tidy(res, "Q8", "Care-gap risk logistic"), summary], ignore_index=True), profile
