#!/usr/bin/env python3
"""Model 3: Q2 follow-up / preventive care / orders and Q4 referral completion.

Calibrated synthetic data only. No outputs are patient findings.

Run:
 python src/models/model3.py --data-dir data/scenario/calibrated --output-dir results/model3_calibrated

Dependencies: pandas numpy matplotlib statsmodels scipy patsy
All source input tables are read-only. A missing optional analysis field causes a
clear skip notice, not a fabricated result.
"""
from __future__ import annotations
import argparse
import json
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from scipy.special import expit

PURPLE = '#57068C'
LAVENDER = '#B89BCC'
OBS_END_DEFAULT = '2026-09-30'
SEED = 2026


def col(df, *names):
    for name in names:
        if name in df.columns:
            return name
    return None


def normalized(df):
    d = df.copy()
    d.columns = [re.sub(r'[^a-z0-9]+', '_', str(x).lower()).strip('_') for x in d.columns]
    return d


def as_bool(s):
    return s.astype('string').str.lower().str.strip().map({
        'true': 1., 'false': 0., '1': 1., '0': 0., '1.0': 1., '0.0': 0., 'yes': 1., 'no': 0.,
        'y': 1., 'n': 0., 'active': 1., 'inactive': 0.})


def dates(s):
    return pd.to_datetime(s, errors='coerce')


def save_fig(fig, path):
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  figure:', path.name)


def setup_ax(ax, title, y_label='Adjusted probability (%)'):
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(y_label)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)


def load_tables(root):
    tables = {}
    for name in ['patients', 'appointments', 'preventive_care', 'orders', 'referrals']:
        path = root / f'{name}.csv'
        if path.exists():
            tables[name] = normalized(pd.read_csv(path, low_memory=False))
        else:
            print(f'WARNING: {path} missing; corresponding analysis will be skipped')
    return tables


def add_patient_covariates(episodes, patients):
    pkey = col(patients, 'study_patient_id', 'patient_id')
    ekey = col(episodes, 'study_patient_id', 'patient_id')
    if not pkey or not ekey:
        raise ValueError('Patient linkage key missing')
    if patients[pkey].duplicated().any():
        raise ValueError('patients primary key not unique')
    available = [x for x in ['age', 'insurance_type', 'mychart_active', 'asymptomatic_chronic',
                              'sdoh_social_isolation', 'sdoh_low_health_literacy',
                              'comorbidity_count'] if x in patients and x not in episodes]
    p = patients[[pkey] + available].rename(columns={pkey: '__patient_key'})
    left = episodes.copy()
    left['__patient_key'] = left[ekey]
    return left.merge(p, on='__patient_key', how='left', validate='many_to_one').drop(columns='__patient_key')


def model_ready(df, outcome, proposed):
    data = df.copy()
    terms = []
    for name, kind in proposed:
        if name not in data:
            continue
        if kind == 'binary':
            data[name] = as_bool(data[name])
            if data[name].dropna().nunique() > 1:
                terms.append(name)
        elif kind == 'numeric':
            data[name] = pd.to_numeric(data[name], errors='coerce')
            if data[name].dropna().nunique() > 1:
                terms.append(name)
        elif kind == 'category':
            data[name] = data[name].astype('string').str.strip().replace({'': pd.NA})
            if 1 < data[name].dropna().nunique() <= 15:
                data[name] = data[name].astype('object')
                terms.append(f'C({name})')
    columns = [outcome] + [t[2:-1] if t.startswith('C(') else t for t in terms]
    data[outcome] = pd.to_numeric(data[outcome], errors='coerce')
    data = data.dropna(subset=columns).copy()
    return data, terms


def fit_glm(df, outcome, proposed, name, out):
    data, terms = model_ready(df, outcome, proposed)
    if len(data) < 50 or data[outcome].nunique() != 2 or min(data[outcome].value_counts()) < 10:
        print(f'SKIP {name}: insufficient observed events / variable completeness')
        return None
    # Limit sparse-category / complete-separation risk by requiring counts >= 10
    for term in list(terms):
        if term.startswith('C('):
            var = term[2:-1]
            if ((lambda z: (z['count'] - z['sum']).min() < 5 or z['sum'].min() < 5)(
                    data.groupby(var)[outcome].agg(['sum', 'count']))):
                terms.remove(term)
    formula = outcome + ' ~ ' + (' + '.join(terms) if terms else '1')
    # Explicitly save Patsy's design information for future standardized predictions.
    # statsmodels versions do not all expose fit.model.data.design_info.
    design_info = patsy.dmatrices(formula, data=data, return_type='dataframe')[1].design_info
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fit = smf.glm(formula, data=data, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': data['study_patient_id']}
            ) if 'study_patient_id' in data else smf.glm(
                formula, data=data, family=sm.families.Binomial()
            ).fit(cov_type='HC3')
    except Exception as exc:
        print(f'SKIP {name}: regression failed: {exc}')
        return None
    fit._m3_design_info = design_info
    coeffs = pd.DataFrame({'term': fit.params.index,
                           'odds_ratio': np.exp(np.clip(fit.params, -40, 40)),
                           'ci_low': np.exp(np.clip(fit.conf_int().iloc[:,0], -40, 40)),
                           'ci_high': np.exp(np.clip(fit.conf_int().iloc[:,1], -40, 40)),
                           'p_value': fit.pvalues})
    coeffs.to_csv(out / f'{name}_odds_ratios.csv', index=False)
    (out / f'{name}_model.txt').write_text(fit.summary().as_text())
    print(f'  model {name}: N={len(data)}, events={int(data[outcome].sum())}, terms={len(terms)}')
    return fit, data


def standardized_prob(fit, data, var, values, seed=SEED, draws=500):
    """Average counterfactual predictions and coefficient-draw 95% CIs.
    Statistical adjustment only; not causal impact. Uses robust fit covariance.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(fit.params, dtype=float)
    cov = np.asarray(fit.cov_params(), dtype=float)
    cov = (cov + cov.T)/2
    eigen, vec = np.linalg.eigh(cov)
    cov_psd = (vec * np.maximum(eigen, 0)) @ vec.T
    draws_beta = rng.multivariate_normal(beta, cov_psd, size=draws, check_valid='ignore')
    rows = []
    for v in values:
        d = data.copy()
        d[var] = v
        try:
            info = getattr(fit, '_m3_design_info', None)
            if info is None:
                info = getattr(fit.model.data, 'design_info', None)
            if info is None:
                info = getattr(getattr(fit.model.data, 'orig_exog', None), 'design_info', None)
            if info is None:
                raise ValueError('Patsy design information unavailable')
            design = patsy.build_design_matrices([info], d, return_type='dataframe')[0]
            design = design.loc[:, fit.params.index]
        except Exception as exc:
            print(f'  SKIP comparison {var}={v}: {exc}')
            continue
        X = design.to_numpy()
        p = expit(X @ beta).mean()
        sim = np.array([expit(X @ b).mean() for b in draws_beta])
        low, high = np.quantile(sim, [.025,.975])
        rows.append(dict(level=str(v), probability=p, low=low, high=high))
    return pd.DataFrame(rows)


def bar_probabilities(tab, title, dest, labels=None):
    if tab.empty:
        return
    labs = [labels.get(v, v) if labels else v for v in tab['level']]
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(8,4.7))
    y = tab.probability.to_numpy()*100
    err = np.array([y-tab.low.to_numpy()*100, tab.high.to_numpy()*100-y])
    ax.bar(x, y, width=.52, color=PURPLE, alpha=.85)
    ax.errorbar(x,y,yerr=err,fmt='none',color='#333333',capsize=5)
    ax.set_xticks(x, labs)
    setup_ax(ax,title)
    ax.set_ylim(0,max(8,float((tab.high.max()*100))*1.25))
    ax.text(.01,-.20,'Calibrated synthetic data — not patient findings',
            transform=ax.transAxes,fontsize=9,color='#555')
    fig.tight_layout()
    save_fig(fig,dest)


def followup(tables, out, end):
    a = tables['appointments'].copy()
    p = tables['patients']
    pid = col(a,'study_patient_id','patient_id')
    date = col(a,'appointment_date','visit_date')
    status = col(a,'appointment_status','status')
    interval = col(a,'return_interval_days','recommended_return_days')
    same = col(a,'followup_booked_same_day','booked_at_checkout')
    if not all([pid,date,status,interval,same]):
        raise ValueError('Follow-up requires patient ID, visit date, status, return interval, same-day booking flag')
    a[date] = dates(a[date]); a[interval] = pd.to_numeric(a[interval],errors='coerce')
    a = a.sort_values([pid,date]).reset_index(drop=True)
    # Only completed index visits with documented recommendation; do not infer one from blank interval.
    is_completed = a[status].astype('string').str.lower().str.strip().eq('completed')
    completed = a.loc[is_completed & a[date].notna(),[pid,date]].sort_values([pid,date])
    index = a.loc[is_completed & a[interval].between(1,730) & a[date].notna()].copy()
    if index.empty:
        raise ValueError('No completed visits with documented return intervals')
    # Next completed visit STRICTLY after index visit; same-day visits are not follow-up.
    next_list = completed.rename(columns={date:'next_completed_visit'})
    index = pd.merge_asof(index.sort_values(date), next_list.sort_values('next_completed_visit'),
                          left_on=date,right_on='next_completed_visit',by=pid,
                          direction='forward',allow_exact_matches=False)
    index['deadline'] = index[date] + pd.to_timedelta(index[interval]+30,unit='D')
    index['mature'] = index['deadline'] <= end
    index['on_time'] = (index.next_completed_visit.notna() &
                        (index.next_completed_visit <= index.deadline)).astype(int)
    eligible = index.loc[index.mature].copy()
    eligible['same_day'] = as_bool(eligible[same])
    eligible = add_patient_covariates(eligible,p)
    eligible.rename(columns={pid:'study_patient_id'},inplace=True) if pid != 'study_patient_id' else None
    eligible.to_csv(out/'followup_episodes.csv',index=False)
    model = fit_glm(eligible,'on_time', [
        ('same_day','binary'),('age','numeric'),('insurance_type','category'),
        ('mychart_active','binary'),('asymptomatic_chronic','binary'),
        ('sdoh_social_isolation','binary'),('sdoh_low_health_literacy','binary'),
        ('comorbidity_count','numeric')], 'followup_adherence', out)
    if model and 'same_day' in model[0].params.index:
        fit,d = model
        probs = standardized_prob(fit,d,'same_day',[0,1])
        probs.to_csv(out/'fig1_followup_adjusted_probability.csv',index=False)
        bar_probabilities(probs,'Figure 1 · Adjusted probability of on-time PCP follow-up',
                          out/'fig1_followup_adjusted_probability.png',{'0':'Not booked same day','1':'Booked same day'})
    return {'episodes':len(index),'mature_episodes':len(eligible)}


def preventive(tables,out,end):
    d=tables['preventive_care'].copy()
    topic=col(d,'topic','care_topic','topic_name','preventive_care_type','screening_type','hm_topic','hm_topic_name','hm_topic_id_name','active_subtopic_id_name','item')
    due=col(d,'due_date','next_due_date','recommended_due_date')
    status=col(d,'status','care_status','preventive_status','hm_status_c_name')
    patient=col(d,'study_patient_id','patient_id')
    if not topic or not patient or (not due and not status):
        raise ValueError(f'Preventive data needs patient/topic and due date or status; columns={list(d)}')
    # Snapshot status refers to available observed state, not lifetime history.
    if due:
        d[due]=dates(d[due]); d=d[d[due].notna() & (d[due]<=end)].copy()
    if status:
        s=d[status].astype('string').str.lower().str.strip()
        overdue=s.str.contains('overdue',na=False)
        complete=s.str.contains('complete|done',na=False)
        postponed=s.str.contains('postpon',na=False)
        if due:
            # Do not call a due record overdue when source explicitly says completed/postponed.
            overdue=overdue | ((d[due]<end) & ~complete & ~postponed & ~s.str.contains('not due|due soon',na=False))
        d['overdue']=overdue.astype(int)
    else:
        print('  WARN: no preventive status; overdue approximated from due date only, not actual completion')
        d['overdue']=(d[due]<end).astype(int)
    if d.empty:
        raise ValueError('No preventive topics due by study end')
    summary=d.groupby(topic,dropna=False).agg(n_due=('overdue','size'),n_overdue=('overdue','sum')).reset_index()
    summary['overdue_rate']=summary.n_overdue/summary.n_due
    summary=summary.sort_values('overdue_rate',ascending=False)
    summary.to_csv(out/'fig2_preventive_overdue_by_topic.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,max(4, .5*len(summary)+1)))
    ax.barh(summary[topic].astype(str),summary.overdue_rate*100,color=PURPLE)
    for i,(_,r) in enumerate(summary.iterrows()):
        ax.text(r.overdue_rate*100+.7,i,f"{r.overdue_rate:.0%}  (n={int(r.n_due)})",va='center',fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,max(100,summary.overdue_rate.max()*120))
    ax.set_xlabel('Overdue (%) among records due by observation end')
    ax.set_title('Figure 2 · Preventive care overdue by topic',fontweight='bold')
    ax.text(0,-.16,'Calibrated synthetic data — not patient findings',transform=ax.transAxes,color='#555',fontsize=9)
    fig.tight_layout();save_fig(fig,out/'fig2_preventive_overdue_by_topic.png')
    return {'due_records':len(d)}


def orders(tables,out,end):
    o=tables['orders'].copy()
    pid=col(o,'study_patient_id','patient_id')
    st=col(o,'order_status','status','order_status_c_name')
    order_date=col(o,'order_date','ordering_date','ordered_date','created_date')
    due=col(o,'expected_completion_date','due_date','future_expected_completion_date','fut_expect_comp_dt')
    result_date=col(o,'result_time','completion_date','completed_date')
    if not pid or (not st and not result_date):
        raise ValueError(f'Order completion requires patient ID and order status or completed date; columns={list(o)}')
    if order_date:
        o[order_date]=dates(o[order_date]);o=o[o[order_date].le(end)].copy()
    if due:
        o[due]=dates(o[due]);o=o[o[due].le(end)].copy()
        print('  Order cohort restricted to due dates within observed window')
    elif order_date:
        # Do not classify late orders as uncompleted if observation window insufficient
        o=o[o[order_date].le(end-pd.Timedelta(days=90))].copy()
        print('  WARN: no order due date, using >=90-day observation cutoff')
    else:
        print('  WARN: cannot verify order follow-up time; use descriptive status only')
    # If a due date is unavailable, use completion within 90 days as a
    # transparent synthetic-data proxy, not an actual clinical due window.
    if st:
        o['completed'] = o[st].astype('string').str.lower().str.contains('completed|complete|resulted|done',na=False).astype(int)
    else:
        o['completed'] = 0
    if result_date:
        completed_at = dates(o[result_date])
        valid_completion = completed_at.notna() & completed_at.le(end)
        if not due and order_date:
            valid_completion &= completed_at.ge(o[order_date]) & completed_at.le(o[order_date] + pd.Timedelta(days=90))
        o['completed'] = valid_completion.astype(int)
        print('  Order completion based on completed_date (90-day proxy when due_date unavailable)')
    if o.empty:
        raise ValueError('No eligible orders')
    o=add_patient_covariates(o,tables['patients'])
    o.rename(columns={pid:'study_patient_id'},inplace=True) if pid != 'study_patient_id' else None
    auth=col(o,'prior_auth_required','prior_authorization_required','requires_prior_auth')
    low=col(o,'sdoh_low_health_literacy')
    ordertype=col(o,'order_type','order_type_c_name','type')
    proposals=[]
    if auth:proposals.append((auth,'binary'))
    if low:proposals.append((low,'binary'))
    if ordertype:proposals.append((ordertype,'category'))
    for name,kind in [('age','numeric'),('insurance_type','category'),('comorbidity_count','numeric')]:
        proposals.append((name,kind))
    o.to_csv(out/'order_episodes.csv',index=False)
    model=fit_glm(o,'completed',proposals,'order_completion',out)
    if model:
        fit,data=model
        plot_vars=[(auth,'Prior authorization required') if auth else None,
                   (low,'Low health literacy') if low else None]
        plot_vars=[x for x in plot_vars if x and x[0] in fit.params.index]
        if plot_vars:
            fig,axs=plt.subplots(1,len(plot_vars),figsize=(6*len(plot_vars),4.5),squeeze=False)
            allrows=[]
            for ax,(name,label) in zip(axs[0],plot_vars):
                estimates=standardized_prob(fit,data,name,[0,1]);estimates['comparison']=label
                allrows.append(estimates)
                x=np.arange(len(estimates)); y=estimates.probability.to_numpy()*100
                err=[y-estimates.low.to_numpy()*100,estimates.high.to_numpy()*100-y]
                ax.bar(x,y,color=[LAVENDER,PURPLE],width=.55)
                ax.errorbar(x,y,yerr=err,fmt='none',color='#333',capsize=4)
                ax.set_xticks(x,['No','Yes']); setup_ax(ax,label,'Adjusted completion probability (%)')
                ax.set_ylim(0,100)
            pd.concat(allrows).to_csv(out/'fig3_order_adjusted_completion.csv',index=False)
            fig.suptitle('Figure 3 · Adjusted test / imaging order completion',fontweight='bold')
            fig.text(.1,.01,'Calibrated synthetic data — not patient findings',fontsize=9,color='#555')
            fig.tight_layout(rect=[0,.04,1,.95]);save_fig(fig,out/'fig3_order_adjusted_completion.png')
        else:
            print('  SKIP Figure 3 adjusted comparisons: no usable order-level prior auth / literacy predictors')
    return {'order_records':len(o)}


def referrals(tables,out,end):
    r=tables['referrals'].copy()
    pid=col(r,'study_patient_id','patient_id')
    ref_date=col(r,'referral_date','entry_date','created_date')
    status=col(r,'referral_status','status','rfl_status_c_name')
    status_date=col(r,'status_date','completion_date','decision_date')
    days_field=col(r,'days_to_status','days_to_completion')
    auth=col(r,'prior_auth_required','prior_authorization_required','preauth_required')
    priority=col(r,'priority','referral_priority')
    specialty=col(r,'specialty','referred_specialty')
    reason=col(r,'close_reason','closure_reason')
    if not all([pid,ref_date,status]) or (not status_date and not days_field):
        raise ValueError(f'Referral data missing ID, referral date, status or event time; columns={list(r)}')
    r[ref_date]=dates(r[ref_date]);r=r[r[ref_date].notna() & r[ref_date].le(end)].copy()
    s=r[status].astype('string').str.lower().str.strip()
    is_done=s.str.contains('completed|complete',na=False)
    is_terminal_other=s.str.contains('denied|closed|cancel|refused',na=False)
    if status_date:
        event_date=dates(r[status_date])
        t=(event_date-r[ref_date]).dt.days
    else:
        t=pd.to_numeric(r[days_field],errors='coerce')
        event_date=r[ref_date]+pd.to_timedelta(t,unit='D')
    if days_field:
        t=t.fillna(pd.to_numeric(r[days_field],errors='coerce'))
        event_date=event_date.fillna(r[ref_date]+pd.to_timedelta(t,unit='D'))
    r['end_of_observation_days']=(end-r[ref_date]).dt.days
    r['time_days']=np.where((is_done|is_terminal_other) & t.ge(0) & event_date.le(end),
                             t,r.end_of_observation_days)
    r['event_completed']=((is_done & t.ge(0) & event_date.le(end))).astype(int)
    r['competing_closure']=((is_terminal_other & t.ge(0) & event_date.le(end))).astype(int)
    r=r[r.time_days.notna() & r.time_days.ge(0)].copy()
    r['time_days']=r.time_days.clip(lower=.01)
    r=add_patient_covariates(r,tables['patients'])
    r.rename(columns={pid:'study_patient_id'},inplace=True) if pid!='study_patient_id' else None
    r.to_csv(out/'referral_episodes.csv',index=False)
    # Figure 4: descriptive competing-risk cumulative incidence (AJ), rather than KM.
    if auth:
        ar=as_bool(r[auth]); fig,ax=plt.subplots(figsize=(8,5)); valid=False
        for val,lbl,color in [(0,'No prior auth',PURPLE),(1,'Prior auth required',LAVENDER)]:
            d=r[ar.eq(val)].copy()
            if len(d)<15:continue
            event_code=np.select([d.event_completed.eq(1),d.competing_closure.eq(1)],[1,2],default=0)
            # Aalen–Johansen empirical cumulative incidence, event 2 competing.
            durations=d.time_days.to_numpy(dtype=float)
            event_code=np.asarray(event_code)
            event_times=np.sort(np.unique(durations[event_code>0]))
            survival=1.0; cumulative=0.0
            xx=[0.];yy=[0.]
            for day in event_times:
                n_risk=np.sum(durations>=day)
                event1=np.sum((durations==day)&(event_code==1))
                event2=np.sum((durations==day)&(event_code==2))
                if n_risk == 0: continue
                cumulative += survival*event1/n_risk
                survival *= 1-(event1+event2)/n_risk
                xx.append(day);yy.append(cumulative*100)
            ax.step(xx,yy,where='post',label=f'{lbl} (n={len(d)})',color=color,linewidth=2)
            valid=True
        if valid:
            ax.set_xlim(0, min(180,max(91,float(r.time_days.max()))))
            ax.set_ylim(0,100)
            ax.set_xlabel('Days since referral')
            setup_ax(ax,'Figure 4 · Cumulative incidence of completed referral','Completed referrals (%)')
            ax.legend();fig.tight_layout();save_fig(fig,out/'fig4_referral_cumulative_completion.png')
        else: plt.close(fig)
    else:print('  SKIP Figure 4 by prior auth: field not present')
    # Figure 5: 90-day completion regression among referrals observed for >=90 days
    mature=r[r.end_of_observation_days.ge(90)].copy()
    mature['completed_by_90']=((mature.event_completed.eq(1)) & mature.time_days.le(90)).astype(int)
    props=[]
    if auth:props.append((auth,'binary'))
    if priority:props.append((priority,'category'))
    if specialty:props.append((specialty,'category'))
    props += [('age','numeric'),('insurance_type','category')]
    model=fit_glm(mature,'completed_by_90',props,'referral_day90_completion',out)
    if model:
        fit,data=model
        compare=[]
        if auth and auth in fit.params.index:
            compare.append((auth,[0,1],'Prior authorization required'))
        if priority and f'C({priority})' in fit.model.formula:
            compare.append((priority,sorted(data[priority].dropna().unique().tolist()),'Referral priority'))
        if specialty and f'C({specialty})' in fit.model.formula:
            compare.append((specialty,sorted(data[specialty].dropna().unique().tolist()),'Specialty'))
        allrows=[]
        for c,levels,label in compare:
            temp=standardized_prob(fit,data,c,levels);temp['comparison']=label
            allrows.append(temp)
        if allrows:
            final=pd.concat(allrows,ignore_index=True)
            final.to_csv(out/'fig5_referral_day90_adjusted_probabilities.csv',index=False)
            fig,axes=plt.subplots(len(compare),1,figsize=(9,max(4,len(compare)*3.2)),squeeze=False)
            for ax,(c,levels,label) in zip(axes[:,0],compare):
                v=final[final.comparison.eq(label)]
                x=np.arange(len(v));y=v.probability.to_numpy()*100
                ax.bar(x,y,color=PURPLE,alpha=.82)
                ax.errorbar(x,y,yerr=[y-v.low.to_numpy()*100,v.high.to_numpy()*100-y],
                            fmt='none',ecolor='#333',capsize=4)
                ax.set_xticks(x,v.level,rotation=18,ha='right');ax.set_ylim(0,100)
                ax.set_ylabel('Completed by day 90 (%)'); ax.set_title(label,fontsize=11)
            fig.suptitle('Figure 5 · Adjusted referral completion at day 90',fontweight='bold')
            fig.tight_layout(rect=[0,0,1,.96]);save_fig(fig,out/'fig5_referral_day90_adjusted_probabilities.png')
    # Figure 6: descriptive closure reasons, no unsupported causality classification.
    if reason:
        closed=r.loc[r.competing_closure.eq(1),reason].fillna('Unknown').astype(str)
        summary=closed.value_counts().rename_axis('close_reason').reset_index(name='count')
        if not summary.empty:
            summary['share']=summary['count']/summary['count'].sum()
            def classify(x):
                z=x.lower()
                if 'no appointment' in z or 'insurance' in z or 'authoriz' in z:
                    return 'System / coverage barrier'
                if 'care elsewhere' in z:return 'Care elsewhere (outcome uncertain)'
                if 'resolved' in z or 'no longer needed' in z:return 'Possibly appropriate closure'
                return 'Unknown / requires review'
            summary['interpretation']=summary.close_reason.apply(classify)
            summary.to_csv(out/'fig6_referral_closure_reasons.csv',index=False)
            fig,ax=plt.subplots(figsize=(9,max(4,len(summary)*.6+1)))
            ax.barh(summary.close_reason,summary.share*100,color=PURPLE)
            for i,row in enumerate(summary.itertuples(index=False)):
                ax.text(row.share*100+.5,i,f'{row.share:.0%} (n={row.count})',va='center')
            ax.invert_yaxis();ax.set_xlim(0,max(100,summary.share.max()*120))
            ax.set_xlabel('Share of terminal closures (%)')
            ax.set_title('Figure 6 · Referral closure reasons',fontweight='bold')
            fig.tight_layout();save_fig(fig,out/'fig6_referral_closure_reasons.png')
    else:print('  SKIP Figure 6: closure reason field not available')
    return {'referral_episodes':len(r),'mature_90day':len(mature),'completed_events':int(r.event_completed.sum()),
            'other_terminal_closures':int(r.competing_closure.sum())}


def main():
    parser=argparse.ArgumentParser(description='Model 3 Q2 + Q4 on calibrated synthetic data')
    parser.add_argument('--data-dir',type=Path,default=Path('data/scenario/calibrated'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/model3_calibrated'))
    parser.add_argument('--end-date',default=OBS_END_DEFAULT)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    end=pd.Timestamp(args.end_date)
    print('MODEL 3 · Calibrated synthetic data — not patient findings')
    tables=load_tables(args.data_dir)
    results={}
    groups=[('followup',['patients','appointments'],followup),
            ('preventive',['preventive_care'],preventive),
            ('orders',['patients','orders'],orders),
            ('referrals',['patients','referrals'],referrals)]
    for name,required,fn in groups:
        if any(x not in tables for x in required):
            results[name]={'status':'skipped','reason':'required table missing'};continue
        try:
            print('\n'+name.upper())
            summary=fn(tables,args.output_dir,end)
            results[name]={'status':'ok',**summary}
        except Exception as e:
            results[name]={'status':'skipped','reason':str(e)}
            print(f'  SKIPPED {name}: {e}')
    (args.output_dir/'model3_summary.json').write_text(json.dumps({
        'data_type':'calibrated synthetic, not patient findings',
        'data_dir':str(args.data_dir),'observation_end':str(end.date()),'modules':results,
        'interpretation_notes':[
            'Adjusted comparisons are statistical standardizations, not intervention effects.',
            'Follow-up analysis uses documented recommended return intervals only.',
            'Preventive and order status fields require clinical definition review.',
            'Referral Figure 4 uses cumulative incidence with recorded terminal closures as competing events.',
            'Referral day-90 regression includes only episodes with >=90 calendar days observation.',
            'Care outside the system is not observable in these synthetic extracts.'
        ]},indent=2))
    print('\nSUMMARY:',json.dumps(results,indent=2))
    print('\nOutputs:',args.output_dir)

if __name__=='__main__':
    main()
