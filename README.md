# langone-patient-pathways

NYU Tandon TMI Capstone (Fall 2026), sponsored project with NYU Langone Health.

We study where patients drop off on the primary care pathway at the Center for Women's Health: missed
appointments, long waits, follow-ups and referrals that never happen, and what happens downstream
(ED, urgent care, hospital). The research questions Q1–Q9 come from our data plan (Epic data request).

> **All results in this repository use synthetic data.** The Epic extract is not available yet.
> Numbers in `results/` test the analysis pipeline; they are not findings about NYU Langone patients.

## Repository layout

```
config/scenarios.yaml        Parameters for the synthetic data (planted effects, with sources)
data/scenario/calibrated/    Synthetic data used by all models (5,000 patients, Oct 2021 – Sep 2026)
src/models/                  One script per model
results/<model>_calibrated/  Tables and figures each model produces
notebooks/                   Exploration notebooks
archive/                     Earlier mock data, scripts and checks (kept for reference)
```

## Models

| Model | Questions | Method | Run | Output |
| --- | --- | --- | --- | --- |
| Model 1: Appointment attendance | Q1 | Logistic regression for no-show and cancellation; adjusted probabilities by lead time, visit type, and MyChart × age 65+; held-out AUROC | `python src/models/model1.py` | `results/model1_calibrated/` |
| Model 2: Patient pathway and access | Q3, Q6, Q9 | Drop-off metrics D1–D14, Gamma regression on wait time, 12-month care gaps | `python src/models/model2.py` | `results/model2_calibrated/` |
| Model 3: Follow-up and referrals | Q2, Q4 | Logistic regression, Kaplan–Meier, Cox regression | in progress | |
| Model 4: Downstream utilization | Q7 | Logistic regression on ED / urgent care / hospital use within 30, 90, 180 days | `python src/models/model4.py` | `results/model4_calibrated/` |

Q5 (messages) and Q8 (highest-risk patients) are planned after Models 1–4.

## Getting started

```bash
pip install -r requirements.txt

# run a model (defaults to data/scenario/calibrated)
python src/models/model2.py

# point a model at another data folder
python src/models/model4.py --data-dir data/scenario/calibrated --output-dir results/model4_calibrated
```

## Synthetic data

`data/scenario/calibrated/` was built by `archive/generate_scenario_data.py` from `config/scenarios.yaml`.
Effects are planted on purpose and calibrated to our interviews and published benchmarks, so the models
have known patterns to find. To rebuild it, or the other scenarios (`no_effect`, `patient_driven`,
`system_driven`):

```bash
python archive/generate_scenario_data.py --scenario calibrated
```

Known simplifications: every completed visit has a true next visit (no "no follow-up needed" case), and
there is no "return to PCP after ED" mechanism, so D14 and Q7.2 are placeholders.
