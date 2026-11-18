# Scenario data (synthetic)

Each folder here is one simulated population of 5,000 patients (Oct 2021 to Sep 2026), built by
`archive/generate_scenario_data.py` from the parameters in `config/scenarios.yaml`.

**These are not findings about NYU Langone patients.** The effects were planted on purpose, calibrated
to our interviews and published benchmarks, so we can check that the models recover them.

| Folder | What it is |
| --- | --- |
| `no_effect/` | Every planted effect switched off. Models should find nothing. |
| `calibrated/` | Effects set to the evidence-calibrated values in the modeling plan. |
| `patient_driven/` | Barriers mostly on the patient side (MyChart, age, prior no-shows, social drivers). |
| `system_driven/` | Barriers mostly on the system side (lead time, prior auth, clinic cancellations, PAC delays). |

Only `calibrated/` is stored in GitHub. The other three are rebuilt locally in a few seconds with
`python archive/generate_scenario_data.py` (same seed, identical data).

Each folder holds `patients`, `appointments`, `referrals`, `hospital_visits`, `orders`, `preventive_care`,
`messages` (CSV) and `scenario_params.json`, the exact values planted in that folder.

## How to use

```bash
# 1. (Re)build the data, all scenarios or one
python archive/generate_scenario_data.py
python archive/generate_scenario_data.py --scenario calibrated

# 2. Run the models (each reads data/scenario/calibrated by default)
python src/models/model1.py
python src/models/model2.py
python src/models/model4.py

# 3. Point a model at another scenario folder
python src/models/model4.py --data-dir data/scenario/system_driven --output-dir results/model4_system_driven
```

To change a scenario, edit `config/scenarios.yaml` and rebuild. Same seed, same data.

## Columns added beyond the old mock data

- patients: race, ethnicity, preferred_language, interpreter_needed, panel_size, sdoh_screened,
  sdoh_* (blank when not screened, about 35%), comorbidity_count, asymptomatic_chronic, new_patient_at_start
- appointments: booking_channel, booked_at_checkout, followup_booked_same_day, return_interval_days
  (blank when not documented), status now includes Late canceled and Clinic canceled
- referrals: priority, prior_auth_required, referred_outside_nyu, status_date, days_to_status, close_reason
- new tables: orders (Q2.3), preventive_care (Q2.2), messages (Q5)

## Things to know when reading results

- About 1 in 20 planted values will fall outside its 95% CI by chance. Check a few seeds before calling a miss a bug.
- The MyChart × 65+ interaction is noisy: the 65+ without MyChart group is small.
- Q2 follow-up and Q7 effects are planted on an underlying step (returning on time, a rate ratio),
  so their estimates are close to, not exactly, the planted value.
- Q7 unadjusted is meant to come out too high: comorbidity drives both missed visits and ED use.
