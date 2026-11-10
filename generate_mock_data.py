
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 0. CONFIGURATION
# ============================================================

# Reproducible synthetic data
rng = np.random.default_rng(42)
missing_rng = np.random.default_rng(123)

N_PATIENTS = 5000  # Change to 100 for initial testing

START = pd.Timestamp("2021-10-01")
END = pd.Timestamp("2026-09-30")

# Output directories
CLEAN_OUT = Path("data/synthetic/clean")
MESSY_OUT = Path("data/synthetic/messy")

CLEAN_OUT.mkdir(parents=True, exist_ok=True)
MESSY_OUT.mkdir(parents=True, exist_ok=True)


def random_dates(n, start=START, end=END):
    days = (end - start).days
    offsets = rng.integers(0, days + 1, size=n)
    return start + pd.to_timedelta(offsets, unit="D")


# ============================================================
# 1. PATIENTS
# ============================================================

patients = pd.DataFrame({
    "study_patient_id": [
        f"P{i:05d}" for i in range(1, N_PATIENTS + 1)
    ],
    "age": rng.integers(18, 91, N_PATIENTS),
    "sex": rng.choice(
        ["Female", "Male", "Other"],
        N_PATIENTS,
        p=[0.90, 0.09, 0.01]
    ),
    "pcp_id": rng.choice(
        ["PCP_A", "PCP_B", "PCP_C", "PCP_D", "PCP_E"],
        N_PATIENTS
    ),
    "insurance_type": rng.choice(
        ["Commercial", "Medicare", "Medicaid", "Other"],
        N_PATIENTS
    ),
    "mychart_active": rng.choice(
        [True, False],
        N_PATIENTS,
        p=[0.75, 0.25]
    )
})


# ============================================================
# 2. APPOINTMENTS
# ============================================================

# Some patients have zero appointments (important for Q9)
appointment_counts = rng.choice(
    [0, 1, 2, 3, 4, 5, 6],
    size=N_PATIENTS,
    p=[0.10, 0.14, 0.20, 0.22, 0.17, 0.11, 0.06]
)

appt_patient_ids = np.repeat(
    patients["study_patient_id"].to_numpy(),
    appointment_counts
)

n_appts = len(appt_patient_ids)

visit_dates = random_dates(n_appts)

# Booking dates precede appointment dates
lead_days = rng.integers(1, 121, n_appts)

booking_dates = visit_dates - pd.to_timedelta(
    lead_days,
    unit="D"
)

appointments = pd.DataFrame({
    "encounter_id": [
        f"E{i:06d}" for i in range(1, n_appts + 1)
    ],
    "study_patient_id": appt_patient_ids,
    "booking_date": booking_dates,
    "appointment_date": visit_dates,
    "appointment_status": rng.choice(
        [
            "Completed",
            "No-show",
            "Patient canceled",
            "Clinic canceled"
        ],
        n_appts,
        p=[0.78, 0.07, 0.10, 0.05]
    ),
    "visit_type": rng.choice(
        ["Annual Physical", "Follow-up", "New Patient"],
        n_appts
    ),
    "visit_mode": rng.choice(
        ["Office", "Video", "Phone"],
        n_appts,
        p=[0.80, 0.15, 0.05]
    )
})

# Some completed visits have no recorded return interval
appointments["return_interval_days"] = rng.choice(
    [90, 180, 365, np.nan],
    n_appts,
    p=[0.15, 0.20, 0.20, 0.45]
)

# Only completed appointments can have a recorded
# follow-up recommendation in this simplified model
appointments.loc[
    appointments["appointment_status"] != "Completed",
    "return_interval_days"
] = np.nan


# ============================================================
# 3. REFERRALS
# ============================================================

# Generate referrals from completed appointments
completed = appointments[
    appointments["appointment_status"] == "Completed"
].copy()

referral_sources = completed.sample(
    frac=0.18,
    random_state=42
).reset_index(drop=True)

n_referrals = len(referral_sources)

referrals = pd.DataFrame({
    "referral_id": [
        f"R{i:06d}" for i in range(1, n_referrals + 1)
    ],
    "study_patient_id": referral_sources[
        "study_patient_id"
    ],
    "source_encounter_id": referral_sources[
        "encounter_id"
    ],
    "referral_date": referral_sources[
        "appointment_date"
    ],
    "specialty": rng.choice(
        [
            "Cardiology",
            "Dermatology",
            "Neurology",
            "Endocrinology",
            "Orthopedics"
        ],
        n_referrals
    ),
    "referral_status": rng.choice(
        [
            "Completed",
            "Scheduled",
            "Pending",
            "Denied",
            "Closed without visit"
        ],
        n_referrals,
        p=[0.55, 0.15, 0.15, 0.05, 0.10]
    )
})

# Simulate referral completion dates
delay = pd.to_timedelta(
    rng.integers(3, 121, n_referrals),
    unit="D"
)

referrals["completion_date"] = (
    referrals["referral_date"] + delay
)

# A completed referral must have a completion date
# within the observation window
valid_completion = (
    (referrals["referral_status"] == "Completed")
    & (referrals["completion_date"] <= END)
)

referrals.loc[
    ~valid_completion,
    "completion_date"
] = pd.NaT

# Referrals completed after the observation window
# are treated as pending in this mock dataset
referrals.loc[
    (referrals["referral_status"] == "Completed")
    & ~valid_completion,
    "referral_status"
] = "Pending"


# ============================================================
# 4. HOSPITAL VISITS
# ============================================================

n_hospital = max(1, int(N_PATIENTS * 0.20))

hospital_visits = pd.DataFrame({
    "hospital_encounter_id": [
        f"H{i:06d}" for i in range(1, n_hospital + 1)
    ],
    "study_patient_id": rng.choice(
        patients["study_patient_id"],
        n_hospital,
        replace=True
    ),
    "admission_date": random_dates(n_hospital),
    "visit_category": rng.choice(
        ["ED", "Urgent Care", "Inpatient"],
        n_hospital,
        p=[0.55, 0.30, 0.15]
    )
})


# ============================================================
# 5. SAVE BASELINE SYNTHETIC DATA
# ============================================================

clean_tables = {
    "patients": patients,
    "appointments": appointments,
    "referrals": referrals,
    "hospital_visits": hospital_visits
}

print("\n=== Saving Baseline Synthetic Data ===")

for name, df in clean_tables.items():
    output_path = CLEAN_OUT / f"{name}.csv"

    df.to_csv(output_path, index=False)

    print(f"{name}: {len(df):,} rows saved")


# ============================================================
# 6. CREATE MESSY SYNTHETIC DATA
# ============================================================

# Create independent copies so baseline data is unchanged
messy_tables = {
    name: df.copy(deep=True)
    for name, df in clean_tables.items()
}


def introduce_missing(df, column, missing_rate):
    """
    Randomly introduce missing values into one column.
    """

    df = df.copy()

    if column not in df.columns:
        raise KeyError(f"Column not found: {column}")

    if not 0 <= missing_rate <= 1:
        raise ValueError(
            "missing_rate must be between 0 and 1"
        )

    # Convert column to a nullable pandas data type
    df[column] = df[column].convert_dtypes()

    # Generate missing value mask
    mask = missing_rng.random(len(df)) < missing_rate

    # Insert missing values
    df.loc[mask, column] = pd.NA

    return df



# ------------------------------------------------------------
# 6.1 Patient missing values
# ------------------------------------------------------------

messy_tables["patients"] = introduce_missing(
    messy_tables["patients"],
    "age",
    0.02
)

messy_tables["patients"] = introduce_missing(
    messy_tables["patients"],
    "insurance_type",
    0.10
)

messy_tables["patients"] = introduce_missing(
    messy_tables["patients"],
    "mychart_active",
    0.05
)


# ------------------------------------------------------------
# 6.2 Appointment missing values
# ------------------------------------------------------------

messy_tables["appointments"] = introduce_missing(
    messy_tables["appointments"],
    "appointment_status",
    0.02
)

# Note:
# return_interval_days already contains missing values
# from the original synthetic data generation.


# ============================================================
# 7. SAVE MESSY SYNTHETIC DATA
# ============================================================

print("\n=== Saving Messy Synthetic Data ===")

for name, df in messy_tables.items():
    output_path = MESSY_OUT / f"{name}.csv"

    df.to_csv(output_path, index=False)

    print(f"{name}: {len(df):,} rows saved")


# ============================================================
# 8. MISSING VALUE SUMMARY
# ============================================================

print("\n=== Missing Value Summary ===")

for name, df in messy_tables.items():

    print(f"\n{name.upper()}")

    missing_summary = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2)
    })

    print(missing_summary)


# ============================================================
# 9. BASIC DATA VALIDATION
# ============================================================

print("\n=== Data Validation ===")

# Check unique identifiers
assert clean_tables["patients"][
    "study_patient_id"
].is_unique

assert clean_tables["appointments"][
    "encounter_id"
].is_unique

assert clean_tables["referrals"][
    "referral_id"
].is_unique

assert clean_tables["hospital_visits"][
    "hospital_encounter_id"
].is_unique

# Check patient linkage
patient_ids = set(
    clean_tables["patients"]["study_patient_id"]
)

for name in [
    "appointments",
    "referrals",
    "hospital_visits"
]:
    assert set(
        clean_tables[name]["study_patient_id"]
    ).issubset(patient_ids)

# Check referral-to-encounter linkage
assert clean_tables["referrals"][
    "source_encounter_id"
].isin(
    clean_tables["appointments"]["encounter_id"]
).all()

# Check appointment date logic
assert (
    clean_tables["appointments"]["booking_date"]
    <= clean_tables["appointments"]["appointment_date"]
).all()

# Verify identical row counts between clean and messy
for name in clean_tables:
    assert len(clean_tables[name]) == len(
        messy_tables[name]
    )

print("All basic data validation checks passed.")

print("\nSynthetic data generation completed successfully!")
