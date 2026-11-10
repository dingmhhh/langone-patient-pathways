
import pandas as pd
from pathlib import Path

# ============================================================
# 0. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "messy"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    names = [
        "patients",
        "appointments",
        "referrals",
        "hospital_visits"
    ]

    return {
        name: pd.read_csv(INPUT_DIR / f"{name}.csv")
        for name in names
    }


# ============================================================
# 2. PATIENT CLEANING
# ============================================================

def clean_patients(df):
    df = df.copy()

    # Convert age to numeric, preserving missing values
    df["age"] = pd.to_numeric(
        df["age"], errors="coerce"
    )

    # Flag impossible ages rather than deleting patients
    df["invalid_age"] = (
        df["age"].notna()
        & ~df["age"].between(0, 120)
    )

    df.loc[df["invalid_age"], "age"] = pd.NA

    # Normalize insurance categories
    df["insurance_type"] = (
        df["insurance_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Preserve True, False, and missing
    df["mychart_active"] = (
        df["mychart_active"]
        .astype("boolean")
    )

    # Flags for missing patient-level information
    df["age_missing"] = df["age"].isna()
    df["insurance_missing"] = df["insurance_type"].isna()
    df["mychart_missing"] = df["mychart_active"].isna()

    return df


# ============================================================
# 3. APPOINTMENT CLEANING
# ============================================================

def clean_appointments(df):
    df = df.copy()

    # Parse dates
    date_columns = [
        "booking_date",
        "appointment_date"
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        )

    # Standardize appointment status
    df["appointment_status"] = (
        df["appointment_status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Standardized status labels
    status_mapping = {
        "completed": "Completed",
        "no-show": "No-show",
        "patient canceled": "Patient canceled",
        "clinic canceled": "Clinic canceled"
    }

    df["appointment_status"] = (
        df["appointment_status"].map(status_mapping)
    )

    # Preserve original missing status
    df["status_missing"] = (
        df["appointment_status"].isna()
    )

    # Calculate appointment lead time
    df["lead_time_days"] = (
        df["appointment_date"] - df["booking_date"]
    ).dt.days

    # Flag invalid booking dates
    df["invalid_lead_time"] = (
        df["lead_time_days"].notna()
        & (df["lead_time_days"] < 0)
    )

    # Follow-up interval remains missing if unknown
    df["return_interval_days"] = pd.to_numeric(
        df["return_interval_days"],
        errors="coerce"
    )

    df["return_interval_missing"] = (
        df["return_interval_days"].isna()
    )

    return df


# ============================================================
# 4. REFERRAL CLEANING
# ============================================================

def clean_referrals(df):
    df = df.copy()

    df["referral_date"] = pd.to_datetime(
        df["referral_date"], errors="coerce"
    )

    df["completion_date"] = pd.to_datetime(
        df["completion_date"], errors="coerce"
    )

    df["referral_status"] = (
        df["referral_status"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["days_to_completion"] = (
        df["completion_date"] - df["referral_date"]
    ).dt.days

    df["invalid_completion_date"] = (
        df["days_to_completion"].notna()
        & (df["days_to_completion"] < 0)
    )

    # Detect contradictory status/date combinations
    df["completion_inconsistent"] = (
        (
            (df["referral_status"] == "Completed")
            & df["completion_date"].isna()
        )
        |
        (
            (df["referral_status"] != "Completed")
            & df["completion_date"].notna()
        )
    ).fillna(False)

    return df


# ============================================================
# 5. HOSPITAL VISIT CLEANING
# ============================================================

def clean_hospital_visits(df):
    df = df.copy()

    df["admission_date"] = pd.to_datetime(
        df["admission_date"], errors="coerce"
    )

    df["visit_category"] = (
        df["visit_category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    category_mapping = {
        "ed": "ED",
        "urgent care": "Urgent Care",
        "inpatient": "Inpatient"
    }

    df["visit_category"] = (
        df["visit_category"].map(category_mapping)
    )

    df["category_missing"] = (
        df["visit_category"].isna()
    )

    return df


# ============================================================
# 6. DATA QUALITY VALIDATION
# ============================================================

def validate_data(tables):
    primary_keys = {
        "patients": "study_patient_id",
        "appointments": "encounter_id",
        "referrals": "referral_id",
        "hospital_visits": "hospital_encounter_id"
    }

    report = []

    for name, df in tables.items():
        key = primary_keys[name]

        report.append({
            "dataset": name,
            "rows": len(df),
            "duplicate_primary_keys": (
                df[key].duplicated().sum()
            ),
            "missing_primary_keys": (
                df[key].isna().sum()
            ),
            "total_missing_values": (
                df.isna().sum().sum()
            )
        })

    patient_ids = tables["patients"]["study_patient_id"]
    encounter_ids = tables["appointments"]["encounter_id"]

    for name in [
        "appointments",
        "referrals",
        "hospital_visits"
    ]:
        unmatched = (
            ~tables[name]["study_patient_id"].isin(patient_ids)
        ).sum()

        print(f"{name}: {unmatched} unmatched patient IDs")

    unmatched_referrals = (
        ~tables["referrals"]["source_encounter_id"]
        .isin(encounter_ids)
    ).sum()

    print(
        "Referrals with unmatched encounters:",
        unmatched_referrals
    )

    return pd.DataFrame(report)


# ============================================================
# 7. RUN PIPELINE
# ============================================================

def main():
    raw = load_data()

    cleaned = {
        "patients": clean_patients(raw["patients"]),
        "appointments": clean_appointments(
            raw["appointments"]
        ),
        "referrals": clean_referrals(raw["referrals"]),
        "hospital_visits": clean_hospital_visits(
            raw["hospital_visits"]
        )
    }

    report = validate_data(cleaned)

    for name, df in cleaned.items():
        df.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False
        )

    report.to_csv(
        OUTPUT_DIR / "data_quality_report.csv",
        index=False
    )

    print("\n=== DATA QUALITY REPORT ===")
    print(report.to_string(index=False))

    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    main()
