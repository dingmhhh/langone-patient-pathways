"""
NYU Langone Capstone
Q1: Appointment No-show Logistic Regression

Input:
    data/processed/patients.csv
    data/processed/appointments.csv

Output:
    results/tables/no_show_odds_ratios.csv
    results/tables/no_show_model_summary.txt
    results/tables/no_show_sample_summary.csv
    results/figures/no_show_forest_plot.png
    results/figures/no_show_roc_curve.png

IMPORTANT:
All current data are synthetic.
Results are for pipeline testing only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)


# ============================================================
# 0. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

NYU_PURPLE = "#57068C"

# Explicit reference categories
REFERENCE_INSURANCE = "Commercial"
REFERENCE_VISIT_TYPE = "Annual Physical"
REFERENCE_VISIT_MODE = "Office"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    """Load cleaned patient and appointment data."""

    patients = pd.read_csv(
        DATA_DIR / "patients.csv"
    )

    appointments = pd.read_csv(
        DATA_DIR / "appointments.csv"
    )

    print("\n=== DATA LOADED ===")
    print(f"Patients: {len(patients):,}")
    print(f"Appointments: {len(appointments):,}")

    return patients, appointments


# ============================================================
# 2. PREPARE REGRESSION DATA
# ============================================================

def prepare_regression_data(patients, appointments):
    """
    Prepare appointment-level logistic regression data.

    Outcome:
        1 = No-show
        0 = Completed

    Canceled and unknown appointments are excluded
    from this specific model.

    Missing predictors are excluded from model fitting,
    but the original data files remain unchanged.
    """

    # Keep eligible appointment outcomes
    df = appointments[
        appointments["appointment_status"].isin(
            ["Completed", "No-show"]
        )
    ].copy()

    n_eligible = len(df)

    # Check patient identifiers
    if patients["study_patient_id"].duplicated().any():
        raise ValueError(
            "Duplicate patient IDs found in patients.csv"
        )

    if df["encounter_id"].duplicated().any():
        raise ValueError(
            "Duplicate encounter IDs found."
        )

    # Merge patient characteristics
    patient_columns = [
        "study_patient_id",
        "age",
        "insurance_type",
        "mychart_active",
        "pcp_id"
    ]

    df = df.merge(
        patients[patient_columns],
        on="study_patient_id",
        how="left",
        validate="many_to_one"
    )

    # Dependent variable
    df["is_no_show"] = (
        df["appointment_status"] == "No-show"
    ).astype(int)

    # Convert dates
    df["booking_date"] = pd.to_datetime(
        df["booking_date"],
        errors="coerce"
    )

    df["appointment_date"] = pd.to_datetime(
        df["appointment_date"],
        errors="coerce"
    )

    # Booking lead time
    df["lead_time_days"] = (
        df["appointment_date"] -
        df["booking_date"]
    ).dt.days

    # Invalid lead times are treated as missing
    df.loc[
        df["lead_time_days"] < 0,
        "lead_time_days"
    ] = np.nan

    # Age conversion and validation
    df["age"] = pd.to_numeric(
        df["age"],
        errors="coerce"
    )

    df.loc[
        ~df["age"].between(0, 120),
        "age"
    ] = np.nan

    # Interpretable regression scales
    df["age_10"] = df["age"] / 10

    df["lead_time_30"] = (
        df["lead_time_days"] / 30
    )

    # Convert MyChart status into 1 / 0 / missing
    df["mychart_active"] = (
        df["mychart_active"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0
        })
    )

    # Standardize categorical variables
    categorical_columns = [
        "insurance_type",
        "visit_type",
        "visit_mode"
    ]

    for col in categorical_columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    # Required regression variables
    model_columns = [
        "study_patient_id",
        "encounter_id",
        "is_no_show",
        "age_10",
        "lead_time_30",
        "mychart_active",
        "insurance_type",
        "visit_type",
        "visit_mode"
    ]

    # Complete-case analysis for this model only
    model_df = (
        df[model_columns]
        .dropna()
        .copy()
    )

    # Convert nullable pandas types to standard
    # numerical types for statsmodels
    model_df["age_10"] = (
        model_df["age_10"].astype(float)
    )

    model_df["lead_time_30"] = (
        model_df["lead_time_30"].astype(float)
    )

    model_df["mychart_active"] = (
        model_df["mychart_active"].astype(int)
    )

    for col in categorical_columns:
        model_df[col] = model_df[col].astype(str)

    # Ensure reference categories are present
    references = {
        "insurance_type": REFERENCE_INSURANCE,
        "visit_type": REFERENCE_VISIT_TYPE,
        "visit_mode": REFERENCE_VISIT_MODE
    }

    for col, reference in references.items():
        if reference not in model_df[col].unique():
            raise ValueError(
                f"Reference category '{reference}' "
                f"is missing from {col}"
            )

    if model_df.empty:
        raise ValueError(
            "No observations remain after preprocessing."
        )

    if model_df["is_no_show"].nunique() < 2:
        raise ValueError(
            "Both Completed and No-show observations "
            "are required."
        )

    # Save sample information
    sample_summary = pd.DataFrame([{
        "eligible_appointments": n_eligible,
        "model_appointments": len(model_df),
        "excluded_missing_predictors": (
            n_eligible - len(model_df)
        ),
        "unique_patients": (
            model_df["study_patient_id"].nunique()
        ),
        "no_show_count": int(
            model_df["is_no_show"].sum()
        ),
        "no_show_rate": (
            model_df["is_no_show"].mean()
        )
    }])

    sample_summary.to_csv(
        TABLE_DIR / "no_show_sample_summary.csv",
        index=False
    )

    print("\n=== REGRESSION SAMPLE ===")

    print(f"Eligible appointments: {n_eligible:,}")
    print(f"Model observations: {len(model_df):,}")

    print(
        "Excluded due to missing predictors:",
        f"{n_eligible - len(model_df):,}"
    )

    print(
        "Unique patients:",
        f"{model_df['study_patient_id'].nunique():,}"
    )

    print(
        "No-show rate:",
        f"{model_df['is_no_show'].mean():.2%}"
    )

    return model_df


# ============================================================
# 3. FIT LOGISTIC REGRESSION
# ============================================================

def fit_logistic_regression(model_df):
    """
    Fit multivariable logistic regression using GLM.

    Cluster-robust standard errors account for
    multiple appointments belonging to the same patient.
    """

    formula = """
        is_no_show ~
            age_10
            + lead_time_30
            + mychart_active
            + C(
                insurance_type,
                Treatment(reference="Commercial")
              )
            + C(
                visit_type,
                Treatment(reference="Annual Physical")
              )
            + C(
                visit_mode,
                Treatment(reference="Office")
              )
    """

    model = smf.glm(
        formula=formula,
        data=model_df,
        family=sm.families.Binomial()
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_df["study_patient_id"]
        }
    )

    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(result.summary())

    # Save model summary
    summary_path = (
        TABLE_DIR / "no_show_model_summary.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "NYU Langone Capstone\n"
            "Q1: No-show Logistic Regression\n"
            "SYNTHETIC DATA - TESTING ONLY\n\n"
        )
        f.write(result.summary().as_text())

    return result


# ============================================================
# 4. ODDS RATIO TABLE
# ============================================================

def create_odds_ratio_table(result):
    """
    Convert logistic regression coefficients
    into odds ratios and 95% confidence intervals.
    """

    conf = result.conf_int()

    or_table = pd.DataFrame({
        "variable": result.params.index,
        "coefficient": result.params.values,
        "odds_ratio": np.exp(result.params.values),
        "ci_lower": np.exp(conf.iloc[:, 0].values),
        "ci_upper": np.exp(conf.iloc[:, 1].values),
        "p_value": result.pvalues.values
    })

    # Remove intercept
    or_table = or_table[
        or_table["variable"] != "Intercept"
    ].copy()

    # Save full precision
    or_table.to_csv(
        TABLE_DIR / "no_show_odds_ratios.csv",
        index=False
    )

    print("\n=== ODDS RATIO TABLE ===")

    print(
        or_table[
            [
                "variable",
                "odds_ratio",
                "ci_lower",
                "ci_upper",
                "p_value"
            ]
        ].round(3).to_string(index=False)
    )

    return or_table


# ============================================================
# 5. CLEAN VARIABLE LABELS
# ============================================================

def format_variable_label(variable):
    """Create readable labels for the forest plot."""

    simple_labels = {
        "age_10": "Age (per 10 years)",
        "lead_time_30": "Booking Lead Time (per 30 days)",
        "mychart_active": "MyChart Active (vs Inactive)"
    }

    if variable in simple_labels:
        return simple_labels[variable]

    category_labels = {
        "insurance_type": "Insurance",
        "visit_type": "Visit Type",
        "visit_mode": "Visit Mode"
    }

    for field, label in category_labels.items():
        if variable.startswith(f"C({field},"):
            if "[T." in variable:
                category = variable.split("[T.", 1)[1]
                category = category.rstrip("]")

                return f"{label}: {category}"

    return variable


# ============================================================
# 6. ODDS RATIO FOREST PLOT
# ============================================================

def plot_forest(or_table):
    """
    Plot adjusted odds ratios with
    95% confidence intervals.
    """

    plot_df = or_table.copy()

    plot_df["label"] = (
        plot_df["variable"].apply(
            format_variable_label
        )
    )

    # Sort by odds ratio
    plot_df = plot_df.sort_values(
        "odds_ratio"
    ).reset_index(drop=True)

    y_positions = np.arange(len(plot_df))

    fig_height = max(
        5,
        len(plot_df) * 0.55 + 2
    )

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    # Error bars
    lower_error = (
        plot_df["odds_ratio"] -
        plot_df["ci_lower"]
    )

    upper_error = (
        plot_df["ci_upper"] -
        plot_df["odds_ratio"]
    )

    ax.errorbar(
        plot_df["odds_ratio"],
        y_positions,
        xerr=[
            lower_error,
            upper_error
        ],
        fmt="o",
        color=NYU_PURPLE,
        ecolor=NYU_PURPLE,
        markersize=6,
        capsize=4,
        linewidth=1.5
    )

    # Reference line
    ax.axvline(
        x=1,
        linestyle="--",
        color="gray",
        linewidth=1.2
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["label"])

    ax.set_xscale("log")

    # Plain-number tick labels (0.8, 0.9, 1.0 ...) instead of 8x10^-1
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    candidate_ticks = [0.25, 0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]
    x_min = plot_df["ci_lower"].min() * 0.95
    x_max = plot_df["ci_upper"].max() * 1.05
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(FixedLocator(
        [t for t in candidate_ticks if x_min <= t <= x_max]
    ))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    ax.set_xlabel(
        "Adjusted Odds Ratio (95% CI)",
        fontsize=11
    )

    ax.set_title(
        "Factors Associated with Appointment No-shows\n"
        "Synthetic Data — Model Testing Only",
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "no_show_forest_plot.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nForest plot saved: {output_path}")


# ============================================================
# 7. MODEL PERFORMANCE
# ============================================================

def evaluate_model(result, model_df):
    """
    Evaluate model discrimination using training AUROC.

    Note:
        This is an in-sample diagnostic,
        not a valid estimate of out-of-sample performance.
    """

    y_true = model_df["is_no_show"]

    y_pred = result.predict(model_df)

    auc = roc_auc_score(
        y_true,
        y_pred
    )

    print("\n=== MODEL PERFORMANCE ===")
    print(f"Training AUROC: {auc:.3f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        color=NYU_PURPLE,
        linewidth=2,
        label=f"Training AUROC = {auc:.3f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "No-show Logistic Regression — ROC Curve\n"
        "Synthetic Data — Training Performance"
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "no_show_roc_curve.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"ROC curve saved: {output_path}")

    return auc


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("NYU LANGONE CAPSTONE")
    print("Q1: NO-SHOW LOGISTIC REGRESSION")
    print("SYNTHETIC DATA ONLY")
    print("=" * 60)

    # Step 1: Load data
    patients, appointments = load_data()

    # Step 2: Prepare regression dataset
    model_df = prepare_regression_data(
        patients,
        appointments
    )

    # Step 3: Fit regression
    result = fit_logistic_regression(model_df)

    # Step 4: Generate odds ratio table
    or_table = create_odds_ratio_table(result)

    # Step 5: Plot forest
    plot_forest(or_table)

    # Step 6: Evaluate model
    auc = evaluate_model(result, model_df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"Model observations: {len(model_df):,}")
    print(f"Training AUROC: {auc:.3f}")

    print("\nResults saved in:")
    print(TABLE_DIR)
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
