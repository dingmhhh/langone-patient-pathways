import pandas as pd

def calculate_no_show_rate(df):
    """
    Calculate no-show rate among completed
    and no-show appointments.
    """
    eligible = df[
        df["appointment_status"].isin(
            ["Completed", "No-show"]
        )
    ].copy()

    if eligible.empty:
        return float("nan")

    return (
        eligible["appointment_status"]
        .eq("No-show")
        .mean()
    )

def calculate_referral_completion_rate(df):
    """
    Calculate observed referral completion rate.
    """
    if df.empty:
        return float("nan")

    return (
        df["referral_status"]
        .eq("Completed")
        .mean()
    )