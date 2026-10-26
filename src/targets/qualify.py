import pandas as pd


def solar_pen_filter(telem_data, start_date='2021-01-01', end_date='2026-01-01', morning_peak_sp_thresh=0.20):
  """Filters for days where morning peak solar penetration exceeds threshold.

  Returns qualifying dates (or daily summary table).
  """
  data = telem_data

  data = data[data["interval_start_local"] >= pd.to_datetime(start_date).tz_localize('US/Eastern')]
  data = data[data["interval_start_local"] < pd.to_datetime(end_date).tz_localize('US/Eastern')]

  data = data.copy()
  data["total_solar"] = data["btm_solar"] + data["utility_solar"]
  expected_clear_cf = data["clearsky_ghi"] / 1000.0
  data["kt"] = (data["solar_capacity_factor"] / expected_clear_cf).clip(0, 1.5)
  data["solar_penetration"] = data["total_solar"] / data["gross_demand"]

  # Morning evaluation window: 10:00 AM to 12:00 PM Eastern
  morning_mask = (data["interval_start_local"].dt.hour >= 10) & (
      data["interval_start_local"].dt.hour < 12
  )
  morning_df = data[morning_mask].copy()
  morning_df["date"] = morning_df["interval_start_local"].dt.date

  daily_summary = (
      morning_df.groupby("date")
      .agg(
          morning_peak_mw=("total_solar", "max"),
          morning_peak_kt=("kt", "max"),
          morning_peak_cf=("solar_capacity_factor", "max"),
          morning_peak_sp=("solar_penetration", "max"),
      )
      .reset_index()
  )

  daily_summary["qualifies"] = (daily_summary["morning_peak_sp"] >= morning_peak_sp_thresh )

  print("Total days in window:", len(daily_summary))
  print(f"Qualifying days: {daily_summary['qualifies'].sum()} ({daily_summary['qualifies'].mean():.1%})" )

  return daily_summary[daily_summary["qualifies"]]["date"].reset_index(drop=True)
