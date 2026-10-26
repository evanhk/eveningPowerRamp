"""Driver script to process 2026 raw data and append to processed datasets.

This script executes four sequential stages:
1. Extend 5-minute telemetry with 2026 grid and solar geometry data.
2. Qualify duck curve ramp days in 2026 using the 20% morning solar penetration gate.
3. Extract 4D evening ramp targets for qualifying 2026 days and append to daily_ramp_targets.parquet.
4. Extract pre-noon candidate features for 2026, select the locked 22 features, and append to features_selected.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

from src.data.solar_geometry import compute_solar_geometry
from src.targets.qualify import solar_pen_filter
from src.targets.target import extract_ramp_targets

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
DATA_DIR = Path("data") if Path("data").exists() else Path("../data")
CENTROIDS = ["BOS", "ORH", "BDL", "BGR"]
ORH_LAT, ORH_LON = 42.2626, -71.8023

LOCKED_COLUMNS = [
    "date",
    "gross_demand_noon",
    "gross_demand_slope_am",
    "gross_demand_std_1h",
    "solar_cf_noon",
    "solar_cf_decay_noon",
    "solar_penetration_noon",
    "solar_kt_mean_3h",
    "solar_kt_std_3h",
    "net_load_noon",
    "total_solar_capacity_mw",
    "temp_noon_BOS",
    "temp_trend_am_BOS",
    "cloud_cover_am_ORH",
    "temp_trend_am_BDL",
    "temp_trend_am_BGR",
    "sunset_minute_offset",
    "temp_drop_afternoon_BOS",
    "temp_drop_afternoon_ORH",
    "cloud_cover_afternoon_ORH",
    "temp_drop_afternoon_BDL",
    "day_of_year_sin",
    "is_weekend",
    "delta_mw_norm",
    "s_max",
    "t_start_min",
    "delta_t_min",
]


# -----------------------------------------------------------------------------
# 1. Telemetry Harmonization for 2026
# -----------------------------------------------------------------------------
def extend_telemetry_5min() -> pd.DataFrame:
    """Harmonizes 2026 raw telemetry and merges with capacity and solar geometry."""
    print("=" * 60)
    print("Stage 1: Harmonizing 2026 Telemetry")
    print("=" * 60)

    raw_2026_path = DATA_DIR / "raw/telemetry/isone_telemetry_2026.parquet"
    if not raw_2026_path.exists():
        raise FileNotFoundError(f"Raw telemetry file not found: {raw_2026_path}")

    telem_raw = pd.read_parquet(raw_2026_path)
    print(f"Loaded raw 2026 telemetry: {len(telem_raw):,} intervals")

    cols = [
        "interval_start_local",
        "interval_end_local",
        "interval_start_utc",
        "interval_end_utc",
        "net_load",
        "load.load",
        "fuel_mix.solar",
        "fuel_mix.wind",
        "btm_solar.btm_solar",
    ]
    data = telem_raw[cols].copy()
    data = data.rename(
        columns={
            "load.load": "reported_load",
            "fuel_mix.solar": "utility_solar",
            "fuel_mix.wind": "utility_wind",
            "btm_solar.btm_solar": "btm_solar",
        }
    )
    data["gross_demand"] = data["reported_load"] + data["btm_solar"]

    # Filter to 2026 dates
    data = data[data["interval_start_local"] >= "2026-01-01"].copy()

    # Merge CELT monthly capacity
    cap_path = DATA_DIR / "processed/capacity.csv"
    cap = pd.read_csv(cap_path, index_col=0, parse_dates=True)
    cap["year"] = cap.index.year
    cap["month"] = cap.index.month

    data["year"] = data["interval_start_local"].dt.year
    data["month"] = data["interval_start_local"].dt.month
    data = data.merge(cap, on=["year", "month"], how="left").drop(columns=["year", "month"])

    data["solar_capacity_factor"] = (
        data["btm_solar"] + data["utility_solar"]
    ) / data["TOTAL_SOLAR_CAPACITY_MW"]

    # Compute solar geometry for Worcester (ORH)
    start_str = "2026-01-01"
    end_str = data["interval_start_local"].max().strftime("%Y-%m-%d")
    print(f"Computing solar geometry for {start_str} to {end_str}...")
    solar_geo = compute_solar_geometry(start_str, end_str)
    solar_geo = solar_geo[solar_geo["point_stid"] == "ORH"][
        ["valid_time_utc", "solar_zenith", "clearsky_ghi"]
    ]

    data = data.merge(
        solar_geo,
        left_on="interval_start_utc",
        right_on="valid_time_utc",
        how="left",
    ).drop(columns=["valid_time_utc"])

    # Load existing telemetry and append
    existing_telem_path = DATA_DIR / "processed/telemetry_5min.parquet"
    existing_df = pd.read_parquet(existing_telem_path)
    print(f"Existing telemetry records (2021-2025): {len(existing_df):,}")

    combined = pd.concat([existing_df, data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["interval_start_utc"]).sort_values(
        "interval_start_utc"
    ).reset_index(drop=True)

    combined.to_parquet(existing_telem_path, index=False)
    print(f"Saved updated telemetry to {existing_telem_path}: {len(combined):,} total intervals")
    print(f"Telemetry span: {combined['interval_start_local'].min()} to {combined['interval_start_local'].max()}\n")

    return combined


# -----------------------------------------------------------------------------
# 2. Qualification and Target Extraction for 2026
# -----------------------------------------------------------------------------
def qualify_and_extract_2026_targets(telem_df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Applies qualification filter and extracts ramp targets for 2026 days."""
    print("=" * 60)
    print("Stage 2: Qualifying 2026 Days & Extracting Ramp Targets")
    print("=" * 60)

    # Filter 2026 telemetry
    telem_2026 = telem_df[telem_df["interval_start_local"] >= "2026-01-01"].copy()
    end_date_str = (telem_2026["interval_start_local"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Evaluating 2026 solar penetration qualification (2026-01-01 to {end_date_str})...")
    qualifying_dates_series = solar_pen_filter(
        telem_df,
        start_date="2026-01-01",
        end_date=end_date_str,
        morning_peak_sp_thresh=0.20,
    )
    qualifying_dates = set(qualifying_dates_series)
    print(f"Identified {len(qualifying_dates)} qualifying ramp days in 2026.")

    # Extract target vectors
    telem_2026["date"] = telem_2026["interval_start_local"].dt.date
    df_qual = telem_2026[telem_2026["date"].isin(qualifying_dates)].copy()

    target_records = []
    for date_val, day_group in df_qual.groupby("date"):
        res = extract_ramp_targets(day_group)
        if res is not None:
            # Add metadata columns matching existing daily_ramp_targets schema
            solar_cap = day_group["TOTAL_SOLAR_CAPACITY_MW"].iloc[0]
            res["total_solar_capacitY_mw"] = float(solar_cap)
            res["year"] = int(date_val.year)
            target_records.append(res)

    targets_2026_df = pd.DataFrame(target_records)
    print(f"Extracted targets for {len(targets_2026_df)} of {len(qualifying_dates)} qualifying days in 2026.")

    # Load existing targets and append
    targets_path = DATA_DIR / "processed/daily_ramp_targets.parquet"
    existing_targets = pd.read_parquet(targets_path)
    existing_targets["date"] = pd.to_datetime(existing_targets["date"]).dt.date
    existing_targets = existing_targets[existing_targets["date"] < pd.to_datetime("2026-01-01").date()].copy()
    print(f"Existing targets (2021-2025): {len(existing_targets)} days")

    combined_targets = pd.concat([existing_targets, targets_2026_df], ignore_index=True)
    combined_targets = combined_targets.drop_duplicates(subset=["date"]).sort_values(
        "date"
    ).reset_index(drop=True)

    combined_targets.to_parquet(targets_path, index=False)
    print(f"Saved updated targets to {targets_path}: {len(combined_targets)} total target days\n")

    return targets_2026_df, qualifying_dates


# -----------------------------------------------------------------------------
# 3. Pre-Noon Feature Extraction Helpers
# -----------------------------------------------------------------------------
def extract_telemetry_features(df_day: pd.DataFrame) -> dict:
    """Extract pre-noon demand, solar, and net load features for a single day."""
    morning = df_day[df_day["interval_start_local"].dt.hour < 12].copy()
    if morning.empty:
        return {}

    noon_rows = df_day[df_day["interval_start_local"].dt.hour == 12]
    noon_row = noon_rows.iloc[0] if not noon_rows.empty else morning.iloc[-1]

    gross_noon = float(noon_row["gross_demand"])
    gross_peak_am = float(morning["gross_demand"].max())

    h10 = morning[morning["interval_start_local"].dt.hour >= 10]
    gross_h10 = float(h10.iloc[0]["gross_demand"]) if not h10.empty else gross_noon
    gross_slope_am = (gross_noon - gross_h10) / 2.0

    h11 = morning[morning["interval_start_local"].dt.hour >= 11]
    gross_std_1h = float(h11["gross_demand"].std()) if len(h11) > 1 else 0.0

    solar_cf_noon = float(noon_row["solar_capacity_factor"])
    h8_12 = morning[morning["interval_start_local"].dt.hour >= 8]
    solar_cf_max_am = float(h8_12["solar_capacity_factor"].max()) if not h8_12.empty else solar_cf_noon
    solar_cf_decay = float(solar_cf_noon / solar_cf_max_am) if solar_cf_max_am > 0 else 1.0

    total_solar_noon = float(noon_row["btm_solar"] + noon_row["utility_solar"])
    solar_pen_noon = float(total_solar_noon / gross_noon) if gross_noon > 0 else 0.0

    expected_clear_cf = (h8_12["clearsky_ghi"] / 1000.0).clip(lower=0.01)
    kt_series = (h8_12["solar_capacity_factor"] / expected_clear_cf).clip(0.0, 1.5)
    h9_12_mask = h8_12["interval_start_local"].dt.hour >= 9
    kt_3h = kt_series[h9_12_mask]
    solar_kt_mean_3h = float(kt_3h.mean()) if not kt_3h.empty else 0.0
    solar_kt_std_3h = float(kt_3h.std()) if len(kt_3h) > 1 else 0.0

    net_load_noon = float(noon_row["net_load"])
    net_load_trough_am = float(h8_12["net_load"].min()) if not h8_12.empty else net_load_noon
    solar_zenith_noon = float(noon_row["solar_zenith"])
    total_solar_cap = float(noon_row["TOTAL_SOLAR_CAPACITY_MW"])

    return {
        "date": df_day["date"].iloc[0],
        "gross_demand_noon": gross_noon,
        "gross_demand_peak_am": gross_peak_am,
        "gross_demand_slope_am": gross_slope_am,
        "gross_demand_std_1h": gross_std_1h,
        "solar_cf_noon": solar_cf_noon,
        "solar_cf_decay_noon": solar_cf_decay,
        "solar_penetration_noon": solar_pen_noon,
        "solar_kt_mean_3h": solar_kt_mean_3h,
        "solar_kt_std_3h": solar_kt_std_3h,
        "net_load_noon": net_load_noon,
        "net_load_trough_am": net_load_trough_am,
        "solar_zenith_noon": solar_zenith_noon,
        "total_solar_capacity_mw": total_solar_cap,
    }


def extract_observed_weather_features(df_obs_all: pd.DataFrame, target_dates: set) -> pd.DataFrame:
    """Extract observed weather metrics across all 4 centroids for 2026."""
    # Standardize timestamp column: 'datetime_utc' -> 'valid_time_local'
    df_obs = df_obs_all.copy()
    if "valid_time_local" not in df_obs.columns:
        if "datetime_utc" in df_obs.columns:
            df_obs["valid_time_local"] = pd.to_datetime(df_obs["datetime_utc"]).dt.tz_convert("US/Eastern")
        elif "date" in df_obs.columns:
            df_obs["valid_time_local"] = pd.to_datetime(df_obs["date"]).dt.tz_convert("US/Eastern")

    df_obs["date_local"] = df_obs["valid_time_local"].dt.date
    df_morning = df_obs[
        (df_obs["date_local"].isin(target_dates))
        & (df_obs["valid_time_local"].dt.hour >= 9)
        & (df_obs["valid_time_local"].dt.hour <= 12)
    ].copy()

    results = []
    for date_val, group in df_morning.groupby("date_local"):
        row_dict = {"date": date_val}
        for stid in CENTROIDS:
            st_data = group[group["point_stid"] == stid].sort_values("valid_time_local")
            if st_data.empty:
                continue

            t_noon_match = st_data[st_data["valid_time_local"].dt.hour == 12]
            t_noon = float(t_noon_match.iloc[0]["temperature_2m"]) if not t_noon_match.empty else float(st_data.iloc[-1]["temperature_2m"])
            row_dict[f"temp_noon_{stid}"] = t_noon

            t_09_match = st_data[st_data["valid_time_local"].dt.hour == 9]
            t_09 = float(t_09_match.iloc[0]["temperature_2m"]) if not t_09_match.empty else t_noon
            row_dict[f"temp_trend_am_{stid}"] = t_noon - t_09

            row_dict[f"cloud_cover_am_{stid}"] = float(st_data["cloud_cover"].mean())

        results.append(row_dict)
    return pd.DataFrame(results)


def extract_forecast_features(df_hrrr_all: pd.DataFrame, sunset_map: dict) -> pd.DataFrame:
    """Extract HRRR forecast features for all 4 centroids for 2026."""
    results = []
    for date_val, group in df_hrrr_all.groupby("forecast_date"):
        if date_val not in sunset_map:
            continue
        sunset_time = sunset_map[date_val]
        sunset_hour = sunset_time.hour

        noon_local = pd.Timestamp(f"{date_val} 12:00", tz="US/Eastern")
        sunset_min_offset = (sunset_time - noon_local).total_seconds() / 60.0

        row_dict = {
            "date": date_val,
            "sunset_minute_offset": sunset_min_offset,
        }

        for stid in CENTROIDS:
            st_fcst = group[group["point_stid"] == stid].sort_values("lead_hour")
            if st_fcst.empty:
                continue

            t_sunset_match = st_fcst[st_fcst["valid_time_local"].dt.hour == sunset_hour]
            t_sunset = float(t_sunset_match.iloc[0]["tmp_2m_c"]) if not t_sunset_match.empty else float(st_fcst.iloc[-1]["tmp_2m_c"])
            t_noon_fcst = float(st_fcst.iloc[0]["tmp_2m_c"])
            row_dict[f"temp_drop_afternoon_{stid}"] = t_sunset - t_noon_fcst

            row_dict[f"cloud_cover_afternoon_{stid}"] = float(st_fcst["tcdc"].mean())

            sunset_window = st_fcst[
                (st_fcst["valid_time_local"].dt.hour >= sunset_hour - 1)
                & (st_fcst["valid_time_local"].dt.hour <= sunset_hour + 1)
            ]
            row_dict[f"cloud_cover_sunset_{stid}"] = float(sunset_window["tcdc"].mean()) if not sunset_window.empty else float(st_fcst.iloc[-1]["tcdc"])

        results.append(row_dict)
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 4. Feature Assembly and Parquet Append
# -----------------------------------------------------------------------------
def append_2026_features(
    telem_df: pd.DataFrame,
    qualifying_dates: set,
    targets_2026_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extracts features for 2026, selects the 22 locked features, and appends to parquet."""
    print("=" * 60)
    print("Stage 3: Extracting 2026 Features & Appending to Table")
    print("=" * 60)

    # Restrict target dates to those where both qualification and target extraction succeeded
    target_dates = set(targets_2026_df["date"])
    print(f"Extracting features for {len(target_dates)} validated 2026 days...")

    # 1. Telemetry Features
    telem_df_qual = telem_df[telem_df["interval_start_local"].dt.date.isin(target_dates)].copy()
    telem_df_qual["date"] = telem_df_qual["interval_start_local"].dt.date

    telem_rows = []
    for date_val, g in telem_df_qual.groupby("date"):
        telem_rows.append(extract_telemetry_features(g))
    telem_feat_df = pd.DataFrame(telem_rows)
    print(f"Extracted telemetry features: {len(telem_feat_df)} days")

    # 2. Observed Weather Features
    obs_path = DATA_DIR / "raw/weather_observed/weather_observations_2026.parquet"
    obs_raw = pd.read_parquet(obs_path)
    obs_feat_df = extract_observed_weather_features(obs_raw, target_dates)
    print(f"Extracted observed weather features: {len(obs_feat_df)} days")

    # 3. Forecast Weather Features
    hrrr_path = DATA_DIR / "raw/weather_forecast/hrrr_forecasts_2026.parquet"
    hrrr_raw = pd.read_parquet(hrrr_path)
    hrrr_raw["forecast_date"] = pd.to_datetime(hrrr_raw["forecast_date"]).dt.date
    if "tmp_2m" in hrrr_raw.columns:
        hrrr_raw["tmp_2m_c"] = hrrr_raw["tmp_2m"].astype(np.float32) - 273.15

    # Compute astronomical sunset for each date at Worcester (ORH)
    dates_dt = pd.to_datetime(list(target_dates))
    sun_times = pvlib.solarposition.sun_rise_set_transit_spa(
        dates_dt.tz_localize("US/Eastern"),
        ORH_LAT,
        ORH_LON,
    )
    sunset_dict = dict(zip(dates_dt.date, sun_times["sunset"].dt.tz_convert("US/Eastern")))
    fcst_feat_df = extract_forecast_features(hrrr_raw, sunset_dict)
    print(f"Extracted HRRR forecast features: {len(fcst_feat_df)} days")

    # Merge feature blocks
    merged_df = telem_feat_df.merge(obs_feat_df, on="date", how="inner")
    merged_df = merged_df.merge(fcst_feat_df, on="date", how="inner")

    # Add Calendar Baselines
    date_series = pd.to_datetime(merged_df["date"])
    doy = date_series.dt.dayofyear
    merged_df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    merged_df["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(int)

    # Merge target columns
    target_cols = ["date", "t_start_min", "delta_mw_norm", "delta_t_min", "s_max"]
    merged_df = merged_df.merge(targets_2026_df[target_cols], on="date", how="inner")

    # Subset to the exact locked 27 columns
    candidate_2026 = merged_df[LOCKED_COLUMNS].copy()
    print(f"Prepared 2026 features table: {len(candidate_2026)} days with {len(LOCKED_COLUMNS)} columns")

    # Load existing locked features table and append
    feat_selected_path = DATA_DIR / "processed/features_selected.parquet"
    existing_feats = pd.read_parquet(feat_selected_path)
    existing_feats["date"] = pd.to_datetime(existing_feats["date"]).dt.date
    existing_feats = existing_feats[existing_feats["date"] < pd.to_datetime("2026-01-01").date()].copy()
    print(f"Existing feature records (2021-2025): {len(existing_feats)} days")

    combined_feats = pd.concat([existing_feats, candidate_2026], ignore_index=True)
    combined_feats = combined_feats.drop_duplicates(subset=["date"]).sort_values(
        "date"
    ).reset_index(drop=True)

    combined_feats.to_parquet(feat_selected_path, index=False)
    print(f"Saved updated features to {feat_selected_path}: {len(combined_feats)} total days\n")

    return combined_feats


# -----------------------------------------------------------------------------
# Main Orchestration
# -----------------------------------------------------------------------------
def main():
    print("Starting 2026 Data Extension Pipeline...\n")
    telem_df = extend_telemetry_5min()
    targets_2026_df, qualifying_dates = qualify_and_extract_2026_targets(telem_df)
    features_df = append_2026_features(telem_df, qualifying_dates, targets_2026_df)

    n_train = (pd.to_datetime(features_df["date"]).dt.year < 2026).sum()
    n_test = (pd.to_datetime(features_df["date"]).dt.year == 2026).sum()

    print("=" * 60)
    print("Pipeline Complete! Ready for Model Testing.")
    print(f"  Training samples (2021-2025): {n_train:,} days")
    print(f"  Test samples (2026 holdout):  {n_test:,} days")
    print("=" * 60)


if __name__ == "__main__":
    main()
