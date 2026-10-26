import openmeteo_requests
import requests_cache
from retry_requests import retry
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import time
import warnings
import fsspec
from herbie import Herbie
import numpy as np
import pandas as pd
import xarray as xr
import herbie
from datetime import date, timedelta

#  Opt in to xarray defaults (silences cfgrib combine warnings)
herbie.accessors.xr.set_options(use_new_combine_kwarg_defaults=True)
#  Suppress remaining non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="This pattern is interpreted as a regular expression",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Will not remove GRIB file",
)

# Station pixel coordinates on the 1059 x 1799 HRRR grid
CENTROIDS = {
    "BOS": {"y": 776, "x": 1615},
    "ORH": {"y": 767, "x": 1596},
    "BDL": {"y": 742, "x": 1578},
    "BGR": {"y": 881, "x": 1647},
}
# Station geographical coordinates (for Herbie GRIB pick_points, and other uses)
STATION_POINTS = pd.DataFrame({
    "latitude": [42.3601, 42.2626, 41.7658, 44.8016],
    "longitude": [-71.0589, -71.8023, -72.6734, -68.7712],
    "stid": ["BOS", "ORH", "BDL", "BGR"],
})

# Mapping of model variables to S3 Zarr store subpaths
ZARR_VARIABLES = {
    "dswrf": "surface/DSWRF/surface",
    "vbdsf": "surface/VBDSF/surface",
    "tmp_2m": "2m_above_ground/TMP/2m_above_ground",
    "dpt_2m": "2m_above_ground/DPT/2m_above_ground",
    "ugrd_10m": "10m_above_ground/UGRD/10m_above_ground",
    "vgrd_10m": "10m_above_ground/VGRD/10m_above_ground",
    "gust": "surface/GUST/surface",
    "tcdc": "entire_atmosphere/TCDC/entire_atmosphere",
    "lcdc": "low_cloud_layer/LCDC/low_cloud_layer",
    "mcdc": "middle_cloud_layer/MCDC/middle_cloud_layer",
    "hcdc": "high_cloud_layer/HCDC/high_cloud_layer",
}

def fetch_yearly_weather(start_year :int, end_year :int,points:pd.DataFrame) -> pd.DataFrame:
	""" Fetches hourly observed weather from Open-Meteo API for years specified. [inclusive,exclusive)
		points should be a dataframe with cols ["latitutde","longitude","stid"]
		"""

	##Temperature in Celsius, Windspeed in m/s

	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)
    
    
	end_date = min(f"{end_year - 1}-12-31", (date.today() - timedelta(days=1)).isoformat())
        
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": points["latitude"],
		"longitude": points["longitude"],
		"start_date": f"{start_year}-01-01",
		"end_date": f"{end_date}",
		"hourly": ["temperature_2m", "dew_point_2m", "cloud_cover", "cloud_cover_mid", "cloud_cover_high", "cloud_cover_low", "wind_speed_10m", 
					"wind_direction_10m", "wind_gusts_10m", "shortwave_radiation", "direct_radiation"],
		"wind_speed_unit": "ms",
	}
	responses = openmeteo.weather_api(url, params = params)

	return_df = []
	# Process 4 locations
	for i,response in enumerate(responses):
		print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
		print(f"Elevation: {response.Elevation()} m asl")
		print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
		
		# Process hourly data. The order of variables needs to be the same as requested.
		hourly = response.Hourly()
		hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
		hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
		hourly_cloud_cover = hourly.Variables(2).ValuesAsNumpy()
		hourly_cloud_cover_mid = hourly.Variables(3).ValuesAsNumpy()
		hourly_cloud_cover_high = hourly.Variables(4).ValuesAsNumpy()
		hourly_cloud_cover_low = hourly.Variables(5).ValuesAsNumpy()
		hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
		hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()
		hourly_wind_gusts_10m = hourly.Variables(8).ValuesAsNumpy()
		hourly_shortwave_radiation = hourly.Variables(9).ValuesAsNumpy()
		hourly_direct_radiation = hourly.Variables(10).ValuesAsNumpy()
		

		hourly_data = {
			"datetime_utc": pd.date_range(
				start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
				end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
				freq = pd.Timedelta(seconds = hourly.Interval()),
				inclusive = "left"
			)
		}
		hourly_data["point_stid"] = points.iloc[i]["stid"]
		
		hourly_data["temperature_2m"] = hourly_temperature_2m
		hourly_data["dew_point_2m"] = hourly_dew_point_2m
		hourly_data["cloud_cover"] = hourly_cloud_cover
		hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
		hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
		hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
		hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
		hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
		hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
		hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
		hourly_data["direct_radiation"] = hourly_direct_radiation
		
		hourly_dataframe = pd.DataFrame(data = hourly_data)

		return_df.append(hourly_dataframe)
	return_df = pd.concat(return_df,ignore_index=True)
	return return_df			
	

def get_noon_utc(date_str: str) -> pd.Timestamp:
    """Convert YYYY-MM-DD local noon (12:00 US/Eastern) to UTC."""
    local_dt = pd.Timestamp(f"{date_str} 12:00").tz_localize("US/Eastern")
    return local_dt.tz_convert("UTC")


def fetch_single_day_zarr(date_str: str) -> pd.DataFrame:
    """Fetch 12pm forecast f01-f09 for all centroids and variables for one date."""
    run_utc = get_noon_utc(date_str)
    date_path = run_utc.strftime("%Y%m%d")
    run_str = f"{date_path}_{run_utc.hour:02d}z_fcst.zarr"

    y_idx = [c["y"] for c in CENTROIDS.values()]
    x_idx = [c["x"] for c in CENTROIDS.values()]
    stids = list(CENTROIDS.keys())

    # Build base index: 9 lead hours x 4 stations = 36 rows
    records = []
    for lead in range(1, 10):
        valid_utc = run_utc + pd.Timedelta(hours=lead)
        valid_local = valid_utc.tz_convert("US/Eastern")
        for stid in stids:
            records.append(
                {
                    "forecast_date": date_str,
                    "init_time_utc": run_utc,
                    "lead_hour": lead,
                    "valid_time_utc": valid_utc,
                    "valid_time_local": valid_local,
                    "point_stid": stid,
                }
            )
    df_day = pd.DataFrame(records)

    # Extract each variable array and assign as column
    for col_name, subpath in ZARR_VARIABLES.items():
        url = f"s3://hrrrzarr/sfc/{date_path}/{run_str}/{subpath}"
        mapper = fsspec.get_mapper(url, anon=True)
        ds = xr.open_zarr(mapper)
        var_key = list(ds.data_vars)[0]

        # Slice f01-f09 (time 0:9) and 4 station coordinates
        # Output shape: (9 lead hours, 4 stations)
        vals = (
            ds[var_key]
            .isel(
                time=slice(0, 9),
                projection_y_coordinate=xr.DataArray(y_idx, dims="point"),
                projection_x_coordinate=xr.DataArray(x_idx, dims="point"),
            )
            .values
        )

        df_day[col_name] = vals.flatten().astype(np.float32)

    return df_day


def download_historical_forecasts(
    start_year: int = 2021,
    end_year: int = 2025,
    output_dir: str = "data/raw/weather_forecast",
):
    """Download daily 12pm HRRR forecasts with annual Parquet checkpoints.
        WARNING: this process can take upward of 45 minutes per year"""
    os.makedirs(output_dir, exist_ok=True)

    for year in range(start_year, end_year + 1):
        parquet_path = os.path.join(output_dir, f"hrrr_forecasts_{year}.parquet")
        existing_df = None
        done_dates = set()

        if os.path.exists(parquet_path):
            existing_df = pd.read_parquet(parquet_path)
            done_dates = set(existing_df["forecast_date"].unique())
            print(
                f"[{year}] Found existing checkpoint with {len(done_dates)} days."
            )

        # Generate full date range for the year
        all_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        dates_to_pull = [
            d.strftime("%Y-%m-%d")
            for d in all_dates
            if d.strftime("%Y-%m-%d") not in done_dates
        ]

        if not dates_to_pull:
            print(f"[{year}] All dates already downloaded. Skipping year.")
            continue

        print(
            f"[{year}] Downloading {len(dates_to_pull)} remaining days out of {len(all_dates)}..."
        )

        year_buffer = [] if existing_df is None else [existing_df]
        unsaved_count = 0

        for i, date_str in enumerate(dates_to_pull, 1):
            try:
                df_single = fetch_single_day_zarr(date_str)
                year_buffer.append(df_single)
                unsaved_count += 1
                print(f"  [{i}/{len(dates_to_pull)}] Processed {date_str}")
            except Exception as e:
                print(f"  [{i}/{len(dates_to_pull)}] Failed {date_str}: {e}")

            # Checkpoint flush every 30 days
            if unsaved_count >= 30:
                checkpoint_df = pd.concat(year_buffer, ignore_index=True)
                checkpoint_df.to_parquet(parquet_path, index=False)
                year_buffer = [checkpoint_df]
                unsaved_count = 0
                print(f"  --> Checkpoint saved ({len(checkpoint_df)} rows)")

        # Final save for the year
        if year_buffer:
            final_df = pd.concat(year_buffer, ignore_index=True)
            final_df.to_parquet(parquet_path, index=False)
            print(
                f"[{year}] Finished. Total rows: {len(final_df)} -> {parquet_path}\n"
            )


# Suppress regex capture group warning from Herbie
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="This pattern is interpreted as a regular expression",
)


# ... existing get_noon_utc, fetch_single_day_zarr, download_historical_forecasts ...
def _fetch_single_hour_herbie(
    noon_utc: pd.Timestamp,
    date_str: str,
    fxx: int,
    points_df: pd.DataFrame,
    max_retries: int = 3,
) -> pd.DataFrame:
  """Worker: Fetch DSWRF & VBDSF for 1 forecast hour with retry logic."""
  for attempt in range(max_retries):
    try:
      H = Herbie(
          noon_utc, model="hrrr", product="sfc", fxx=fxx, verbose=False
      )
      ds = H.xarray(":(?:DSWRF|VBDSF):surface:", overwrite=True)
      df = ds.herbie.pick_points(points_df, tree_name="hrrr").to_dataframe()
      df = df.rename(columns={"sdswrf": "dswrf"})
      df["forecast_date"] = date_str
      df["lead_hour"] = fxx
      df["init_time_utc"] = noon_utc
      return df[[
          "forecast_date",
          "init_time_utc",
          "lead_hour",
          "valid_time",
          "point_stid",
          "dswrf",
          "vbdsf",
      ]]
    except Exception as e:
      if attempt == max_retries - 1:
        print(
            f"  [ERROR] {date_str} f{fxx:02d} failed after {max_retries}"
            f" attempts: {e}"
        )
        return pd.DataFrame()
      time.sleep(1.0)
  return pd.DataFrame()
def fetch_single_day_solar(
    date_str: str,
    points_df: pd.DataFrame = STATION_POINTS,
    max_workers: int = 9,
) -> pd.DataFrame:
  """Fetch all 9 forecast hours for one date concurrently."""
  noon_utc = get_noon_utc(date_str).tz_localize(None)
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(
            _fetch_single_hour_herbie, noon_utc, date_str, fxx, points_df
        )
        for fxx in range(1, 10)
    ]
    results = [f.result() for f in futures if not f.result().empty]
  if not results:
    return pd.DataFrame()
  return pd.concat(results, ignore_index=True)
def _update_and_save_solar_checkpoint(
    existing_df: pd.DataFrame, solar_buffer: list, parquet_path: str
) -> pd.DataFrame:
  """Helper: Merge accumulated solar records into existing parquet dataframe."""
  if not solar_buffer:
    return existing_df
  solar_df = pd.concat(solar_buffer, ignore_index=True)
  solar_df["dswrf"] = solar_df["dswrf"].astype(np.float32)
  solar_df["vbdsf"] = solar_df["vbdsf"].astype(np.float32)
  keys = ["forecast_date", "lead_hour", "point_stid"]
  df_indexed = existing_df.set_index(keys)
  solar_indexed = solar_df.set_index(keys)
  # Update only dswrf and vbdsf in-place
  df_indexed.update(solar_indexed[["dswrf", "vbdsf"]])
  updated_df = df_indexed.reset_index()
  updated_df["dswrf"] = updated_df["dswrf"].astype(np.float32)
  updated_df["vbdsf"] = updated_df["vbdsf"].astype(np.float32)
  updated_df.to_parquet(parquet_path, index=False)
  return updated_df

def backfill_historical_solar(
    start_year: int = 2021,
    end_year: int = 2024,
    output_dir: str = "data/raw/weather_forecast",
    checkpoint_days: int = 90,
    max_workers: int = 9,
):
  """Backfill missing dswrf and vbdsf solar data via Herbie with 90-day checkpoints."""
  for year in range(start_year, end_year + 1):
    parquet_path = os.path.join(output_dir, f"hrrr_forecasts_{year}.parquet")
    if not os.path.exists(parquet_path):
      print(
          f"[{year}] No checkpoint file found at {parquet_path}. Run"
          " download_historical_forecasts first."
      )
      continue
    df = pd.read_parquet(parquet_path)
    # Auto-resume: identify only dates with missing solar data
    missing_dates = sorted(df.loc[df["dswrf"].isna(), "forecast_date"].unique())
    if not missing_dates:
      print(f"[{year}] All solar data is already populated. Skipping year.")
      continue
    print(
        f"[{year}] Found {len(missing_dates)} dates needing solar backfill (out"
        f" of {df['forecast_date'].nunique()})."
    )

    #Warm up eccodes definitions in the main thread once
    print("  Initializing GRIB definitions...")
    _ = _fetch_single_hour_herbie(
        get_noon_utc(missing_dates[0]).tz_localize(None),
        missing_dates[0],
        1,
        STATION_POINTS,
    )

    solar_buffer = []
    days_since_checkpoint = 0
    t_start_year = time.time()
    for i, date_str in enumerate(missing_dates, 1):
      t_day = time.time()
      day_df = fetch_single_day_solar(
          date_str, points_df=STATION_POINTS, max_workers=max_workers
      )
      if not day_df.empty:
        solar_buffer.append(day_df)
        days_since_checkpoint += 1
        print(
            f"  [{i}/{len(missing_dates)}] {date_str} fetched in"
            f" {time.time() - t_day:.1f}s"
        )
      else:
        print(f"  [{i}/{len(missing_dates)}] {date_str} warning: no data")
      # Checkpoint flush every checkpoint_days
      if days_since_checkpoint >= checkpoint_days:
        df = _update_and_save_solar_checkpoint(df, solar_buffer, parquet_path)
        solar_buffer = []
        days_since_checkpoint = 0
        print(f"  --> Checkpoint saved to {parquet_path}")
    # Final save for remaining dates in this year
    if solar_buffer:
      df = _update_and_save_solar_checkpoint(df, solar_buffer, parquet_path)
      print(
          f"[{year}] Complete in {(time.time() - t_start_year)/60:.1f} min."
          f" Saved to {parquet_path}\n"
      )
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="HRRR Weather Data Loader")
  parser.add_argument(
      "--backfill-solar",
      action="store_true",
      help="Backfill missing solar data via Herbie",
  )
  parser.add_argument(
      "--start-year",
      type=int,
      default=2021,
      help="Start year (default: 2021)",
  )
  parser.add_argument(
      "--end-year",
      type=int,
      default=2024,
      help="End year (default: 2024)",
  )
  args = parser.parse_args()
  if args.backfill_solar:
    backfill_historical_solar(
        start_year=args.start_year, end_year=args.end_year
    )
  else:
    download_historical_forecasts(
        start_year=args.start_year, end_year=args.end_year
    )


