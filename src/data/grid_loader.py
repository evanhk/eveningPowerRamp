from gridstatusio import GridStatusClient
import pandas as pd
from pathlib import Path
#import gridstatus
from dotenv import load_dotenv
from gridstatus.isone_api import isone_api


load_dotenv()

def fetch_telemetry(start_date,end_date):
    data_path = Path("data/raw/telemetry/")

    start_dt = pd.to_datetime(start_date).tz_localize("US/Eastern")
    end_dt = pd.to_datetime(end_date).tz_localize("US/Eastern")

    #figure out what years are needed for request
    years_to_fetch = range(start_dt.year,end_dt.year+1)

    for year in years_to_fetch:
        print(f"fetching data for {year}...") 
        file_path = data_path / f"isone_telemetry_{year}.parquet"

        #Checks if file is on disk, if not then fetch from API
        if not file_path.exists():
            print(f"Downloading {year} Telemetry from ISO-NE API")
            year_start = f"{year}-01-01"
            year_end = f"{year+1}-01-01"
            
            client = GridStatusClient()
            # Fetch data as pandas DataFrame
            telem = client.get_dataset(
            dataset="isone_standardized_5_min",
            start=year_start,
            end=year_end,
            timezone="market",
            )
            
            # Convert all numeric/null telemetry columns to float64
            for col in telem.columns:
                if col != "primary_power_source" and not pd.api.types.is_datetime64_any_dtype(telem[col]):
                    telem[col] = pd.to_numeric(telem[col], errors="coerce")
            telem.to_parquet(file_path)


    #lets start with reading all Demand files and filtering after. If this proves to be inefficient
    #I'll switch to targeting only the necessary files
    request_df = pd.read_parquet(data_path)

    incl_rows = (request_df["interval_start_local"] >= start_dt) & (request_df["interval_end_local"] <= end_dt)
    request_df = request_df[incl_rows]

    request_df = request_df.sort_values("interval_start_local").reset_index(drop=True)


    return request_df

"""
Less efficient then using fetch_telemetry. Dont use

def fetch_demand(start_date,end_date):
    data_path = Path("data/raw/demand/")
    iso = isone_api.ISONEAPI()

    start_dt = pd.to_datetime(start_date).tz_localize("US/Eastern")
    end_dt = pd.to_datetime(end_date).tz_localize("US/Eastern")

    #figure out what years are needed for request
    years_to_fetch = range(start_dt.year,end_dt.year+1)
    saved_files = data_path.iterdir()
    for year in years_to_fetch:
        print(f"fetching data for {year}...") 
        file_path = data_path / f"isone_demand_{year}.parquet"

        #Checks if file is on disk, if not then fetch from API
        if not file_path.exists():
            print(f"Downloading {year} from ISO-NE API")
            year_start = f"01-01-{year}"
            year_end = f"12-31-{year}"
            
            demand = iso.get_total_demand(year_start, year_end)
            _demand_df_check(demand)

            demand.to_parquet(file_path)

    #lets start with reading all Demand files and filtering after. If this proves to be inefficient
    #I'll switch to targeting only the necessary files
    request_df = pd.read_parquet(data_path)

    incl_rows = (request_df["Interval Start"] >= start_dt) & (request_df["Interval End"] <= end_dt)
    request_df = request_df[incl_rows]

    return request_df
"""

def _demand_df_check(demand_df):
    if type(demand_df) != pd.DataFrame:
        raise ValueError("Demand Dataframe is not an actual DataFrame object")
    if len(demand_df) < 105000:
        raise ValueError(f"Demand DataFrame only has {len(demand_df)} rows")

    expected_columns = pd.Index(['Interval Start', 'Interval End', 'Total Load', 'Native Load',
       'Storage Load', 'Total Load With Estimated Solar',
       'Native Load With Estimated Solar'],
      dtype='object')
    if not demand_df.columns.equals(expected_columns):
        raise ValueError("Demand DataFrame has an unexpected schema")
    
