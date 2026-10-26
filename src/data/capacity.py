from matplotlib.dates import MonthLocator
import pandas as pd


def interpolate_solar_capacity():
    """This process reads solar nameplate capacity from data/raw/ and resamples and interpolates to the monthly level
        It then both returns and writes the dataframe to /data/processed/capacity.csv
    """
    
    celt = pd.read_csv("data/raw/solar_capacity.csv", parse_dates= ['DATE'], index_col="DATE")
    growth_2025 = celt.loc['2025-12-31'] - celt.loc['2024-12-31']

    celt.loc[pd.Timestamp('2026-12-31')] = celt.loc['2025-12-31'] + growth_2025
    monthly_ind = pd.date_range('2020-12-31','2026-12-31', freq = 'ME')

    monthly_df = celt.reindex(monthly_ind)
    monthly_df = monthly_df.interpolate(method = "time")

    monthly_df["TOTAL_SOLAR_CAPACITY_MW"] = monthly_df["CELT_BTM_CAPACITY_MW"] + monthly_df["CELT_UTILITY_CAPACITY_MW"]

    monthly_df.to_csv("data/processed/capacity.csv")

    return monthly_df