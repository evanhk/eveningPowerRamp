import pandas as pd
import numpy as np
import pvlib as pv

# Station geographical coordinates (for Herbie GRIB pick_points, and other uses)
STATION_POINTS = pd.DataFrame({
    "latitude": [42.3601, 42.2626, 41.7658, 44.8016],
    "longitude": [-71.0589, -71.8023, -72.6734, -68.7712],
    "stid": ["BOS", "ORH", "BDL", "BGR"],
})
def compute_solar_geometry(start_date:pd.Timestamp,end_date:str|pd.Timestamp, stations = STATION_POINTS) -> pd.DataFrame:
    """calculates apparent zenith, azimuth, theoretical GHI, theoretical DNI, theorteical DHI in 5 minute increments for requested dates"""
    
    end_date = pd.Timestamp(end_date)
    times = pd.date_range(start_date,end_date.replace(hour=23,minute =55),freq="5min",tz="UTC")

    records = []
    
    for _, row in stations.iterrows():
        stid = row["stid"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        loc = pv.location.Location(latitude=lat, longitude=lon, tz="UTC")
        solpos = loc.get_solarposition(times)
        clearsky = loc.get_clearsky(times)
        
        station_df = pd.DataFrame({
            "valid_time_utc": times,
            "point_stid": stid,
            "solar_zenith": solpos["apparent_zenith"].values,
            "solar_azimuth": solpos["azimuth"].values,
            "clearsky_ghi": clearsky["ghi"].values,
            "clearsky_dni": clearsky["dni"].values,
            "clearsky_dhi": clearsky["dhi"].values,
        })
        records.append(station_df)
        
    return pd.concat(records, ignore_index=True)


def compute_solar_landmarks(start_date ,end_date , stations: pd.DataFrame) -> pd.DataFrame:
    """calculates sunrise, transit (solar noon), sunset for requested dates and stations/points"""
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    dates = pd.date_range(start_date,end_date,freq="D", tz = "UTC")
    records = []
    
    for _, row in stations.iterrows():
        stid = row["stid"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        loc = pv.location.Location(latitude=lat, longitude=lon, tz="UTC")
        soltime = loc.get_sun_rise_set_transit(dates)
        
        
        station_df = pd.DataFrame({
            "date": dates,
            "point_stid": stid,
            "sunrise": soltime["sunrise"].reset_index(drop=True),
            "transit": soltime["transit"].reset_index(drop=True),
            "sunset": soltime["sunset"].reset_index(drop=True)
        })
           
        records.append(station_df)
        
    return pd.concat(records, ignore_index=True)
