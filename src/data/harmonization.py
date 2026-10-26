import src.data.solar_geometry as sg
import pandas as pd

#ORH_CENTROID = [42.2626,  -71.8023 ]
def harmonize():
    telem = pd.read_parquet('data/raw/telemetry/')
    cap = pd.read_csv('data/processed/capacity.csv', index_col=0, parse_dates=True)

    data = telem[['interval_start_local','interval_end_local','interval_start_utc','interval_end_utc','net_load','load.load','fuel_mix.solar','fuel_mix.wind','btm_solar.btm_solar']]

    data = data.rename(columns = {'load.load':'reported_load',
                'fuel_mix.solar':'utility_solar',
                'fuel_mix.wind':'utility_wind',
                'btm_solar.btm_solar':'btm_solar'})

    data['gross_demand'] = data['reported_load']  + data['btm_solar']

    data = data[(data['interval_start_local'] >= '2021-01-01') & (data['interval_end_local'] < '2026-01-01')]

    cap['year'] =  cap.index.year
    cap['month'] = cap.index.month

    data['year'] = data['interval_start_local'].dt.year
    data['month'] = data['interval_start_local'].dt.month
    data = data.merge(cap, on = ['year','month'],how = 'left').drop(columns = ['year','month'])

    data['solar_capacity_factor'] = (data['btm_solar'] + data['utility_solar']) / data['TOTAL_SOLAR_CAPACITY_MW']


    solar_geo = sg.compute_solar_geometry('2021-01-01','2026-01-01')
    solar_geo = solar_geo[solar_geo['point_stid'] == 'ORH'][['valid_time_utc','solar_zenith','clearsky_ghi']]

    data = data.merge(
        solar_geo,
        left_on='interval_start_utc',
        right_on = 'valid_time_utc',
        how = 'left'
    ).drop(columns = ['valid_time_utc'])

    data.to_parquet('data/processed/telemetry_5min.parquet')

    return data