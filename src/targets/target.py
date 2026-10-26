import numpy as np
import scipy.signal as signal
import pandas as pd


def extract_ramp_targets(day_data):
    """Apply Savitzky-Golay smoothing and then extract ramp start, duration, amplitude, peak_change (s_max)"""    
    # Filter afternoon search window (12:00 PM up to 10:00 PM)
    afternoon = day_data[
        (day_data["interval_start_local"].dt.hour >= 12) & (day_data["interval_start_local"].dt.hour <= 22)
    ].copy()
    
    # Interpolate isolated missing telemetry intervals (e.g. 1-2 intervals)
    afternoon["net_load"] = afternoon["net_load"].interpolate(method="linear").bfill().ffill()
    if afternoon["net_load"].isna().any():
        return None

    # Savitzky-Golay smoothing (35-min window = 7 intervals)
    afternoon["net_load_smooth"] = signal.savgol_filter(afternoon["net_load"].values, window_length=7, polyorder=2)
    y = afternoon["net_load_smooth"].values
    times = afternoon["interval_start_local"].values
    
    # Identify solar peak and 10% decay threshold
    peak_solar_val = day_data["solar_capacity_factor"].max()
    peak_solar_idx = day_data["solar_capacity_factor"].idxmax()
    post_peak_data = day_data.loc[peak_solar_idx:]
    decay_candidates = post_peak_data[
        post_peak_data["solar_capacity_factor"] <= 0.90 * peak_solar_val
    ]
    if not decay_candidates.empty:
        t_decay = decay_candidates["interval_start_local"].min()
    else:
        t_decay = day_data.loc[peak_solar_idx, "interval_start_local"]

    # Identify Ramp Peak (t_peak)
    peak_idx = int(np.argmax(y))
    t_peak = afternoon["interval_start_local"].iloc[peak_idx]

    # Find index in afternoon corresponding to t_decay
    decay_idx = 0
    for idx, t in enumerate(afternoon["interval_start_local"]):
        if t >= t_decay:
            decay_idx = idx
            break

    decay_idx = min(decay_idx, peak_idx)

    # Initialize start_idx to net load trough from decay_idx up to peak_idx
    start_idx = decay_idx + int(np.argmin(y[decay_idx:peak_idx])) if decay_idx < peak_idx else decay_idx

    # Identify Ramp Start (t_start):
    # Earliest inflection before peak where slope is >= 5 MW/min and sustained for 60 min (12 steps)
    for i in range(start_idx, len(y) - 12):
        slope_30m = (y[min(i + 6, len(y) - 1)] - y[i]) / 30.0  # MW per minute
        sustained = np.mean(y[i + 1 : i + 13]) > y[i]  # 60-min sustained rise

        if slope_30m >= 5.0 and sustained and i < peak_idx:
            start_idx = i
            break

    t_start = afternoon["interval_start_local"].iloc[start_idx]

    # Calculate the Target Vector (measured relative to 12:00 PM local noon reference)
    noon_ref = afternoon["interval_start_local"].iloc[0].replace(hour=12, minute=0, second=0)
    t_start_minutes = (t_start - noon_ref).total_seconds() / 60.0
    delta_t_minutes = (t_peak - t_start).total_seconds() / 60.0

    delta_mw_raw = afternoon["net_load"].iloc[peak_idx] - afternoon["net_load"].iloc[start_idx]
    solar_cap = afternoon["TOTAL_SOLAR_CAPACITY_MW"].iloc[start_idx]
    delta_mw_norm = delta_mw_raw / solar_cap

    # Max 5-minute ramp speed during the event (MW/min)
    ramp_segment = afternoon["net_load_smooth"].iloc[start_idx : peak_idx + 1].values
    s_max = np.max(np.diff(ramp_segment) / 5.0)

    total_solar_capacity_mw = afternoon["TOTAL_SOLAR_CAPACITY_MW"].max()
    
    # Disqualify days where peak is not in the evening or ramp is too brief
    if t_peak.hour < 16 or delta_t_minutes < 120:
        return None

    return {
        "date": afternoon["interval_start_utc"].iloc[0].date(),
        "t_start_min": t_start_minutes,
        "delta_t_min": delta_t_minutes,
        "delta_mw_norm": delta_mw_norm,
        "s_max": s_max,
        "t_start_local": t_start,
        "t_peak_local": t_peak,
        "delta_mw_raw": delta_mw_raw,
        "total_solar_capacitY_mw": total_solar_capacity_mw,
    }
