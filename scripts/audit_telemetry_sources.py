
"""Audit telemetry discrepancies between ISO-NE Web Services API and GridStatus.

This script tests three empirical properties across telemetry sources:
1. Real-time parity: Verifies that live broadcast streams match on the current interval.
2. Longitudinal invariance: Evaluates a recorded interval over time to determine which endpoint updates.
3. Metered load invariance: Verifies that physical substation load matches across all time horizons.
"""

import logging
from dotenv import load_dotenv
import pandas as pd
from gridstatusio import GridStatusClient
from gridstatus.isone_api import isone_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_live_stream_parity(iso: isone_api.ISONEAPI, client: GridStatusClient) -> None:
    """Compare the latest completed 5-minute interval across both APIs."""
    recent_iso = iso.get_total_demand("latest")
    recent_ts = recent_iso["Interval Start"].iloc[0]

    recent_gs = client.get_dataset(
        "isone_standardized_5_min",
        start=recent_ts - pd.Timedelta(minutes=20),
        end=recent_ts + pd.Timedelta(minutes=5),
        timezone="market",
    ).dropna(subset=["load.load", "btm_solar.btm_solar"])

    latest_gs_row = recent_gs.iloc[-1]
    matching_date_str = latest_gs_row["interval_start_local"].strftime("%Y-%m-%d")
    iso_day = iso.get_total_demand(matching_date_str)
    iso_row = iso_day[iso_day["Interval Start"] == latest_gs_row["interval_start_local"]].iloc[0]

    load_delta = abs(latest_gs_row["load.load"] - iso_row["Native Load"])
    iso_solar = iso_row["Native Load With Estimated Solar"] - iso_row["Native Load"]
    solar_delta = abs(latest_gs_row["btm_solar.btm_solar"] - iso_solar)

    logger.info("Live interval tested: %s", latest_gs_row["interval_start_local"])
    logger.info("  Metered Native Load delta: %.4f MW", load_delta)
    logger.info("  BTM Solar delta:            %.4f MW", solar_delta)

    assert load_delta == 0.0, "Metered native load mismatch on live stream."
    assert solar_delta == 0.0, "Live BTM solar broadcast estimate mismatch."


def audit_longitudinal_invariance(iso: isone_api.ISONEAPI, client: GridStatusClient) -> None:
    """Evaluate whether GridStatus or ISO-NE altered historical records after the operating day closed."""
    # Recorded live broadcast from 2026-10-27 15:30:00 EDT
    test_ts = pd.to_datetime("2026-10-27 15:30:00-04:00")
    recorded_live_solar = 2696.683

    gs_day = client.get_dataset(
        "isone_standardized_5_min",
        start="2026-10-27",
        end="2026-10-28",
        timezone="market",
    )
    iso_day = iso.get_total_demand("2026-10-27", "2026-10-28")

    gs_row = gs_day[gs_day["interval_start_local"] == test_ts].iloc[0]
    iso_row = iso_day[iso_day["Interval Start"] == test_ts].iloc[0]

    gs_solar = float(gs_row["btm_solar.btm_solar"])
    iso_solar = float(iso_row["Native Load With Estimated Solar"] - iso_row["Native Load"])

    load_delta = abs(float(gs_row["load.load"]) - float(iso_row["Native Load"]))

    logger.info("Longitudinal test timestamp: %s", test_ts)
    logger.info("  Original live broadcast recorded: %.3f MW", recorded_live_solar)
    logger.info("  GridStatus archive value:         %.4f MW", gs_solar)
    logger.info("  ISO-NE retrospective API value:   %.4f MW", iso_solar)
    logger.info("  Metered load delta:               %.4f MW", load_delta)

    assert load_delta == 0.0, "Metered native load changed retrospectively."
    assert abs(gs_solar - recorded_live_solar) < 1e-3, "GridStatus retroactively altered broadcast value."
    assert abs(iso_solar - recorded_live_solar) > 1.0, "ISO-NE did not update to post-operational figure."


def main() -> None:
    load_dotenv()
    logger.info("Initializing API clients...")
    iso = isone_api.ISONEAPI()
    client = GridStatusClient()

    logger.info("Running live stream parity audit...")
    audit_live_stream_parity(iso, client)

    logger.info("Running longitudinal invariance audit...")
    audit_longitudinal_invariance(iso, client)

    logger.info("Audit complete. All assertions passed.")


if __name__ == "__main__":
    main()
