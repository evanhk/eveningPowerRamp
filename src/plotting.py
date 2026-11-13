import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def apply_style():
    """Apply consistent matplotlib styling across project notebooks."""
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "font.size": 10,
    })


def plot_qualification_trends(qualifying_dates, df_5yr):
    """Plot seasonal distribution and annual frequency of qualifying days alongside installed solar capacity."""
    qual_dt = pd.to_datetime(qualifying_dates)
    month_counts = qual_dt.dt.month.value_counts().sort_index()
    yearly_counts = qual_dt.dt.year.value_counts().sort_index()
    yearly_cap = df_5yr.groupby(df_5yr["interval_start_local"].dt.year)["TOTAL_SOLAR_CAPACITY_MW"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax1.bar(
        months,
        [month_counts.get(m, 0) for m in range(1, 13)],
        color="tab:blue",
        width=0.6,
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_ylabel("Qualifying Days (2021–2025 Total)")
    ax1.set_title("Seasonal Distribution of Qualifying Days", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 150)

    years = [str(y) for y in yearly_counts.index]
    bars = ax2.bar(years, yearly_counts.values, color="tab:blue", width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Qualifying Days per Year")
    ax2.set_ylim(0, 300)
    ax2.set_title("Annual Qualification Frequency & Fleet Growth", fontsize=11, fontweight="bold")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, f"{int(yval)} d", ha="center", va="bottom", fontsize=9)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(years, yearly_cap.values, color="darkorange", marker="o", linewidth=2.2)
    ax2_twin.set_ylabel("Installed Solar Capacity (MW)", color="darkorange")
    ax2_twin.tick_params(axis="y", labelcolor="darkorange")
    ax2_twin.set_ylim(3000, 8500)
    ax2_twin.grid(False)

    plt.show()


def plot_seasonal_load_comparison(df, spring_date, winter_date):
    """Plot gross demand, net load, and renewable masking for a spring and winter qualifying day."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, target_date, label in zip(
        axes, [spring_date, winter_date], ["Spring Qualifying Day", "Winter Qualifying Day"]
    ):
        day_slice = df[df["interval_start_local"].dt.date == target_date]
        x_time = day_slice["interval_start_local"].dt.tz_localize(None)

        ax.plot(x_time, day_slice["gross_demand"], color="tab:gray", linestyle="--", linewidth=1.8, label="Gross Demand")
        ax.plot(x_time, day_slice["net_load"], color="tab:blue", linewidth=2.2, label="Net Load (Grid Burden)")
        ax.fill_between(
            x_time,
            day_slice["net_load"],
            day_slice["gross_demand"],
            color="gold",
            alpha=0.35,
            label="Solar + Wind Generation",
        )
        ax.set_title(f"{label} ({target_date})", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.set_ylim(bottom=0, top=22000)

    axes[0].set_ylabel("Power (MW)")
    axes[0].legend(loc="upper left", framealpha=0.9)
    plt.show()


def plot_segmentation_benchmark(day_data, afternoon, cp_times, targets):
    """Plot side-by-side comparison of PELT changepoints vs. physical trajectory ramp extraction."""
    sample_date = day_data["interval_start_local"].iloc[0].date()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_aft = afternoon["interval_start_local"].dt.tz_localize(None)
    ax1.plot(x_aft, afternoon["net_load"], color="lightgray", linewidth=1.5, label="Raw Net Load")
    ax1.plot(x_aft, afternoon["net_load_smooth"], color="tab:blue", linewidth=2.2, label="Savitzky-Golay (35-min)")
    for i, cp_time in enumerate(cp_times):
        ax1.axvline(
            cp_time.tz_localize(None),
            color="tab:red",
            linestyle="--",
            alpha=0.85,
            label="PELT Changepoint" if i == 0 else "",
        )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
    ax1.set_title(f"PELT Changepoint Segmentation ({sample_date})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Net Load (MW)")
    ax1.legend(loc="upper left", framealpha=0.9)

    x_full = day_data["interval_start_local"].dt.tz_localize(None)
    t_start_naive = targets["t_start_local"].tz_localize(None)
    t_peak_naive = targets["t_peak_local"].tz_localize(None)

    ax2.plot(x_full, day_data["gross_demand"], color="tab:gray", linestyle="--", linewidth=1.5, label="Gross Demand")
    ax2.plot(x_full, day_data["net_load"], color="lightblue", linewidth=1.8, label="Raw Net Load")
    ax2.plot(x_aft, afternoon["net_load_smooth"], color="tab:blue", linewidth=2.3, label="Savitzky-Golay (Afternoon)")
    ax2.axvspan(t_start_naive, t_peak_naive, color="orange", alpha=0.2, label=f"Extracted Ramp ({targets['delta_t_min']:.0f} min)")
    ax2.axvline(pd.Timestamp(f"{sample_date} 12:00"), color="purple", linestyle=":", linewidth=1.8, label="12:00 PM Cutoff")
    ax2.axvline(t_start_naive, color="green", linewidth=2.0, label=f"Onset ({targets['t_start_local'].strftime('%I:%M %p')})")
    ax2.axvline(t_peak_naive, color="red", linewidth=2.0, label=f"Peak ({targets['t_peak_local'].strftime('%I:%M %p')})")

    ax2.set_ylim(bottom=0, top=day_data["gross_demand"].max() * 1.15)
    ax2.set_xlim(pd.Timestamp(f"{sample_date} 00:00"), pd.Timestamp(f"{sample_date} 23:55"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax2.set_title(f"Physical Trajectory Extraction ({sample_date})", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Power (MW)")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.show()


def plot_target_distributions(targets_df):
    """Plot a 2x2 distribution grid for the 4D evening ramp target vector."""
    specs = [
        ("t_start_min", "Ramp Start Time", "Minutes Past 12:00 PM", "tab:blue"),
        ("delta_mw_norm", "Normalized Ramp Magnitude", "ΔMW / Solar Capacity", "tab:orange"),
        ("delta_t_min", "Event Duration", "Duration (minutes)", "tab:green"),
        ("s_max", "Peak Ramping Speed", "Max 5-Min Rate (MW/min)", "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for idx, (ax, (col, title, xlabel, color)) in enumerate(zip(axes.flat, specs)):
        series = targets_df[col]
        med = series.median()
        ax.hist(series, bins=25, color=color, alpha=0.75, edgecolor="black", linewidth=0.7)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.6, label=f"Median: {med:.2f}" if col == "delta_mw_norm" else f"Median: {med:.0f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 280)
        ax.set_yticks(range(0, 281, 50))
        if idx % 2 == 0:
            ax.set_ylabel("Event Count (Days)")
        ax.legend(loc="upper right", framealpha=0.9)

    plt.show()

