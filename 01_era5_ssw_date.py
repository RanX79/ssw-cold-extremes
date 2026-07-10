# -*- coding: utf-8 -*-
"""
Identify Northern Hemisphere stratospheric sudden warming (SSW) dates
from ERA5 daily zonal wind data.

The script applies the standard 10-hPa, 60°N zonal-mean zonal-wind
reversal criterion during November-March. Events that satisfy the
final-warming criterion are excluded.

Input:
    Daily ERA5 zonal wind data containing pressure-level, latitude,
    longitude, and time dimensions. Raw ERA5 data are not distributed
    with this repository.

Output:
    CSV files containing all detected events and the retained NDJFM SSW
    events, written to OUTPUT_DIR.

Before running:
    Update ERA5_U_FILE to the location of your downloaded ERA5 file.

Run:
    python scripts/01_era5_ssw_dates.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------
# Replace this placeholder with the path to your locally downloaded
# ERA5 daily zonal-wind NetCDF file.
ERA5_U_FILE = Path("path/to/ERA5/u_daily_1940_2025_10hPa_no_feb29.nc")

# Generated CSV and text files are written here.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1940
END_YEAR = 2024
LATITUDE = 60
TARGET_LEVEL = 10
MIN_WESTERLY_DAYS = 20
MIN_RECOVERY_DAYS = 10
DETECTION_MONTHS = (11, 12, 1, 2, 3)
TARGET_MONTHS = (11, 12, 1, 2, 3)

PRESSURE_LABEL = f"{TARGET_LEVEL}hPa"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def get_lat_lon_dim_names(da):
    """Return latitude and longitude dimension names."""
    lat_candidates = [d for d in da.dims if "lat" in d.lower()]
    lon_candidates = [d for d in da.dims if "lon" in d.lower()]

    if not lat_candidates:
        raise ValueError(f"Latitude dimension not found. Dimensions: {da.dims}")
    if not lon_candidates:
        raise ValueError(f"Longitude dimension not found. Dimensions: {da.dims}")

    return lat_candidates[0], lon_candidates[0]


def get_time_dim_name(da):
    """Return the time dimension name."""
    for name in ("valid_time", "time"):
        if name in da.dims:
            return name
    raise ValueError(
        f"Time dimension ('valid_time' or 'time') not found. Dimensions: {da.dims}"
    )


def get_level_dim_name(da):
    """Return the pressure-level dimension name."""
    for name in ("pressure_level", "level", "plev", "lev"):
        if name in da.dims:
            return name
    raise ValueError(f"Pressure-level dimension not found. Dimensions: {da.dims}")


def to_winter_year_label(init_year):
    """Return a winter-year label, e.g. 1981 -> '1981-1982'."""
    return f"{init_year}-{init_year + 1}"


def calc_max_westerly_streak(series):
    """Return the maximum consecutive duration with zonal wind >= 0."""
    max_streak = 0
    current_streak = 0

    for value in series:
        if pd.notna(value) and value >= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def is_final_warming(u_series_full, event_date, min_recovery_days=10):
    """
    Determine whether an easterly-wind reversal is a final warming.

    An event is considered a final warming if there is no return to
    westerlies for at least min_recovery_days consecutive days before
    30 April of the relevant year.
    """
    if event_date.month >= 10:
        deadline = pd.Timestamp(f"{event_date.year + 1}-04-30")
    else:
        deadline = pd.Timestamp(f"{event_date.year}-04-30")

    after_event = u_series_full.loc[
        (u_series_full.index > event_date)
        & (u_series_full.index <= deadline)
    ]

    if len(after_event) == 0:
        return True

    return calc_max_westerly_streak(after_event) < min_recovery_days


def inspect_dataset_structure(ds):
    """Print dataset variables and coordinates for diagnostics."""
    print("\n[DEBUG] Dataset structure:")
    for name, da in ds.data_vars.items():
        print(f"  data variable: {name}, dims={da.dims}, shape={da.shape}")
    for name, da in ds.coords.items():
        print(f"  coordinate: {name}, dims={da.dims}, shape={da.shape}")


def choose_u_variable(ds):
    """Identify the zonal-wind variable in an ERA5 dataset."""
    if "u" in ds.data_vars:
        return "u"

    candidate_names = []

    for name, da in ds.data_vars.items():
        dims_lower = [d.lower() for d in da.dims]
        has_time = any(d in ("time", "valid_time") for d in dims_lower)
        has_level = any(
            d in ("level", "lev", "pressure_level", "plev")
            for d in dims_lower
        )
        has_lat = any("lat" in d for d in dims_lower)
        has_lon = any("lon" in d for d in dims_lower)

        if has_time and has_level and has_lat and has_lon:
            candidate_names.append(name)

    if len(candidate_names) == 1:
        return candidate_names[0]

    if len(candidate_names) > 1:
        print(
            f"[WARNING] Multiple candidate wind variables found: "
            f"{candidate_names}. Using {candidate_names[0]}."
        )
        return candidate_names[0]

    inspect_dataset_structure(ds)
    raise ValueError(
        "No zonal-wind variable with time, pressure-level, latitude, "
        "and longitude dimensions was found."
    )


def open_u_dataarray(file_path):
    """Open the ERA5 dataset and select its zonal-wind variable."""
    ds = xr.open_dataset(file_path)
    variable_name = choose_u_variable(ds)
    da = ds[variable_name]

    print(f"[INFO] Using variable: {variable_name}")
    print(f"[INFO] Dimensions: {da.dims}")

    return ds, da


def select_pressure_level(da, target_level=10):
    """Select the requested pressure level from the DataArray."""
    level_name = get_level_dim_name(da)
    level_values = da[level_name].values
    level_values_numeric = pd.to_numeric(
        pd.Index(level_values),
        errors="coerce",
    )

    if np.any(level_values_numeric == target_level):
        matched_index = np.where(level_values_numeric == target_level)[0][0]
        matched_value = level_values[matched_index]
        return da.sel({level_name: matched_value})

    sample_levels = list(level_values[: min(10, len(level_values))])
    raise ValueError(
        f"Target level {target_level} hPa was not found. "
        f"Available levels include: {sample_levels}"
    )


def preprocess_era5_u(file_path, latitude=60, target_level=10):
    """Extract daily 60°N zonal-mean zonal wind at a pressure level."""
    ds, da = open_u_dataarray(file_path)

    try:
        da = select_pressure_level(da, target_level=target_level)
        lat_name, lon_name = get_lat_lon_dim_names(da)
        time_name = get_time_dim_name(da)

        da_60n = da.sel({lat_name: latitude}, method="nearest")
        da_zonal_mean = da_60n.mean(dim=lon_name)

        times = pd.to_datetime(da_zonal_mean[time_name].values)
        da_zonal_mean = da_zonal_mean.sel({time_name: times}).sortby(time_name)

        da_zonal_mean.load()

    finally:
        ds.close()

    return da_zonal_mean, time_name


def extract_one_winter(u_series_all, init_year):
    """Extract November-March zonal wind for one winter."""
    start_date = pd.Timestamp(f"{init_year}-11-01")
    end_date = pd.Timestamp(f"{init_year + 1}-03-31")

    return u_series_all.loc[
        (u_series_all.index >= start_date)
        & (u_series_all.index <= end_date)
    ].dropna().sort_index()


# ---------------------------------------------------------------------
# SSW detection
# ---------------------------------------------------------------------
def detect_ssw_events_for_one_winter(
    u_series_winter,
    u_series_full,
    min_westerly_days=20,
    min_recovery_days=10,
):
    """
    Identify independent SSW events in one winter and exclude final warmings.

    An SSW onset is defined as a transition from u >= 0 to u < 0 at
    60°N and 10 hPa. Independent events must be separated by at least
    min_westerly_days consecutive days with u >= 0.
    """
    u_series_winter = u_series_winter.dropna().sort_index()

    if len(u_series_winter) < 2:
        return []

    reversal_dates = [
        u_series_winter.index[i]
        for i in range(1, len(u_series_winter))
        if u_series_winter.iloc[i] < 0
        and u_series_winter.iloc[i - 1] >= 0
        and u_series_winter.index[i].month in DETECTION_MONTHS
    ]

    if not reversal_dates:
        return []

    raw_events = []

    for event_date in reversal_dates:
        if is_final_warming(
            u_series_full,
            event_date,
            min_recovery_days=min_recovery_days,
        ):
            continue

        end_date = event_date
        start_index = u_series_winter.index.get_loc(event_date)

        for index in range(start_index + 1, len(u_series_winter)):
            if u_series_winter.iloc[index] >= 0:
                end_date = u_series_winter.index[index - 1]
                break
            end_date = u_series_winter.index[index]

        event_u = u_series_winter.loc[event_date:end_date]

        raw_events.append(
            {
                "ssw_date": event_date,
                "end_date": end_date,
                "duration_days": int((end_date - event_date).days + 1),
                "min_u": float(event_u.min()),
                "min_u_date": event_u.idxmin(),
            }
        )

    if not raw_events:
        return []

    filtered_events = [raw_events[0]]

    for event in raw_events[1:]:
        previous_event = filtered_events[-1]

        interval = u_series_winter.loc[
            (u_series_winter.index > previous_event["end_date"])
            & (u_series_winter.index < event["ssw_date"])
        ]

        if (
            len(interval) > 0
            and calc_max_westerly_streak(interval) >= min_westerly_days
        ):
            filtered_events.append(event)

    return filtered_events


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main():
    """Run ERA5 SSW detection and write output files."""
    if not ERA5_U_FILE.exists():
        raise FileNotFoundError(
            f"ERA5 input file was not found:\n  {ERA5_U_FILE}\n\n"
            "Edit ERA5_U_FILE in the user-configuration section at "
            "the top of this script."
        )

    print("\n" + "=" * 80)
    print("Reading ERA5 data")
    print("=" * 80)

    da_zm, time_name = preprocess_era5_u(
        file_path=ERA5_U_FILE,
        latitude=LATITUDE,
        target_level=TARGET_LEVEL,
    )

    times_all = pd.to_datetime(da_zm[time_name].values)
    u_all = pd.Series(da_zm.values, index=times_all).sort_index()

    print(f"File: {ERA5_U_FILE.name}")
    print(f"Time range: {times_all[0]} to {times_all[-1]}")
    print(f"Number of daily values: {len(times_all)}")

    winter_records = []

    for year in range(START_YEAR, END_YEAR):
        print(f"\nProcessing winter {year}-{year + 1}")

        winter_year = to_winter_year_label(year)
        u_series_winter = extract_one_winter(u_all, year)

        if len(u_series_winter) == 0:
            print("  No data available; skipping.")
            continue

        print(
            f"  Time range: {u_series_winter.index[0]} to "
            f"{u_series_winter.index[-1]}"
        )
        print(f"  Number of daily values: {len(u_series_winter)}")

        events = detect_ssw_events_for_one_winter(
            u_series_winter,
            u_series_full=u_all,
            min_westerly_days=MIN_WESTERLY_DAYS,
            min_recovery_days=MIN_RECOVERY_DAYS,
        )

        events = [
            event
            for event in events
            if event["ssw_date"].month in TARGET_MONTHS
        ]

        if not events:
            winter_records.append(
                {
                    "init_year": year,
                    "winter_year": winter_year,
                    "event_index": np.nan,
                    "has_ssw_NDJFM": False,
                    "ssw_date": pd.NaT,
                    "ssw_year": np.nan,
                    "ssw_month": np.nan,
                    "ssw_day": np.nan,
                    "end_date": pd.NaT,
                    "duration_days": np.nan,
                    "min_u": np.nan,
                    "min_u_date": pd.NaT,
                    "u_at_ssw": np.nan,
                }
            )
            print("  No SSW detected.")

        else:
            print(f"  Number of independent SSW events: {len(events)}")

            for event_index, event in enumerate(events, start=1):
                winter_records.append(
                    {
                        "init_year": year,
                        "winter_year": winter_year,
                        "event_index": event_index,
                        "has_ssw_NDJFM": True,
                        "ssw_date": event["ssw_date"],
                        "ssw_year": event["ssw_date"].year,
                        "ssw_month": event["ssw_date"].month,
                        "ssw_day": event["ssw_date"].day,
                        "end_date": event["end_date"],
                        "duration_days": event["duration_days"],
                        "min_u": event["min_u"],
                        "min_u_date": event["min_u_date"],
                        "u_at_ssw": float(
                            u_series_winter.loc[event["ssw_date"]]
                        ),
                    }
                )

    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    winter_df = pd.DataFrame(winter_records)

    for column in ("ssw_date", "end_date", "min_u_date"):
        if column in winter_df.columns:
            winter_df[column] = pd.to_datetime(
                winter_df[column],
                errors="coerce",
            )

    winter_df = winter_df.sort_values(
        ["init_year", "ssw_date"]
    ).reset_index(drop=True)

    all_csv = OUTPUT_DIR / (
        f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_"
        f"{START_YEAR}_{END_YEAR}.csv"
    )
    winter_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    print(f"All results: {all_csv}")

    events_only_df = winter_df[winter_df["has_ssw_NDJFM"]].copy()

    events_only_csv = OUTPUT_DIR / (
        f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_events_only_"
        f"{START_YEAR}_{END_YEAR}.csv"
    )
    events_only_df.to_csv(events_only_csv, index=False, encoding="utf-8-sig")
    print(f"SSW-event dates: {events_only_csv}")

    text_file = OUTPUT_DIR / (
        f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_"
        f"{START_YEAR}_{END_YEAR}.txt"
    )

    with open(text_file, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write(f"ERA5 SSW dates ({PRESSURE_LABEL}, NDJFM)\n")
        file.write(f"Analysis period: {START_YEAR}-{END_YEAR}\n")
        file.write(
            f"Criterion: {PRESSURE_LABEL}, {LATITUDE}N zonal-mean "
            "zonal-wind reversal\n"
        )
        file.write(f"Detection months: {DETECTION_MONTHS}\n")
        file.write(f"Retained months: {TARGET_MONTHS}\n")
        file.write(
            f"Minimum westerly separation: {MIN_WESTERLY_DAYS} days\n"
        )
        file.write(
            f"Final-warming filter: recovery < {MIN_RECOVERY_DAYS} "
            "consecutive westerly days before 30 April; excluded\n"
        )
        file.write(
            "Input: daily ERA5 multi-level zonal-wind data; "
            "29 February removed in the source data.\n"
        )
        file.write("=" * 80 + "\n\n")

        for year in range(START_YEAR, END_YEAR):
            subset = events_only_df[
                events_only_df["init_year"] == year
            ].copy()

            winter_year = to_winter_year_label(year)
            file.write(f"{winter_year}\n" + "-" * 60 + "\n")

            if subset.empty:
                file.write("No NDJFM SSW detected.\n\n")
                continue

            subset = subset.sort_values(["ssw_date", "event_index"])

            for _, row in subset.iterrows():
                file.write(
                    f"event={int(row['event_index'])}, "
                    f"ssw_date={row['ssw_date'].strftime('%Y-%m-%d')}, "
                    f"duration={int(row['duration_days'])} days, "
                    f"min_u={row['min_u']:.2f} m s-1\n"
                )

            file.write("\n")

    print(f"Text summary: {text_file}")
    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
