# -*- coding: utf-8 -*-
"""
Identify stratospheric sudden warming (SSW) dates for the first
25 members of the SEAS5 hindcast.

The script identifies reversals of the 10-hPa, 60°N zonal-mean zonal
wind during November-March, excludes final warmings, and writes
member-wise SSW dates to CSV files.

Input:
    One daily SEAS5 zonal-wind NetCDF file per initialization year.
    Raw SEAS5 data are not distributed with this repository.

Output:
    CSV files containing all member-wise events and retained NDJFM SSW
    events, written to OUTPUT_DIR.

Before running:
    Update U_DAILY_DIR to the directory containing your daily SEAS5
    zonal-wind files.

Expected input filename pattern:
    SEAS5_u10hPa_NH_{year}11_system51_m25_daily.nc

Run:
    python scripts/02_seas5_ssw_dates.py
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
PRESSURE_LEVEL = 10

# Replace this placeholder with the directory containing your locally
# downloaded daily SEAS5 zonal-wind NetCDF files.
U_DAILY_DIR = Path("path/to/SEAS5/u10_daily")

# Generated CSV and text files will be saved here.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1981
END_YEAR = 2024
LATITUDE = 60
N_MEMBERS = 25
MIN_WESTERLY_DAYS = 20
MIN_RECOVERY_DAYS = 10
DETECTION_MONTHS = (11, 12, 1, 2, 3)
TARGET_MONTHS = (11, 12, 1, 2, 3)

PRESSURE_LABEL = f"{PRESSURE_LEVEL}hPa"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def get_lat_lon_dim_names(da):
    """Return latitude and longitude dimension names."""
    lat_candidates = [dim for dim in da.dims if "lat" in dim.lower()]
    lon_candidates = [dim for dim in da.dims if "lon" in dim.lower()]

    if not lat_candidates:
        raise ValueError(f"Latitude dimension not found. Dimensions: {da.dims}")
    if not lon_candidates:
        raise ValueError(f"Longitude dimension not found. Dimensions: {da.dims}")

    return lat_candidates[0], lon_candidates[0]


def get_time_dim_name(da):
    """Return the time dimension name."""
    for name in ("valid_time", "time", "forecast_period"):
        if name in da.dims:
            return name

    raise ValueError(
        f"Time dimension not found. Expected valid_time, time, or "
        f"forecast_period; dimensions: {da.dims}"
    )


def to_winter_year_label(init_year):
    """Return a winter-year label, e.g. 1981 -> '1981-1982'."""
    return f"{init_year}-{init_year + 1}"


def safe_member_label(number_coord, index):
    """Return an integer ensemble-member label where possible."""
    try:
        return int(number_coord[index])
    except (TypeError, ValueError):
        return index + 1


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


def is_final_warming(u_series, event_date, min_recovery_days=10):
    """
    Determine whether an easterly-wind reversal is a final warming.

    An event is classified as a final warming if, between the reversal
    date and 30 April of the relevant year, there is no recovery to
    westerlies for at least min_recovery_days consecutive days.
    """
    if event_date.month >= 10:
        deadline = pd.Timestamp(f"{event_date.year + 1}-04-30")
    else:
        deadline = pd.Timestamp(f"{event_date.year}-04-30")

    after_event = u_series.loc[
        (u_series.index > event_date)
        & (u_series.index <= deadline)
    ]

    if len(after_event) == 0:
        return True

    return calc_max_westerly_streak(after_event) < min_recovery_days


def get_input_file_path(init_year):
    """Return the expected daily SEAS5 zonal-wind file path."""
    filename = (
        f"SEAS5_u{PRESSURE_LEVEL}hPa_NH_"
        f"{init_year}11_system51_m25_daily.nc"
    )
    return U_DAILY_DIR / filename


# ---------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------
def preprocess_one_year(init_year, latitude=60, n_members=25):
    """
    Read one SEAS5 initialization-year file and extract 60°N zonal-mean U.

    Only November-March dates are retained, because these are the SSW
    detection months.
    """
    input_file = get_input_file_path(init_year)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ds = xr.open_dataset(input_file)

    try:
        if not ds.data_vars:
            raise ValueError(f"No data variables found in: {input_file}")

        variable_name = list(ds.data_vars)[0]
        da = ds[variable_name].load()

    finally:
        ds.close()

    if "number" not in da.dims:
        raise ValueError(
            f"Ensemble-member dimension 'number' not found. "
            f"Dimensions: {da.dims}"
        )

    da = da.transpose("number", *[dim for dim in da.dims if dim != "number"])
    da = da.isel(number=slice(0, n_members))

    time_name = get_time_dim_name(da)
    lat_name, lon_name = get_lat_lon_dim_names(da)

    da_60n = da.sel({lat_name: latitude}, method="nearest")
    da_zonal_mean = da_60n.mean(dim=lon_name)

    times = pd.to_datetime(da_zonal_mean[time_name].values)
    month_mask = np.isin(times.month, DETECTION_MONTHS)

    da_zonal_mean = da_zonal_mean.sel(
        {time_name: times[month_mask]}
    ).sortby(time_name)

    return da_zonal_mean, time_name, input_file.name


# ---------------------------------------------------------------------
# SSW detection
# ---------------------------------------------------------------------
def detect_ssw_events_for_one_member(
    u_series,
    min_westerly_days=20,
    min_recovery_days=10,
):
    """
    Identify independent SSW events for one ensemble member.

    An onset is defined as a transition from westerly wind (U >= 0)
    to easterly wind (U < 0) at 60°N and 10 hPa. Final warmings are
    removed. Independent events require at least min_westerly_days of
    intervening consecutive westerlies.
    """
    u_series = u_series.dropna().sort_index()

    if len(u_series) < 2:
        return []

    reversal_dates = [
        u_series.index[index]
        for index in range(1, len(u_series))
        if u_series.iloc[index] < 0
        and u_series.iloc[index - 1] >= 0
        and u_series.index[index].month in DETECTION_MONTHS
    ]

    if not reversal_dates:
        return []

    raw_events = []

    for event_date in reversal_dates:
        end_date = event_date
        start_index = u_series.index.get_loc(event_date)

        for index in range(start_index + 1, len(u_series)):
            if u_series.iloc[index] >= 0:
                end_date = u_series.index[index - 1]
                break
            end_date = u_series.index[index]

        event_u = u_series.loc[event_date:end_date]

        if is_final_warming(
            u_series,
            event_date,
            min_recovery_days=min_recovery_days,
        ):
            continue

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

        interval = u_series.loc[
            (u_series.index > previous_event["end_date"])
            & (u_series.index < event["ssw_date"])
        ]

        if (
            len(interval) > 0
            and calc_max_westerly_streak(interval) >= min_westerly_days
        ):
            filtered_events.append(event)

    return filtered_events


# ---------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------
def write_output_files(memberwise_records):
    """Write complete event records, event-only records, and text summary."""
    memberwise_df = pd.DataFrame(memberwise_records)

    if memberwise_df.empty:
        raise RuntimeError(
            "No SEAS5 records were produced. Check U_DAILY_DIR, the "
            "filename pattern, and the selected analysis years."
        )

    for column in ("ssw_date", "end_date", "min_u_date"):
        memberwise_df[column] = pd.to_datetime(
            memberwise_df[column],
            errors="coerce",
        )

    memberwise_df = memberwise_df.sort_values(
        ["init_year", "member", "ssw_date"]
    ).reset_index(drop=True)

    output_all = OUTPUT_DIR / (
        f"SEAS5_first25members_SSW_dates_NDJFM_"
        f"{START_YEAR}_{END_YEAR}.csv"
    )
    memberwise_df.to_csv(output_all, index=False, encoding="utf-8-sig")
    print(f"All records: {output_all}")

    events_df = memberwise_df[
        memberwise_df["has_ssw_NDJFM"]
    ].copy()

    output_events = OUTPUT_DIR / (
        f"SEAS5_first25members_SSW_dates_NDJFM_events_only_"
        f"{START_YEAR}_{END_YEAR}.csv"
    )
    events_df.to_csv(output_events, index=False, encoding="utf-8-sig")
    print(f"SSW events only: {output_events}")

    text_file = OUTPUT_DIR / (
        f"SEAS5_first25members_SSW_dates_NDJFM_"
        f"{START_YEAR}_{END_YEAR}.txt"
    )

    with open(text_file, "w", encoding="utf-8") as file:
        file.write("=" * 80 + "\n")
        file.write("SEAS5 first 25 members SSW dates (NDJFM)\n")
        file.write(
            f"Period: {START_YEAR}-{END_YEAR} | "
            f"{PRESSURE_LABEL} | {LATITUDE}N\n"
        )
        file.write(
            f"Minimum westerly separation: {MIN_WESTERLY_DAYS} days\n"
        )
        file.write(
            f"Final-warming filter: recovery < {MIN_RECOVERY_DAYS} "
            "consecutive westerly days before 30 April; excluded\n"
        )
        file.write("=" * 80 + "\n\n")

        for year in range(START_YEAR, END_YEAR + 1):
            subset = events_df[
                events_df["init_year"] == year
            ].sort_values(["ssw_date", "member"])

            file.write(f"{to_winter_year_label(year)}\n")
            file.write("-" * 60 + "\n")

            if subset.empty:
                file.write("No NDJFM SSW detected.\n\n")
                continue

            for _, row in subset.iterrows():
                file.write(
                    f"member={int(row['member']):02d}, "
                    f"event={int(row['event_index'])}, "
                    f"ssw_date={row['ssw_date'].strftime('%Y-%m-%d')}, "
                    f"duration={int(row['duration_days'])} days, "
                    f"min_u={row['min_u']:.2f} m s-1\n"
                )

            file.write("\n")

    print(f"Text summary: {text_file}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main():
    """Run member-wise SEAS5 SSW detection."""
    print("\n" + "=" * 80)
    print("SEAS5 member-wise SSW detection")
    print("=" * 80)
    print(f"Input directory: {U_DAILY_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Pressure level: {PRESSURE_LABEL}; latitude: {LATITUDE}N")
    print(f"Number of members: {N_MEMBERS}")
    print("=" * 80)

    memberwise_records = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nProcessing initialization year {year}")

        try:
            da_zonal_mean, time_name, filename = preprocess_one_year(
                year,
                latitude=LATITUDE,
                n_members=N_MEMBERS,
            )

            times = pd.to_datetime(da_zonal_mean[time_name].values)
            winter_year = to_winter_year_label(year)

            print(f"  File: {filename}")
            print(f"  Date range: {times[0].date()} to {times[-1].date()}")

            n_members_with_ssw = 0
            n_events_total = 0

            for member_index in range(da_zonal_mean.sizes["number"]):
                u_values = da_zonal_mean.isel(number=member_index).values
                u_series = pd.Series(u_values, index=times)

                events = detect_ssw_events_for_one_member(
                    u_series,
                    min_westerly_days=MIN_WESTERLY_DAYS,
                    min_recovery_days=MIN_RECOVERY_DAYS,
                )

                events = [
                    event
                    for event in events
                    if event["ssw_date"].month in TARGET_MONTHS
                ]

                member_id = safe_member_label(
                    da_zonal_mean["number"].values,
                    member_index,
                )

                if not events:
                    memberwise_records.append(
                        {
                            "init_year": year,
                            "winter_year": winter_year,
                            "member": member_id,
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

                else:
                    n_members_with_ssw += 1
                    n_events_total += len(events)

                    for event_index, event in enumerate(events, start=1):
                        memberwise_records.append(
                            {
                                "init_year": year,
                                "winter_year": winter_year,
                                "member": member_id,
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
                                    u_series.loc[event["ssw_date"]]
                                ),
                            }
                        )

            print(
                f"  Members with SSW: {n_members_with_ssw}/"
                f"{da_zonal_mean.sizes['number']}; "
                f"total events: {n_events_total}"
            )

        except Exception as error:
            print(f"  Failed for initialization year {year}: {error}")

    print("\n" + "=" * 80)
    print("Writing output files")
    print("=" * 80)

    write_output_files(memberwise_records)

    print("\n" + "=" * 80)
    print("Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
