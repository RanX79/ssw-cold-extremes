# -*- coding: utf-8 -*-
"""
Reproduce Figure 4: post-SSW cold-anomaly area in SEAS5.

For days 15-59 after each SSW onset, the script calculates the
cosine-latitude-weighted regional area where daily 2-m temperature
anomalies are below the baseline 10th-percentile anomaly threshold.
It compares SSW events with matched-date non-SSW member-year analogs
in overlapping 10-year windows.

Raw SEAS5 data are not distributed with this repository. Update
SSW_CSV_PATH and T2M_DAILY_DIR before running.

Output:
    - NPZ cache with sliding-window statistics
    - Figure 4 PDF in OUTPUT_DIR

Run:
    python scripts/06_figure4_cumulative_cold_anomaly_area.py
"""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import theilslopes

plt.rcParams["font.family"] = "Arial"

# ================================================================
# SETTINGS
# ================================================================
SEAS5_SSW_CSV_PATH = Path(
    "path/to/analysis_results/"
    "SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv"
)

T2M_DAILY_DIR = Path("path/to/SEAS5/t2m_daily")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1981
END_YEAR = 2024
N_MEMBERS = 25

DAY_START = 15
DAY_END = 59
N_DAYS = DAY_END - DAY_START + 1
SLIDING_WINDOW = 10

BASELINE_START = 1981
BASELINE_END = 2010

PERCENTILE_THRESHOLD = 10.0

T2M_VAR_CANDIDATES = (
    "t2m",
    "2t",
    "2m_temperature",
    "t2m_daily",
)

REGION_BOXES = {
    "NorthAmerica": {
        "lat_min": 45,
        "lat_max": 70,
        "lon_min": -140,
        "lon_max": -60,
    },
    "Europe": {
        "lat_min": 45,
        "lat_max": 70,
        "lon_min": 0,
        "lon_max": 40,
    },
    "EastAsia": {
        "lat_min": 45,
        "lat_max": 70,
        "lon_min": 60,
        "lon_max": 120,
    },
}

REGION_LABELS = {
    "NorthAmerica": "North America",
    "Europe": "Europe",
    "EastAsia": "East Asia",
}

REGION_COLORS = {
    "NorthAmerica": "#874F8D",
    "Europe": "#1C6AB1",
    "EastAsia": "#ED4043",
}

REGION_ORDER = (
    "NorthAmerica",
    "Europe",
    "EastAsia",
)

NPZ_PATH = OUTPUT_DIR / (
    f"fig4_SSW_cumColdArea_TanomP{int(PERCENTILE_THRESHOLD)}"
    f"_day{DAY_START}to{DAY_END}"
    f"_mem{N_MEMBERS}_{START_YEAR}_{END_YEAR}"
    f"_sliding{SLIDING_WINDOW}"
    f"_NDJFM_daily_{BASELINE_END}.npz"
)

PDF_PATH = OUTPUT_DIR / (
    f"fig4_SEAS5_SSW_cumColdArea_TanomP{int(PERCENTILE_THRESHOLD)}"
    f"_10yrsliding_NDJFM_daily_{BASELINE_END}.pdf"
)


# ================================================================
# BASIC TOOLS
# ================================================================
def get_var_name(ds):
    """Identify the 2-m temperature variable."""
    for variable_name in T2M_VAR_CANDIDATES:
        if variable_name in ds.data_vars:
            return variable_name

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(
        f"Cannot identify T2m variable: {list(ds.data_vars)}"
    )


def get_lat_lon_dims(da):
    """Return latitude and longitude dimension names."""
    lat_candidates = [
        dimension
        for dimension in da.dims
        if "lat" in dimension.lower()
    ]
    lon_candidates = [
        dimension
        for dimension in da.dims
        if "lon" in dimension.lower()
    ]

    if not lat_candidates:
        raise ValueError(f"No latitude dimension: {da.dims}")

    if not lon_candidates:
        raise ValueError(f"No longitude dimension: {da.dims}")

    return lat_candidates[0], lon_candidates[0]


def get_time_dim(da):
    """Return the time dimension name."""
    for name in ("valid_time", "time", "forecast_period"):
        if name in da.dims:
            return name

    raise ValueError(f"No time dimension: {da.dims}")


def month_day_key(timestamp):
    """Return a calendar date key in MM-DD format."""
    timestamp = pd.Timestamp(timestamp)
    return f"{timestamp.month:02d}-{timestamp.day:02d}"


def get_daily_filepath(year):
    """Return the expected SEAS5 daily T2m file path."""
    return T2M_DAILY_DIR / (
        f"SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"
    )


def read_year_data(year):
    """
    Read one daily SEAS5 T2m file.

    Returns
    -------
    arr : ndarray
        Shape: (member, time, latitude, longitude).
    times : DatetimeIndex
    members : list[int]
    lat_values : ndarray
    lon_values : ndarray
    """
    input_file = get_daily_filepath(year)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    ds = xr.open_dataset(input_file)

    try:
        variable_name = get_var_name(ds)
        da = ds[variable_name].load()
    finally:
        ds.close()

    if "number" not in da.dims:
        raise ValueError(
            f"Ensemble-member dimension 'number' not found: {da.dims}"
        )

    for dimension in list(da.dims):
        is_non_spatial_singleton = (
            dimension != "number"
            and da.sizes[dimension] == 1
            and "lat" not in dimension.lower()
            and "lon" not in dimension.lower()
            and "time" not in dimension.lower()
            and "valid" not in dimension.lower()
        )

        if is_non_spatial_singleton:
            da = da.squeeze(dimension, drop=True)

    da = da.transpose(
        "number",
        *[
            dimension
            for dimension in da.dims
            if dimension != "number"
        ],
    )

    da = da.isel(number=slice(0, N_MEMBERS))

    lat_name, lon_name = get_lat_lon_dims(da)
    time_name = get_time_dim(da)

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    original_lon = da[lon_name].values

    if np.nanmax(original_lon) > 180:
        converted_lon = np.where(
            original_lon > 180,
            original_lon - 360,
            original_lon,
        )

        sort_index = np.argsort(converted_lon)

        da = da.isel({lon_name: sort_index})
        da = da.assign_coords({lon_name: converted_lon[sort_index]})

    lat_values = da[lat_name].values.copy()
    lon_values = da[lon_name].values.copy()
    times = pd.to_datetime(da[time_name].values).normalize()

    members = []

    for index in range(da.sizes["number"]):
        try:
            members.append(int(da["number"].values[index]))
        except (TypeError, ValueError):
            members.append(index + 1)

    arr = da.values.astype(np.float32)

    del da
    gc.collect()

    return arr, times, members, lat_values, lon_values


def scan_all_years():
    """
    Scan available daily files and construct metadata for each year.
    """
    metadata = {}

    for year in range(START_YEAR, END_YEAR + 1):
        input_file = get_daily_filepath(year)

        if not input_file.exists():
            print(f"  [WARNING] Missing file for {year}; skipped.")
            continue

        try:
            arr, times, members, lat_values, lon_values = read_year_data(
                year
            )

            metadata[year] = {
                "lat_vals": lat_values,
                "lon_vals": lon_values,
                "md": [month_day_key(timestamp) for timestamp in times],
                "member_to_idx": {
                    member: index
                    for index, member in enumerate(members)
                },
                "time_to_idx": {
                    pd.Timestamp(timestamp): index
                    for index, timestamp in enumerate(times)
                },
                "n_times": len(times),
            }

            print(f"  Metadata scanned: {year}; shape={arr.shape}")

            del arr, times, members, lat_values, lon_values
            gc.collect()

        except Exception as error:
            print(f"  [WARNING] Scan failed for {year}: {error}")
            gc.collect()

    if not metadata:
        raise RuntimeError(
            "No SEAS5 daily files were found. Check T2M_DAILY_DIR."
        )

    return metadata


def cosine_weights_2d(lat_values, lon_values, region_box):
    """Create masks and cosine-latitude weights for a region."""
    lat_mask = (
        (lat_values >= region_box["lat_min"])
        & (lat_values <= region_box["lat_max"])
    )

    lon_mask = (
        (lon_values >= region_box["lon_min"])
        & (lon_values <= region_box["lon_max"])
    )

    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError(
            f"Region does not overlap data grid: {region_box}"
        )

    weights_2d = (
        np.cos(np.deg2rad(lat_values[lat_mask]))[:, None]
        * np.ones(lon_mask.sum())
    )

    return lat_mask, lon_mask, weights_2d


def region_area_fraction_below_threshold(
    field_2d,
    threshold_2d,
    lat_mask,
    lon_mask,
    weights_2d,
):
    """
    Return cosine-weighted area fraction where field < threshold.
    """
    lat_index = np.where(lat_mask)[0]
    lon_index = np.where(lon_mask)[0]

    subset = field_2d[np.ix_(lat_index, lon_index)]
    threshold_subset = threshold_2d[np.ix_(lat_index, lon_index)]

    valid = np.isfinite(subset) & np.isfinite(threshold_subset)
    cold = (subset < threshold_subset) & valid

    total_weight = np.sum(valid * weights_2d)

    if total_weight == 0:
        return np.nan

    return float(np.sum(cold * weights_2d) / total_weight)


def read_ssw_events():
    """Read retained NDJFM SEAS5 SSW events."""
    if not SEAS5_SSW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"SSW-event CSV was not found:\n"
            f"  {SEAS5_SSW_CSV_PATH}\n\n"
            "Run 02_seas5_ssw_dates.py first or update "
            "SEAS5_SSW_CSV_PATH."
        )

    df = pd.read_csv(SEAS5_SSW_CSV_PATH)

    required_columns = ("init_year", "member", "ssw_date")

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"SSW-event CSV missing required column: {column}"
            )

    df["ssw_date"] = pd.to_datetime(
        df["ssw_date"],
        errors="coerce",
    ).dt.normalize()

    df = df.dropna(subset=["ssw_date"]).copy()

    df = df[
        (df["init_year"] >= START_YEAR)
        & (df["init_year"] <= END_YEAR)
    ].copy()

    df["member"] = df["member"].astype(int)
    df["month"] = df["ssw_date"].dt.month

    df = df[
        df["month"].isin((11, 12, 1, 2, 3))
    ].copy()

    df = df.sort_values(
        ["init_year", "member", "ssw_date"]
    ).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid NDJFM SSW events were found.")

    print(f"Loaded {len(df)} SSW events (NDJFM).")

    return df


# ================================================================
# BASELINE CLIMATOLOGY
# ================================================================
def build_baseline_climatology(metadata):
    """
    Build a smoothed baseline daily climatology from ensemble means.

    The climatology uses baseline years BASELINE_START-BASELINE_END
    and applies circular +/-5-day smoothing.
    """
    print(
        f"Building baseline climatology "
        f"({BASELINE_START}-{BASELINE_END}; +/-5 days)"
    )

    day_of_year_fields = {}

    for year in range(BASELINE_START, BASELINE_END + 1):
        if year not in metadata:
            continue

        arr, times, _, _, _ = read_year_data(year)

        ensemble_mean = arr.mean(axis=0)
        day_of_years = pd.to_datetime(times).dayofyear.values

        for index, day_of_year in enumerate(day_of_years):
            day_of_year_fields.setdefault(day_of_year, []).append(
                ensemble_mean[index].astype(np.float32)
            )

        del arr, times, ensemble_mean, day_of_years
        gc.collect()

    if not day_of_year_fields:
        raise RuntimeError(
            "No baseline data were available to construct climatology."
        )

    sample = next(iter(day_of_year_fields.values()))[0]
    n_lat, n_lon = sample.shape

    climatology_raw = np.full(
        (366, n_lat, n_lon),
        np.nan,
        dtype=np.float32,
    )

    for day_of_year, fields in day_of_year_fields.items():
        climatology_raw[day_of_year - 1] = np.nanmean(
            np.stack(fields, axis=0),
            axis=0,
        )

    del day_of_year_fields
    gc.collect()

    window = 11
    padding = window // 2

    climatology_padded = np.concatenate(
        [
            climatology_raw[-padding:],
            climatology_raw,
            climatology_raw[:padding],
        ],
        axis=0,
    )

    climatology_smoothed = np.full_like(
        climatology_raw,
        np.nan,
    )

    for index in range(366):
        values = climatology_padded[index:index + window]

        if np.isfinite(values).any():
            climatology_smoothed[index] = np.nanmean(
                values,
                axis=0,
            )

    del climatology_padded, climatology_raw
    gc.collect()

    baseline_climatology = {}

    for timestamp in pd.date_range("2001-01-01", "2001-12-31"):
        baseline_climatology[month_day_key(timestamp)] = (
            climatology_smoothed[timestamp.dayofyear - 1]
        )

    del climatology_smoothed
    gc.collect()

    print(
        f"  Baseline climatology complete: "
        f"{len(baseline_climatology)} calendar days"
    )

    return baseline_climatology


# ================================================================
# BASELINE P10 THRESHOLD
# ================================================================
def build_baseline_p10_threshold(metadata, baseline_climatology):
    """
    Build the daily baseline P10 threshold for raw T2m anomalies.

    For each target calendar day, the threshold is calculated from all
    baseline years, all ensemble members, and a +/-5-day calendar window.

    The threshold is based on:
        raw anomaly = T2m - baseline climatology
    """
    print(
        f"Building baseline P{PERCENTILE_THRESHOLD:.0f} thresholds "
        f"({BASELINE_START}-{BASELINE_END}; +/-5 days; all members)"
    )

    baseline_years = [
        year
        for year in range(BASELINE_START, BASELINE_END + 1)
        if year in metadata
    ]

    if not baseline_years:
        raise RuntimeError("No baseline years are available.")

    # Set to False if memory is insufficient.
    use_baseline_cache = True
    baseline_data = {}

    if use_baseline_cache:
        for year in baseline_years:
            try:
                arr, _, _, _, _ = read_year_data(year)
                baseline_data[year] = arr
                print(f"  Cached baseline year {year}: {arr.shape}")
            except Exception as error:
                print(
                    f"  [WARNING] Failed to cache baseline year "
                    f"{year}: {error}"
                )

            gc.collect()

    baseline_p10 = {}

    target_dates = pd.date_range("2001-01-01", "2001-12-31")

    for target_date in target_dates:
        key = month_day_key(target_date)
        anomaly_fields = []

        target_window = {
            (
                (target_date + pd.Timedelta(days=offset)).month,
                (target_date + pd.Timedelta(days=offset)).day,
            )
            for offset in range(-5, 6)
        }

        for year in baseline_years:
            if use_baseline_cache:
                if year not in baseline_data:
                    continue

                arr = baseline_data[year]

            else:
                try:
                    arr, _, _, _, _ = read_year_data(year)
                except Exception as error:
                    print(
                        f"  [WARNING] Failed to read baseline year "
                        f"{year}: {error}"
                    )
                    continue

            year_metadata = metadata[year]

            matched_indices = [
                index
                for date, index in year_metadata["time_to_idx"].items()
                if (date.month, date.day) in target_window
            ]

            for time_index in matched_indices:
                monthday = year_metadata["md"][time_index]

                climatology_field = baseline_climatology.get(monthday)

                if climatology_field is None:
                    continue

                anomalies = (
                    arr[:, time_index]
                    - climatology_field[None, :, :]
                )

                anomaly_fields.append(anomalies.astype(np.float32))

                del anomalies

            if not use_baseline_cache:
                del arr
                gc.collect()

        if not anomaly_fields:
            baseline_p10[key] = None
            print(f"  [WARNING] No data available for P10: {key}")
            continue

        values = np.concatenate(anomaly_fields, axis=0).astype(
            np.float32
        )

        baseline_p10[key] = np.nanpercentile(
            values,
            PERCENTILE_THRESHOLD,
            axis=0,
        ).astype(np.float32)

        del values, anomaly_fields
        gc.collect()

        if target_date.day == 1:
            print(
                f"  P{PERCENTILE_THRESHOLD:.0f} threshold "
                f"completed through {key}"
            )

    if use_baseline_cache:
        del baseline_data
        gc.collect()

    print(
        f"  P{PERCENTILE_THRESHOLD:.0f} threshold complete: "
        f"{len(baseline_p10)} fields"
    )

    return baseline_p10


# ================================================================
# LINEAR TREND COEFFICIENTS
# ================================================================
def build_trend_coefficients(metadata, baseline_climatology):
    """
    Build daily linear trend coefficients from ensemble-mean anomalies.

    Returns a dictionary:
        coeffs[monthday] = (intercept, slope, mean_year)
    """
    print("Building cross-year linear trend coefficients")

    daily_entries = {}

    for year in sorted(metadata.keys()):
        arr, _, _, _, _ = read_year_data(year)

        ensemble_mean = arr.mean(axis=0)

        for time_index, monthday in enumerate(metadata[year]["md"]):
            climatology_field = baseline_climatology.get(monthday)

            if climatology_field is None:
                continue

            anomaly = (
                ensemble_mean[time_index] - climatology_field
            ).astype(np.float32)

            daily_entries.setdefault(monthday, []).append(
                (year, anomaly)
            )

            del anomaly

        del arr, ensemble_mean
        gc.collect()

        print(f"  Trend accumulation complete: {year}")

    coefficients = {}

    for monthday, entries in daily_entries.items():
        if len(entries) < 2:
            continue

        years = np.array(
            [entry[0] for entry in entries],
            dtype=np.float64,
        )

        values = np.stack(
            [entry[1] for entry in entries],
            axis=0,
        ).astype(np.float64)

        mean_year = float(years.mean())
        centered_years = years - mean_year
        denominator = np.sum(centered_years ** 2)

        if denominator == 0:
            del years, values, centered_years
            continue

        slope = (
            centered_years[:, None, None] * values
        ).sum(axis=0) / denominator

        intercept = np.nanmean(values, axis=0)

        coefficients[monthday] = (
            intercept.astype(np.float32),
            slope.astype(np.float32),
            mean_year,
        )

        del years, values, centered_years, slope, intercept

    del daily_entries
    gc.collect()

    print(
        f"  Trend coefficients complete: "
        f"{len(coefficients)} calendar days"
    )

    return coefficients


def compute_detrended_year(
    year,
    metadata,
    baseline_climatology,
    coefficients,
):
    """
    Load one year and calculate linearly detrended daily T2m anomalies.

    Detrending removes only the time-varying trend component:
        anomaly - slope * (year - mean_year)

    This preserves the anomaly reference level used by the raw-anomaly
    P10 cold threshold.
    """
    arr, _, _, _, _ = read_year_data(year)
    detrended = np.empty_like(arr)

    for time_index, monthday in enumerate(metadata[year]["md"]):
        climatology_field = baseline_climatology.get(monthday)

        if climatology_field is None:
            detrended[:, time_index] = np.nan
            continue

        anomalies = (
            arr[:, time_index]
            - climatology_field[None, :, :]
        )

        if monthday in coefficients:
            _, slope, mean_year = coefficients[monthday]

            detrended[:, time_index] = (
                anomalies
                - slope[None, :, :] * (year - mean_year)
            )

        else:
            detrended[:, time_index] = anomalies

        del anomalies

    return arr, detrended


# ================================================================
# EVENT METRICS
# ================================================================
def build_event_cache(
    ssw_df,
    metadata,
    baseline_climatology,
    baseline_p10,
    coefficients,
):
    """
    Calculate cumulative post-SSW cold-anomaly area for each event.

    For days DAY_START to DAY_END:
        - raw metric uses raw T2m anomaly
        - detrended metric uses linearly detrended T2m anomaly
        - both are judged against baseline raw-anomaly P10 threshold

    Unit:
        percent area multiplied by days
    """
    print("Building post-SSW event cache")

    first_year = sorted(metadata.keys())[0]

    region_weights = {
        region_name: cosine_weights_2d(
            metadata[first_year]["lat_vals"],
            metadata[first_year]["lon_vals"],
            region_box,
        )
        for region_name, region_box in REGION_BOXES.items()
    }

    event_cache_raw = {}
    event_cache_detrended = {}

    for year in sorted(metadata.keys()):
        events_this_year = ssw_df[
            ssw_df["init_year"] == year
        ]

        if events_this_year.empty:
            continue

        arr, detrended = compute_detrended_year(
            year,
            metadata,
            baseline_climatology,
            coefficients,
        )

        year_metadata = metadata[year]

        for _, row in events_this_year.iterrows():
            member = int(row["member"])
            onset_date = pd.Timestamp(row["ssw_date"]).normalize()

            if member not in year_metadata["member_to_idx"]:
                continue

            if onset_date not in year_metadata["time_to_idx"]:
                continue

            member_index = year_metadata["member_to_idx"][member]
            onset_index = year_metadata["time_to_idx"][onset_date]

            start_index = onset_index + DAY_START
            end_index = onset_index + DAY_END + 1

            if start_index < 0 or end_index > arr.shape[1]:
                print(
                    f"  [SKIP] {year}; member={member}; "
                    f"{onset_date.date()}; out of range"
                )
                continue

            for region_name in REGION_ORDER:
                lat_mask, lon_mask, weights_2d = (
                    region_weights[region_name]
                )

                raw_sum = 0.0
                detrended_sum = 0.0
                raw_valid_days = 0
                detrended_valid_days = 0

                for time_index in range(start_index, end_index):
                    monthday = year_metadata["md"][time_index]

                    climatology_field = baseline_climatology.get(
                        monthday
                    )
                    threshold_field = baseline_p10.get(monthday)

                    if (
                        climatology_field is None
                        or threshold_field is None
                    ):
                        continue

                    raw_anomaly = (
                        arr[member_index, time_index]
                        - climatology_field
                    )

                    detrended_anomaly = detrended[
                        member_index,
                        time_index,
                    ]

                    raw_fraction = region_area_fraction_below_threshold(
                        raw_anomaly,
                        threshold_field,
                        lat_mask,
                        lon_mask,
                        weights_2d,
                    )

                    detrended_fraction = (
                        region_area_fraction_below_threshold(
                            detrended_anomaly,
                            threshold_field,
                            lat_mask,
                            lon_mask,
                            weights_2d,
                        )
                    )

                    if np.isfinite(raw_fraction):
                        raw_sum += raw_fraction * 100.0
                        raw_valid_days += 1

                    if np.isfinite(detrended_fraction):
                        detrended_sum += detrended_fraction * 100.0
                        detrended_valid_days += 1

                    del raw_anomaly, detrended_anomaly

                event_key = (
                    year,
                    member,
                    onset_date,
                    region_name,
                )

                event_cache_raw[event_key] = (
                    raw_sum if raw_valid_days > 0 else np.nan
                )

                event_cache_detrended[event_key] = (
                    detrended_sum
                    if detrended_valid_days > 0
                    else np.nan
                )

        del arr, detrended
        gc.collect()

        print(f"  Event cache complete: {year}")

    return event_cache_raw, event_cache_detrended


# ================================================================
# SLIDING-WINDOW AGGREGATION
# ================================================================
def append_statistics(values, destination):
    """Append mean, standard error, and sample count."""
    array = np.array(values, dtype=float)
    array = array[np.isfinite(array)]

    n_values = len(array)
    destination["n"].append(n_values)

    if n_values == 0:
        destination["mean"].append(np.nan)
        destination["se"].append(np.nan)

    elif n_values == 1:
        destination["mean"].append(array[0])
        destination["se"].append(np.nan)

    else:
        destination["mean"].append(np.nanmean(array))
        destination["se"].append(
            np.nanstd(array, ddof=1) / np.sqrt(n_values)
        )


def cumulative_cold_area_for_member(
    arr,
    detrended,
    member_index,
    matched_indices,
    metadata_for_year,
    baseline_climatology,
    baseline_p10,
    lat_mask,
    lon_mask,
    weights_2d,
):
    """Calculate raw and detrended cumulative cold-area values."""
    raw_sum = 0.0
    detrended_sum = 0.0
    raw_valid_days = 0
    detrended_valid_days = 0

    for time_index in matched_indices:
        monthday = metadata_for_year["md"][time_index]

        climatology_field = baseline_climatology.get(monthday)
        threshold_field = baseline_p10.get(monthday)

        if climatology_field is None or threshold_field is None:
            continue

        raw_anomaly = (
            arr[member_index, time_index]
            - climatology_field
        )

        detrended_anomaly = detrended[member_index, time_index]

        raw_fraction = region_area_fraction_below_threshold(
            raw_anomaly,
            threshold_field,
            lat_mask,
            lon_mask,
            weights_2d,
        )

        detrended_fraction = region_area_fraction_below_threshold(
            detrended_anomaly,
            threshold_field,
            lat_mask,
            lon_mask,
            weights_2d,
        )

        if np.isfinite(raw_fraction):
            raw_sum += raw_fraction * 100.0
            raw_valid_days += 1

        if np.isfinite(detrended_fraction):
            detrended_sum += detrended_fraction * 100.0
            detrended_valid_days += 1

        del raw_anomaly, detrended_anomaly

    raw_value = raw_sum if raw_valid_days > 0 else np.nan

    detrended_value = (
        detrended_sum
        if detrended_valid_days > 0
        else np.nan
    )

    return raw_value, detrended_value


def aggregate_sliding_with_non_ssw_analogs(
    ssw_df,
    metadata,
    baseline_climatology,
    baseline_p10,
    coefficients,
    event_cache_raw,
    event_cache_detrended,
):
    """
    Aggregate SSW events and matched-date non-SSW analogs.

    For each 10-year window:
        - SSW values are the means over all retained SSW events.
        - Non-SSW analogs use the same calendar-day window as each SSW
          event, sampled from all year-member pairs without an SSW
          in that sliding period.
    """
    first_event_year = int(ssw_df["init_year"].min())
    last_event_year = int(ssw_df["init_year"].max())

    windows = [
        (year, year + SLIDING_WINDOW - 1)
        for year in range(
            first_event_year,
            last_event_year - SLIDING_WINDOW + 2,
        )
    ]

    first_year = sorted(metadata.keys())[0]

    region_weights = {
        region_name: cosine_weights_2d(
            metadata[first_year]["lat_vals"],
            metadata[first_year]["lon_vals"],
            region_box,
        )
        for region_name, region_box in REGION_BOXES.items()
    }

    ssw_pairs = set(
        zip(
            ssw_df["init_year"].astype(int),
            ssw_df["member"].astype(int),
        )
    )

    time_series = {
        region_name: {
            tag: {
                "xc": [],
                "mean": [],
                "se": [],
                "n": [],
            }
            for tag in (
                "raw",
                "det",
                "clim_raw",
                "clim_det",
            )
        }
        for region_name in REGION_ORDER
    }

    for window_start, window_end in windows:
        events_in_window = ssw_df[
            ssw_df["init_year"].between(
                window_start,
                window_end,
            )
        ]

        x_center = window_start + (SLIDING_WINDOW - 1) / 2.0

        non_ssw_pairs = [
            (year, member)
            for year in range(window_start, window_end + 1)
            if year in metadata
            for member in metadata[year]["member_to_idx"]
            if (year, member) not in ssw_pairs
        ]

        print(f"\nWindow {window_start}-{window_end}")
        print(f"  SSW events: {len(events_in_window)}")
        print(f"  Non-SSW year-member pairs: {len(non_ssw_pairs)}")

        window_data = {}

        for year in range(window_start, window_end + 1):
            if year not in metadata:
                continue

            try:
                window_data[year] = compute_detrended_year(
                    year,
                    metadata,
                    baseline_climatology,
                    coefficients,
                )
            except Exception as error:
                print(
                    f"  [WARNING] Could not load {year}: {error}"
                )

            gc.collect()

        for region_name in REGION_ORDER:
            lat_mask, lon_mask, weights_2d = (
                region_weights[region_name]
            )

            raw_event_values = []
            detrended_event_values = []

            for _, row in events_in_window.iterrows():
                event_key = (
                    int(row["init_year"]),
                    int(row["member"]),
                    pd.Timestamp(row["ssw_date"]).normalize(),
                    region_name,
                )

                if event_key in event_cache_raw:
                    raw_event_values.append(
                        event_cache_raw[event_key]
                    )
                    detrended_event_values.append(
                        event_cache_detrended[event_key]
                    )

            time_series[region_name]["raw"]["xc"].append(x_center)
            time_series[region_name]["det"]["xc"].append(x_center)

            append_statistics(
                raw_event_values,
                time_series[region_name]["raw"],
            )

            append_statistics(
                detrended_event_values,
                time_series[region_name]["det"],
            )

            non_ssw_raw_values = []
            non_ssw_detrended_values = []

            for _, event_row in events_in_window.iterrows():
                onset_date = pd.Timestamp(
                    event_row["ssw_date"]
                ).normalize()

                target_calendar_days = {
                    (
                        (onset_date + pd.Timedelta(days=offset)).month,
                        (onset_date + pd.Timedelta(days=offset)).day,
                    )
                    for offset in range(DAY_START, DAY_END + 1)
                }

                for analog_year, analog_member in non_ssw_pairs:
                    if analog_year not in window_data:
                        continue

                    arr, detrended = window_data[analog_year]
                    analog_metadata = metadata[analog_year]

                    if analog_member not in analog_metadata["member_to_idx"]:
                        continue

                    member_index = analog_metadata["member_to_idx"][
                        analog_member
                    ]

                    matched_indices = [
                        index
                        for date, index in analog_metadata[
                            "time_to_idx"
                        ].items()
                        if (date.month, date.day)
                        in target_calendar_days
                    ]

                    if not matched_indices:
                        continue

                    raw_value, detrended_value = (
                        cumulative_cold_area_for_member(
                            arr,
                            detrended,
                            member_index,
                            matched_indices,
                            analog_metadata,
                            baseline_climatology,
                            baseline_p10,
                            lat_mask,
                            lon_mask,
                            weights_2d,
                        )
                    )

                    if np.isfinite(raw_value):
                        non_ssw_raw_values.append(raw_value)

                    if np.isfinite(detrended_value):
                        non_ssw_detrended_values.append(
                            detrended_value
                        )

            time_series[region_name]["clim_raw"]["xc"].append(
                x_center
            )
            time_series[region_name]["clim_det"]["xc"].append(
                x_center
            )

            append_statistics(
                non_ssw_raw_values,
                time_series[region_name]["clim_raw"],
            )

            append_statistics(
                non_ssw_detrended_values,
                time_series[region_name]["clim_det"],
            )

        del window_data
        gc.collect()

        print(f"  Window complete: {window_start}-{window_end}")

    for region_name in REGION_ORDER:
        for tag in ("raw", "det", "clim_raw", "clim_det"):
            for key in ("xc", "mean", "se", "n"):
                time_series[region_name][tag][key] = np.array(
                    time_series[region_name][tag][key]
                )

    return time_series


# ================================================================
# CACHE
# ================================================================
def save_cache(path, time_series):
    """Save sliding-window results to NPZ."""
    output = {}

    for region_name in REGION_ORDER:
        for tag in ("raw", "det", "clim_raw", "clim_det"):
            prefix = f"{region_name}__{tag}"

            output[f"{prefix}__xc"] = (
                time_series[region_name][tag]["xc"]
            )
            output[f"{prefix}__mean"] = (
                time_series[region_name][tag]["mean"]
            )
            output[f"{prefix}__se"] = (
                time_series[region_name][tag]["se"]
            )
            output[f"{prefix}__n"] = (
                time_series[region_name][tag]["n"]
            )

    np.savez(path, **output)

    print(f"Saved cache: {path}")


def load_cache(path):
    """Load sliding-window results from NPZ."""
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    data = np.load(path, allow_pickle=False)

    time_series = {
        region_name: {
            tag: {}
            for tag in ("raw", "det", "clim_raw", "clim_det")
        }
        for region_name in REGION_ORDER
    }

    for region_name in REGION_ORDER:
        for tag in ("raw", "det", "clim_raw", "clim_det"):
            prefix = f"{region_name}__{tag}"

            time_series[region_name][tag]["xc"] = (
                data[f"{prefix}__xc"]
            )
            time_series[region_name][tag]["mean"] = (
                data[f"{prefix}__mean"]
            )
            time_series[region_name][tag]["se"] = (
                data[f"{prefix}__se"]
            )
            time_series[region_name][tag]["n"] = (
                data[f"{prefix}__n"].astype(int)
            )

    print(f"Loaded cache: {path}")

    return time_series


# ================================================================
# TREND TEST
# ================================================================
def block_bootstrap_trend(
    x_values,
    mean_values,
    n_boot=2000,
    block_size=7,
    seed=42,
):
    """Calculate Theil-Sen slope and block-bootstrap p value."""
    random_generator = np.random.default_rng(seed)

    n_values = len(mean_values)

    if n_values < 3:
        return np.nan, np.nan

    observed_slope = theilslopes(
        mean_values,
        x_values,
    ).slope

    bootstrap_slopes = []

    for _ in range(n_boot):
        indices = []

        while len(indices) < n_values:
            start = random_generator.integers(
                0,
                max(1, n_values - block_size + 1),
            )

            indices.extend(
                range(start, min(start + block_size, n_values))
            )

        sample_indices = np.array(indices[:n_values])

        bootstrap_slopes.append(
            theilslopes(
                mean_values[sample_indices],
                x_values,
            ).slope
        )

    bootstrap_slopes = np.array(bootstrap_slopes)
    centered_slopes = bootstrap_slopes - np.mean(bootstrap_slopes)

    p_value = float(
        np.mean(
            np.abs(centered_slopes)
            >= np.abs(observed_slope)
        )
    )

    return observed_slope, p_value


# ================================================================
# PLOT
# ================================================================
def plot_results(time_series):
    """
    Plot raw and detrended post-SSW cold-anomaly area.

    Cached cumulative values are divided by N_DAYS only for plotting,
    so the figure displays mean daily cold-area fraction in percent.
    """
    import matplotlib.lines as mlines

    plot_data = {}

    for region_name in REGION_ORDER:
        plot_data[region_name] = {}

        for tag in ("raw", "det", "clim_raw", "clim_det"):
            plot_data[region_name][tag] = {
                "xc": time_series[region_name][tag]["xc"].copy(),
                "mean": (
                    time_series[region_name][tag]["mean"].copy()
                    / N_DAYS
                ),
                "se": (
                    time_series[region_name][tag]["se"].copy()
                    / N_DAYS
                ),
                "n": time_series[region_name][tag]["n"].copy(),
            }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=False,
    )

    panel_info = (
        ("raw", "clim_raw", "Raw T2m anomaly"),
        ("det", "clim_det", "Detrended T2m anomaly"),
    )

    panel_labels = ("(a)", "(b)")

    for panel_index, (tag, climatology_tag, title) in enumerate(
        panel_info
    ):
        ax = axes[panel_index]
        trend_lines = {
            region_name: {}
            for region_name in REGION_ORDER
        }

        # ------------------------------------------------------------
        # Non-SSW matched-date analogs
        # ------------------------------------------------------------
        for region_name in REGION_ORDER:
            color = REGION_COLORS[region_name]

            x_values = plot_data[region_name][climatology_tag]["xc"]
            means = plot_data[region_name][climatology_tag]["mean"]
            errors = plot_data[region_name][climatology_tag]["se"]

            valid = np.isfinite(means)

            x_valid = x_values[valid]
            mean_valid = means[valid]
            error_valid = errors[valid]

            if len(x_valid) == 0:
                continue

            ax.plot(
                x_valid,
                mean_valid,
                color=color,
                lw=1.6,
                ls="--",
                alpha=0.65,
                zorder=1,
            )

            good_errors = np.isfinite(error_valid)

            if np.any(good_errors):
                ax.fill_between(
                    x_valid[good_errors],
                    (mean_valid - error_valid)[good_errors],
                    (mean_valid + error_valid)[good_errors],
                    color=color,
                    alpha=0.08,
                    zorder=1,
                )

            if len(x_valid) >= 3:
                trend_result = theilslopes(
                    mean_valid,
                    x_valid,
                    method="joint",
                )

                _, p_value = block_bootstrap_trend(
                    x_valid,
                    mean_valid,
                    n_boot=2000,
                    block_size=7,
                )

                significance = (
                    "**"
                    if p_value < 0.01
                    else "*" if p_value < 0.05 else ""
                )

                ax.plot(
                    x_valid,
                    trend_result.intercept
                    + trend_result.slope * x_valid,
                    color=color,
                    lw=1.2,
                    ls=":",
                    alpha=0.75,
                    zorder=4,
                )

                trend_lines[region_name]["clim"] = (
                    trend_result.slope,
                    significance,
                )

        # ------------------------------------------------------------
        # Post-SSW events
        # ------------------------------------------------------------
        for region_name in REGION_ORDER:
            color = REGION_COLORS[region_name]

            x_values = plot_data[region_name][tag]["xc"]
            means = plot_data[region_name][tag]["mean"]
            errors = plot_data[region_name][tag]["se"]

            valid = np.isfinite(means)

            x_valid = x_values[valid]
            mean_valid = means[valid]
            error_valid = errors[valid]

            if len(x_valid) == 0:
                continue

            good_errors = np.isfinite(error_valid)

            if np.any(good_errors):
                ax.fill_between(
                    x_valid[good_errors],
                    (mean_valid - error_valid)[good_errors],
                    (mean_valid + error_valid)[good_errors],
                    color=color,
                    alpha=0.22,
                    zorder=2,
                )

            ax.plot(
                x_valid,
                mean_valid,
                color=color,
                lw=2.4,
                alpha=0.92,
                zorder=3,
                label=REGION_LABELS[region_name],
            )

            if len(x_valid) >= 3:
                trend_result = theilslopes(
                    mean_valid,
                    x_valid,
                    method="joint",
                )

                _, p_value = block_bootstrap_trend(
                    x_valid,
                    mean_valid,
                    n_boot=2000,
                    block_size=7,
                )

                significance = (
                    "**"
                    if p_value < 0.01
                    else "*" if p_value < 0.05 else ""
                )

                ax.plot(
                    x_valid,
                    trend_result.intercept
                    + trend_result.slope * x_valid,
                    color=color,
                    lw=1.2,
                    ls=":",
                    alpha=0.75,
                    zorder=4,
                )

                trend_lines[region_name]["ssw"] = (
                    trend_result.slope,
                    significance,
                )

        # ------------------------------------------------------------
        # Trend table
        # ------------------------------------------------------------
        short_names = {
            "NorthAmerica": "N. America",
            "Europe": "Europe",
            "EastAsia": "East Asia",
        }

        lines_text = [
            "Theil-Sen slope\n(% decade$^{-1}$)",
            ("Region", "SSW", "Non-SSW"),
        ]
        lines_color = ["#888888", "#444444"]
        lines_weight = ["normal", "bold"]

        for region_name in REGION_ORDER:
            ssw_slope, ssw_sig = trend_lines[region_name].get(
                "ssw",
                (np.nan, ""),
            )

            clim_slope, clim_sig = trend_lines[region_name].get(
                "clim",
                (np.nan, ""),
            )

            ssw_text = (
                f"{ssw_slope * 10:+.2f}{ssw_sig}"
                if np.isfinite(ssw_slope)
                else "—"
            )

            clim_text = (
                f"{clim_slope * 10:+.2f}{clim_sig}"
                if np.isfinite(clim_slope)
                else "—"
            )

            lines_text.append(
                (
                    short_names[region_name],
                    ssw_text,
                    clim_text,
                )
            )
            lines_color.append(REGION_COLORS[region_name])
            lines_weight.append("normal")

        title_fontsize = 9.0
        data_fontsize = 10.0

        x_left = 0.03
        y_bottom = 0.03
        line_height = 0.068
        n_lines = len(lines_text)

        column_positions = (
            x_left + 0.01,
            x_left + 0.24,
            x_left + 0.37,
        )

        box_width = 0.40
        box_height = line_height * n_lines + 0.06

        ax.add_patch(
            plt.Rectangle(
                (x_left - 0.015, y_bottom - 0.015),
                box_width,
                box_height,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="#AAAAAA",
                lw=1.0,
                alpha=0.93,
                zorder=8,
                clip_on=False,
            )
        )

        for index, (text, color, fontweight) in enumerate(
            zip(lines_text, lines_color, lines_weight)
        ):
            y_position = (
                y_bottom
                + (n_lines - 1 - index) * line_height
                + 0.02
            )

            fontsize = (
                title_fontsize
                if index == 0
                else data_fontsize
            )

            if index == 0:
                ax.text(
                    x_left + box_width / 2 - 0.015,
                    y_position,
                    text,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color=color,
                    zorder=9,
                )

            else:
                region_text, ssw_text, clim_text = text

                ax.text(
                    column_positions[0],
                    y_position,
                    region_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color=color,
                    zorder=9,
                )

                ax.text(
                    column_positions[1],
                    y_position,
                    ssw_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color=color,
                    zorder=9,
                )

                ax.text(
                    column_positions[2],
                    y_position,
                    clim_text,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color=color,
                    zorder=9,
                )

        # ------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------
        ax.axhline(
            0,
            color="black",
            lw=0.8,
            alpha=0.40,
        )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold",
            pad=8,
        )

        ax.set_xlabel(
            "10-year sliding window",
            fontsize=11,
            fontweight="bold",
        )

        ax.grid(
            True,
            ls="--",
            lw=0.6,
            alpha=0.30,
        )

        if panel_index == 0:
            ax.set_ylabel("Mean cold-anomaly area (%)")
        else:
            ax.set_ylabel("")

        all_values = []

        for region_name in REGION_ORDER:
            for current_tag in (tag, climatology_tag):
                values = plot_data[region_name][current_tag]["mean"]
                all_values.extend(
                    values[np.isfinite(values)].tolist()
                )

        if all_values:
            ymax = np.nanmax(all_values)
            ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)

        reference_x = time_series[REGION_ORDER[0]][tag]["xc"]
        reference_mean = time_series[REGION_ORDER[0]][tag]["mean"]

        valid_reference = np.isfinite(reference_mean)

        if np.any(valid_reference):
            used_x = reference_x[valid_reference]

            first_start = int(
                round(
                    used_x[0]
                    - (SLIDING_WINDOW - 1) / 2.0
                )
            )

            last_start = int(
                round(
                    used_x[-1]
                    - (SLIDING_WINDOW - 1) / 2.0
                )
            )

            starts = np.arange(first_start, last_start + 1, 5)

            ticks = starts + (SLIDING_WINDOW - 1) / 2.0
            labels = [
                f"{start}-{start + SLIDING_WINDOW - 1}"
                for start in starts
            ]

            ax.set_xticks(ticks)
            ax.set_xticklabels(
                labels,
                rotation=35,
                ha="right",
                fontsize=9,
            )

            ax.set_xlim(used_x[0] - 0.5, used_x[-1] + 0.5)

        if panel_index == 0:
            style_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color="#555555",
                    lw=2.4,
                    ls="-",
                    label="Post-SSW",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#555555",
                    lw=1.6,
                    ls="--",
                    label="Non-SSW",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#555555",
                    lw=1.2,
                    ls=":",
                    label="Theil-Sen trend",
                ),
            ]

            region_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=REGION_COLORS[region_name],
                    lw=2.4,
                    label=REGION_LABELS[region_name],
                )
                for region_name in REGION_ORDER
            ]

            ax.legend(
                handles=style_handles + region_handles,
                fontsize=8.5,
                loc="upper right",
                framealpha=0.92,
                edgecolor="#CCCCCC",
                frameon=True,
                ncol=1,
                handlelength=2.2,
                labelspacing=0.45,
            )

        ax.text(
            0.02,
            0.98,
            panel_labels[panel_index],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 1.5,
            },
            zorder=10,
        )

    plt.suptitle(
        "Post-SSW mean cold-anomaly area",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(PDF_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {PDF_PATH}")


# ================================================================
# MAIN
# ================================================================
def main():
    """Run the Figure 4 analysis."""
    print("=" * 70)
    print("Figure 4: post-SSW cold-anomaly area")
    print(f"Output figure: {PDF_PATH}")
    print("=" * 70)

    ssw_df = read_ssw_events()

    if NPZ_PATH.exists():
        print("\nCache found. Loading cached data.")
        time_series = load_cache(NPZ_PATH)

    else:
        print("\nScanning daily-file metadata.")
        metadata = scan_all_years()

        baseline_climatology = build_baseline_climatology(
            metadata
        )

        baseline_p10 = build_baseline_p10_threshold(
            metadata,
            baseline_climatology,
        )

        coefficients = build_trend_coefficients(
            metadata,
            baseline_climatology,
        )

        event_cache_raw, event_cache_detrended = build_event_cache(
            ssw_df,
            metadata,
            baseline_climatology,
            baseline_p10,
            coefficients,
        )

        time_series = aggregate_sliding_with_non_ssw_analogs(
            ssw_df,
            metadata,
            baseline_climatology,
            baseline_p10,
            coefficients,
            event_cache_raw,
            event_cache_detrended,
        )

        save_cache(NPZ_PATH, time_series)

        del (
            metadata,
            baseline_climatology,
            baseline_p10,
            coefficients,
            event_cache_raw,
            event_cache_detrended,
        )
        gc.collect()

    plot_results(time_series)

    del ssw_df, time_series
    gc.collect()

    print("\nFinished.")


if __name__ == "__main__":
    main()
