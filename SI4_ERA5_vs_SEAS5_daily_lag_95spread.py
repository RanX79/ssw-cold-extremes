# -*- coding: utf-8 -*-
"""
Daily evolution by two periods:
ERA5 regional lag composite vs SEAS5 95% member spread

Periods:
    1981–1990
    2015–2024

Layout:
    4 rows × 3 columns

Rows:
    Raw 1981–1990
    Raw 2015–2024
    Detrended 1981–1990
    Detrended 2015–2024

Columns:
    North America / Europe / East Asia

SEAS5 spread:
    For each period, region, lag:
        1. average events within each member
        2. obtain 25 member-mean composites
        3. calculate 2.5–97.5 percentile across 25 members

ERA5:
    event-mean composite for each period
"""

import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from pathlib import Path


plt.rcParams["font.family"] = "Arial"


# ================================================================
# PATHS
# ================================================================
SEAS5_SSW_CSV_PATH = Path(
    r"path/to/your/data/SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv"
)

ERA5_SSW_CSV_PATH = Path(
    r"path/to/your/data/ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv"
)

SEAS5_T2M_DIR = Path(r"path/to/your/data/IFS_t2m_daily")

ERA5_T2M_FILE = Path(
    r"path/to/your/data/ERA5_t2m_daily_1940_2024_no229.nc"
)

OUTPUT_DIR = Path(r"path/to/your/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUTPUT_DIR / "SI4_ERA5_vs_SEAS5_daily_lag_twoPeriods_95memberSpread_cache.npz"
OUT_FIG = OUTPUT_DIR / "SI4_ERA5_vs_SEAS5_daily_lag_twoPeriods_95memberSpread.pdf"


# ================================================================
# SETTINGS
# ================================================================
START_YEAR = 1981
END_YEAR = 2024

BASELINE_START = 1981
BASELINE_END = 2010

N_MEMBERS = 25
LAGS = np.arange(-20, 61)

T2M_VAR_CANDIDATES_ERA5 = ["t2m", "2m_temperature"]
T2M_VAR_CANDIDATES_SEAS5 = ["2m_temperature", "t2m"]

PERIODS = {
    "p1": {"years": list(range(1981, 1991)), "label": "1981–1990"},
    "p2": {"years": list(range(2015, 2025)), "label": "2015–2024"},
}

PERIOD_ORDER = ["p1", "p2"]

REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": -140, "lon_max": -60},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min":    0, "lon_max":  40},
    "EastAsia":     {"lat_min": 45, "lat_max": 70, "lon_min":   60, "lon_max": 120},
}

REGION_TITLES = {
    "NorthAmerica": "North America",
    "Europe": "Europe",
    "East Asia": "East Asia",
    "EastAsia": "East Asia",
}

REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]


# ================================================================
# HELPERS
# ================================================================
def get_var_name(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify variable from {list(ds.data_vars)}")


def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c:
        raise ValueError(f"No lat dim: {da.dims}")
    if not lon_c:
        raise ValueError(f"No lon dim: {da.dims}")
    return lat_c[0], lon_c[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time"]:
        if name in da.dims:
            return name
    raise ValueError(f"No time dim: {da.dims}")


def safe_member_label(number_coord, idx):
    try:
        return int(number_coord[idx])
    except Exception:
        return idx + 1


def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def cosine_weighted_region_mean(field2d, lat_vals, lon_vals, rb):
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    lon_mask = (lon_vals >= rb["lon_min"]) & (lon_vals <= rb["lon_max"])

    sub = field2d[np.ix_(lat_mask, lon_mask)]

    weights = np.cos(np.deg2rad(lat_vals[lat_mask]))
    w2d = weights[:, None] * np.ones(lon_mask.sum())

    valid = np.isfinite(sub)

    if valid.sum() == 0:
        return np.nan

    return float(np.nansum(sub * w2d * valid) / np.nansum(w2d * valid))


def period_key_from_year(year):
    for pk in PERIOD_ORDER:
        if year in PERIODS[pk]["years"]:
            return pk
    return None


# ================================================================
# LOAD ERA5
# ================================================================
def load_era5_one_year(year, extend_to_may=True):
    if extend_to_may:
        t0 = f"{year}-01-01"
        t1 = f"{year + 1}-05-31"
    else:
        t0 = f"{year}-01-01"
        t1 = f"{year}-12-31"

    ds = xr.open_dataset(ERA5_T2M_FILE)
    var = get_var_name(ds, T2M_VAR_CANDIDATES_ERA5)
    da = ds[var]

    time_name = get_time_dim_name(da)
    da = da.sel({time_name: slice(t0, t1)})

    for dim in list(da.dims):
        if dim != time_name and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)

    lat_name, lon_name = get_lat_lon_dim_names(da)

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    lon0 = da[lon_name].values

    if np.nanmax(lon0) > 180:
        lon_new = np.where(lon0 > 180, lon0 - 360, lon0)
        sort_idx = np.argsort(lon_new)
        da = da.isel({lon_name: sort_idx})
        da = da.assign_coords({lon_name: lon_new[sort_idx]})

    da = da.load().astype(np.float32)

    if float(da.mean()) > 100:
        da = da - 273.15

    times = pd.to_datetime(da[time_name].values).normalize()

    # ERA5 文件本身是 no229，这里再保险删一次
    mask = ~((times.month == 2) & (times.day == 29))
    da = da.isel({time_name: mask})
    times = times[mask]

    out = {
        "t2m": da.values.astype(np.float32)[None, ...],
        "times": times,
        "monthday": np.array([month_day_key(t) for t in times]),
        "member_to_idx": {0: 0},
        "time_to_idx": {pd.Timestamp(t): i for i, t in enumerate(times)},
        "lat_vals": da[lat_name].values.copy(),
        "lon_vals": da[lon_name].values.copy(),
    }

    ds.close()
    del ds, da
    gc.collect()

    print(f"  Loaded ERA5 {year}: {t0} to {t1}, shape={out['t2m'].shape}")
    return out


# ================================================================
# LOAD SEAS5
# ================================================================
def load_seas5_one_year(year):
    fp = SEAS5_T2M_DIR / f"SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"

    if not fp.exists():
        print(f"  Missing SEAS5 file: {fp}")
        return None

    ds = xr.open_dataset(fp)
    var = get_var_name(ds, T2M_VAR_CANDIDATES_SEAS5)
    da = ds[var].load()
    ds.close()

    if "number" not in da.dims:
        raise ValueError(f"Missing number dim in SEAS5: {da.dims}")

    da = da.transpose("number", *[d for d in da.dims if d != "number"])
    da = da.isel(number=slice(0, N_MEMBERS))

    lat_name, lon_name = get_lat_lon_dim_names(da)
    time_name = get_time_dim_name(da)

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    lon0 = da[lon_name].values

    if np.nanmax(lon0) > 180:
        lon_new = np.where(lon0 > 180, lon0 - 360, lon0)
        sort_idx = np.argsort(lon_new)
        da = da.isel({lon_name: sort_idx})
        da = da.assign_coords({lon_name: lon_new[sort_idx]})

    if float(da.mean()) > 100:
        da = da - 273.15

    member_labels = [
        safe_member_label(da["number"].values, i)
        for i in range(da.sizes["number"])
    ]

    member_to_idx = {m: i for i, m in enumerate(member_labels)}

    for i in range(len(member_labels)):
        member_to_idx[i] = i
        member_to_idx[i + 1] = i

    times = pd.to_datetime(da[time_name].values).normalize()

    # 删除 2月29日，和 no229 ERA5 / 主图保持一致
    mask = ~((times.month == 2) & (times.day == 29))
    da = da.isel({time_name: mask})
    times = times[mask]

    out = {
        "t2m": da.values.astype(np.float32),
        "times": times,
        "monthday": np.array([month_day_key(t) for t in times]),
        "member_labels": member_labels,
        "member_to_idx": member_to_idx,
        "time_to_idx": {pd.Timestamp(t): i for i, t in enumerate(times)},
        "lat_vals": da[lat_name].values.copy(),
        "lon_vals": da[lon_name].values.copy(),
    }

    del da
    gc.collect()

    print(f"  Loaded SEAS5 {year}: shape={out['t2m'].shape}")
    return out


# ================================================================
# READ EVENTS
# ================================================================
def read_era5_ssw():
    df = pd.read_csv(ERA5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()

    df["member"] = 0
    df = df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)

    print(f"ERA5 SSW events: {len(df)}")
    return df


def read_seas5_ssw():
    df = pd.read_csv(SEAS5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()

    df["member"] = df["member"].astype(int)
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)

    print(f"SEAS5 SSW events: {len(df)}")
    return df


# ================================================================
# BASELINE CLIMATOLOGY
# ================================================================
def build_smoothed_baseline(load_func, label):
    print(f"\nBuilding {label} smoothed baseline climatology {BASELINE_START}-{BASELINE_END}...")

    doy_to_fields = {}
    lat_vals = lon_vals = None

    for year in range(BASELINE_START, BASELINE_END + 1):
        if label == "ERA5":
            data = load_func(year, extend_to_may=False)
        else:
            data = load_func(year)

        if data is None:
            continue

        if lat_vals is None:
            lat_vals = data["lat_vals"]
            lon_vals = data["lon_vals"]

        ens_mean = data["t2m"].mean(axis=0)
        doys = pd.to_datetime(data["times"]).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(int(doy), []).append(
                ens_mean[i].astype(np.float32)
            )

        del data, ens_mean, doys
        gc.collect()

    if lat_vals is None:
        raise RuntimeError(f"No data found for {label} baseline.")

    nlat = len(lat_vals)
    nlon = len(lon_vals)

    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        stacked = np.stack(fields, axis=0)
        clim_raw[doy - 1] = np.nanmean(stacked, axis=0).astype(np.float32)
        del stacked

    del doy_to_fields
    gc.collect()

    window = 11
    pad = window // 2

    clim_pad = np.concatenate(
        [
            clim_raw[-pad:],
            clim_raw,
            clim_raw[:pad],
        ],
        axis=0
    )

    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i + window]

        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win, axis=0).astype(np.float32)

    del clim_pad, clim_raw
    gc.collect()

    # 用非闰年 2001 映射，保证没有 02-29
    dates_ref = pd.date_range("2001-01-01", "2001-12-31")
    clim = {}

    for d in dates_ref:
        clim[month_day_key(d)] = clim_smooth[d.dayofyear - 1].astype(np.float32)

    del clim_smooth
    gc.collect()

    print(f"  {label} baseline done: {len(clim)} keys")
    return clim, lat_vals, lon_vals


# ================================================================
# TREND PARAMS
# ================================================================
def build_trend_params(load_func, baseline_clim, label):
    print(f"\nBuilding {label} detrending params...")

    md_ens_anom = {}

    for year in range(START_YEAR, END_YEAR + 1):
        if label == "ERA5":
            data = load_func(year, extend_to_may=False)
        else:
            data = load_func(year)

        if data is None:
            continue

        ens_mean = data["t2m"].mean(axis=0)

        for i, t in enumerate(data["times"]):
            key_str = month_day_key(t)
            cf = baseline_clim.get(key_str)

            if cf is None:
                continue

            key = (pd.Timestamp(t).month, pd.Timestamp(t).day)

            md_ens_anom.setdefault(key, []).append(
                (year, (ens_mean[i] - cf).astype(np.float32))
            )

        del data, ens_mean
        gc.collect()

    trend_params = {}

    for key, entries in md_ens_anom.items():
        if len(entries) < 2:
            continue

        yr_arr = np.array([e[0] for e in entries], dtype=np.float64)
        yr_mean = yr_arr.mean()
        yr_c = yr_arr - yr_mean
        denom = np.sum(yr_c ** 2)

        if denom < 1e-10:
            continue

        stack = np.stack([e[1] for e in entries], axis=0).astype(np.float64)

        b = np.sum(yr_c[:, None, None] * stack, axis=0) / denom
        a = np.nanmean(stack, axis=0)

        trend_params[key] = (
            a.astype(np.float32),
            b.astype(np.float32),
            float(yr_mean)
        )

        del yr_arr, yr_c, stack, a, b
        gc.collect()

    del md_ens_anom
    gc.collect()

    print(f"  {label} trend params done: {len(trend_params)} keys")
    return trend_params


# ================================================================
# ERA5 LAG SERIES BY PERIOD
# ================================================================
def calc_era5_lag_series_by_period(ssw_df, baseline_clim, trend_params):
    print("\nCalculating ERA5 daily lag series by period...")

    store = {
        pk: {
            "raw": {rn: {lag: [] for lag in LAGS} for rn in REGION_ORDER},
            "det": {rn: {lag: [] for lag in LAGS} for rn in REGION_ORDER},
        }
        for pk in PERIOD_ORDER
    }

    for year in range(START_YEAR, END_YEAR + 1):
        pk = period_key_from_year(year)

        if pk is None:
            continue

        dfy = ssw_df[ssw_df["init_year"] == year]

        if dfy.empty:
            continue

        data = load_era5_one_year(year, extend_to_may=True)

        t2m = data["t2m"]
        nT, nlat, nlon = t2m.shape[1], t2m.shape[2], t2m.shape[3]

        clim_arr = np.full((nT, nlat, nlon), np.nan, dtype=np.float32)

        for i, t in enumerate(data["times"]):
            cf = baseline_clim.get(month_day_key(t))
            if cf is not None:
                clim_arr[i] = cf

        raw = t2m - clim_arr[None]
        det = raw.copy()

        for i, t in enumerate(data["times"]):
            key = (pd.Timestamp(t).month, pd.Timestamp(t).day)
            coeff = trend_params.get(key)

            if coeff is None:
                continue

            a, b, yr_mean = coeff
            this_year = pd.Timestamp(t).year
            det[:, i] -= (a + b * (this_year - yr_mean))[None]

        for _, row in dfy.iterrows():
            center = pd.Timestamp(row["ssw_date"]).normalize()

            if center not in data["time_to_idx"]:
                continue

            c_idx = data["time_to_idx"][center]

            for lag in LAGS:
                t_idx = c_idx + lag

                if t_idx < 0 or t_idx >= nT:
                    continue

                for rn, rb in REGION_BOXES.items():
                    rv = cosine_weighted_region_mean(
                        raw[0, t_idx],
                        data["lat_vals"],
                        data["lon_vals"],
                        rb
                    )

                    dv = cosine_weighted_region_mean(
                        det[0, t_idx],
                        data["lat_vals"],
                        data["lon_vals"],
                        rb
                    )

                    if np.isfinite(rv):
                        store[pk]["raw"][rn][lag].append(rv)

                    if np.isfinite(dv):
                        store[pk]["det"][rn][lag].append(dv)

        del data, t2m, clim_arr, raw, det
        gc.collect()

    result = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    counts = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    for pk in PERIOD_ORDER:
        for tag in ["raw", "det"]:
            for rn in REGION_ORDER:
                result[pk][tag][rn] = np.array(
                    [
                        np.nanmean(store[pk][tag][rn][lag])
                        if len(store[pk][tag][rn][lag]) > 0
                        else np.nan
                        for lag in LAGS
                    ],
                    dtype=np.float32
                )

                counts[pk][tag][rn] = np.array(
                    [
                        len(store[pk][tag][rn][lag])
                        for lag in LAGS
                    ],
                    dtype=np.int16
                )

    return result, counts


# ================================================================
# SEAS5 MEMBER LAG SERIES BY PERIOD
# ================================================================
def calc_seas5_member_lag_series_by_period(ssw_df, baseline_clim, trend_params):
    print("\nCalculating SEAS5 member daily lag series by period...")

    store = {
        pk: {
            "raw": {
                rn: {m: {lag: [] for lag in LAGS} for m in range(N_MEMBERS)}
                for rn in REGION_ORDER
            },
            "det": {
                rn: {m: {lag: [] for lag in LAGS} for m in range(N_MEMBERS)}
                for rn in REGION_ORDER
            },
        }
        for pk in PERIOD_ORDER
    }

    for year in range(START_YEAR, END_YEAR + 1):
        pk = period_key_from_year(year)

        if pk is None:
            continue

        dfy = ssw_df[ssw_df["init_year"] == year]

        if dfy.empty:
            continue

        data = load_seas5_one_year(year)

        if data is None:
            continue

        t2m = data["t2m"]
        nT, nlat, nlon = t2m.shape[1], t2m.shape[2], t2m.shape[3]

        clim_arr = np.full((nT, nlat, nlon), np.nan, dtype=np.float32)

        for i, t in enumerate(data["times"]):
            cf = baseline_clim.get(month_day_key(t))

            if cf is not None:
                clim_arr[i] = cf

        raw = t2m - clim_arr[None]
        det = raw.copy()

        for i, t in enumerate(data["times"]):
            key = (pd.Timestamp(t).month, pd.Timestamp(t).day)
            coeff = trend_params.get(key)

            if coeff is None:
                continue

            a, b, yr_mean = coeff
            det[:, i] -= (a + b * (year - yr_mean))[None]

        for _, row in dfy.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()

            if member not in data["member_to_idx"]:
                continue

            if center not in data["time_to_idx"]:
                continue

            m_idx = data["member_to_idx"][member]
            c_idx = data["time_to_idx"][center]

            for lag in LAGS:
                t_idx = c_idx + lag

                if t_idx < 0 or t_idx >= nT:
                    continue

                for rn, rb in REGION_BOXES.items():
                    rv = cosine_weighted_region_mean(
                        raw[m_idx, t_idx],
                        data["lat_vals"],
                        data["lon_vals"],
                        rb
                    )

                    dv = cosine_weighted_region_mean(
                        det[m_idx, t_idx],
                        data["lat_vals"],
                        data["lon_vals"],
                        rb
                    )

                    if np.isfinite(rv):
                        store[pk]["raw"][rn][m_idx][lag].append(rv)

                    if np.isfinite(dv):
                        store[pk]["det"][rn][m_idx][lag].append(dv)

        del data, t2m, clim_arr, raw, det
        gc.collect()

    result = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    member_event_counts = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    for pk in PERIOD_ORDER:
        for tag in ["raw", "det"]:
            for rn in REGION_ORDER:
                arr = np.full((N_MEMBERS, len(LAGS)), np.nan, dtype=np.float32)
                count_arr = np.zeros((N_MEMBERS, len(LAGS)), dtype=np.int16)

                for m in range(N_MEMBERS):
                    for j, lag in enumerate(LAGS):
                        vals = store[pk][tag][rn][m][lag]

                        if len(vals) > 0:
                            # 关键：每个 member 内部先对该 period 的 events 平均
                            arr[m, j] = np.nanmean(vals)

                        count_arr[m, j] = len(vals)

                result[pk][tag][rn] = arr
                member_event_counts[pk][tag][rn] = count_arr

    return result, member_event_counts


# ================================================================
# SAVE / LOAD CACHE
# ================================================================
def save_cache(npz_path, era5, seas5, era5_counts, seas5_counts):
    save_dict = {"lags": LAGS}

    for pk in PERIOD_ORDER:
        for tag in ["raw", "det"]:
            for rn in REGION_ORDER:
                save_dict[f"era5__{pk}__{tag}__{rn}"] = era5[pk][tag][rn]
                save_dict[f"seas5__{pk}__{tag}__{rn}"] = seas5[pk][tag][rn]
                save_dict[f"era5_counts__{pk}__{tag}__{rn}"] = era5_counts[pk][tag][rn]
                save_dict[f"seas5_counts__{pk}__{tag}__{rn}"] = seas5_counts[pk][tag][rn]

    np.savez(npz_path, **save_dict)
    print(f"Saved cache: {npz_path}")


def load_cache(npz_path):
    data = np.load(npz_path, allow_pickle=False)

    era5 = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    seas5 = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    era5_counts = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    seas5_counts = {
        pk: {"raw": {}, "det": {}}
        for pk in PERIOD_ORDER
    }

    for pk in PERIOD_ORDER:
        for tag in ["raw", "det"]:
            for rn in REGION_ORDER:
                era5[pk][tag][rn] = data[f"era5__{pk}__{tag}__{rn}"]
                seas5[pk][tag][rn] = data[f"seas5__{pk}__{tag}__{rn}"]
                era5_counts[pk][tag][rn] = data[f"era5_counts__{pk}__{tag}__{rn}"]
                seas5_counts[pk][tag][rn] = data[f"seas5_counts__{pk}__{tag}__{rn}"]

    print(f"Loaded cache: {npz_path}")

    return era5, seas5, era5_counts, seas5_counts


# ================================================================
# PLOT
# ================================================================
def plot_fig(era5, seas5):
    print(f"\nPlotting → {OUT_FIG}")
    PERIOD_COLORS = {
        "p1": "#1C6AB1",  # 1981–1990
        "p2": "#ED4043",  # 2015–2024
    }

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

    fig, axes = plt.subplots(
        4, 3,
        figsize=(17.5, 12.0),
        sharex=True,
        sharey=True,
        constrained_layout=False
    )

    row_info = [
        ("raw", "p1", f"{PERIODS['p1']['label']}\nRaw"),
        ("raw", "p2", f"{PERIODS['p2']['label']}\nRaw"),
        ("det", "p1", f"{PERIODS['p1']['label']}\nDetrended"),
        ("det", "p2", f"{PERIODS['p2']['label']}\nDetrended"),
    ]

    panel_labels = [
        "(a)", "(b)", "(c)",
        "(d)", "(e)", "(f)",
        "(g)", "(h)", "(i)",
        "(j)", "(k)", "(l)",
    ]

    all_vals = []

    for tag in ["raw", "det"]:
        for pk in PERIOD_ORDER:
            for rn in REGION_ORDER:
                member_series = seas5[pk][tag][rn]
                all_vals.append(np.nanpercentile(member_series, 2.5, axis=0))
                all_vals.append(np.nanpercentile(member_series, 97.5, axis=0))
                all_vals.append(era5[pk][tag][rn])

    all_vals = np.concatenate([np.asarray(v).ravel() for v in all_vals])
    all_vals = all_vals[np.isfinite(all_vals)]

    ymax = np.nanpercentile(np.abs(all_vals), 98)
    ymax = max(1.5, np.ceil(ymax * 2) / 2)
    ymax = min(ymax, 4.0)

    for r, (tag, pk, row_label) in enumerate(row_info):
        for c, rn in enumerate(REGION_ORDER):
            ax = axes[r, c]
            col = PERIOD_COLORS[pk]
            member_series = seas5[pk][tag][rn]

            seas5_mean = np.nanmean(member_series, axis=0)
            seas5_low = np.nanpercentile(member_series, 2.5, axis=0)
            seas5_high = np.nanpercentile(member_series, 97.5, axis=0)

            era5_series = era5[pk][tag][rn]

            outside = (
                (era5_series < seas5_low) |
                (era5_series > seas5_high)
            ) & np.isfinite(era5_series)

            ax.fill_between(
                LAGS,
                seas5_low,
                seas5_high,
                color=col,
                alpha=0.18,
                linewidth=0,
                zorder=1
            )

            ax.plot(
                LAGS,
                seas5_low,
                color=col,
                lw=0.8,
                alpha=0.9,
                zorder=2
            )

            ax.plot(
                LAGS,
                seas5_high,
                color=col,
                lw=0.8,
                alpha=0.9,
                zorder=2
            )


            ax.plot(
                LAGS,
                seas5_mean,
                color=col,
                lw=2.2,
                label="SEAS5 mean",
                zorder=4
            )

            ax.plot(
                LAGS,
                era5_series,
                color="black",
                lw=2.0,
                label="ERA5",
                zorder=5
            )

            ax.scatter(
                LAGS[outside],
                era5_series[outside],
                marker="o",
                s=28,
                facecolors="white",
                edgecolors="black",
                linewidths=1.1,
                zorder=7
            )

            ax.axhline(0, color="0.2", lw=0.9, zorder=0)
            ax.axvline(0, color="0.2", lw=0.9, ls="--", alpha=0.7, zorder=0)

            ax.grid(True, ls="--", alpha=0.25, lw=0.6)

            ax.set_xlim(LAGS[0], LAGS[-1])
            ax.set_ylim(-ymax, ymax)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("black")

            ax.text(
                0.02,
                0.95,
                panel_labels[r * 3 + c],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold"
            )

            if r == 0:
                ax.set_title(
                    REGION_TITLES[rn],
                    fontsize=16,
                    fontweight="bold",
                    pad=8
                )

            if c == 0:
                ax.set_ylabel(
                    f"{row_label}\nT2m anomaly (K)",
                    fontsize=14,
                    fontweight="bold"
                )

            if r == 3:
                ax.set_xlabel(
                    "Lag (days relative to SSW onset)",
                    fontsize=14,
                    fontweight="bold"
                )

            ax.tick_params(labelsize=12)

    legend_handles = [
        mlines.Line2D([], [], color="#1C6AB1", lw=2.2, label="SEAS5 mean, 1981–1990"),
        mlines.Line2D([], [], color="#ED4043", lw=2.2, label="SEAS5 mean, 2015–2024"),
        mlines.Line2D([], [], color="black", lw=2.0, label="ERA5"),
        mpatches.Patch(
            facecolor="0.75",
            alpha=0.35,
            label="SEAS5 95% ensemble members spread"
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="None",
            markerfacecolor="white",
            markersize=6,
            label="ERA5 outside SEAS5 95% ensemble member spread"
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=3.0,
        handletextpad=0.8,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.945,
        bottom=0.105,
        wspace=0.12,
        hspace=0.18
    )

    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_FIG}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("Fig.S2 daily evolution by two periods")
    print("ERA5 vs SEAS5 95% spread across 25 member composites")
    print("=" * 80)

    if OUT_NPZ.exists():
        era5, seas5, era5_counts, seas5_counts = load_cache(OUT_NPZ)

    else:
        era5_ssw = read_era5_ssw()
        seas5_ssw = read_seas5_ssw()

        era5_clim, _, _ = build_smoothed_baseline(load_era5_one_year, "ERA5")
        seas5_clim, _, _ = build_smoothed_baseline(load_seas5_one_year, "SEAS5")

        era5_trend = build_trend_params(load_era5_one_year, era5_clim, "ERA5")
        seas5_trend = build_trend_params(load_seas5_one_year, seas5_clim, "SEAS5")

        era5, era5_counts = calc_era5_lag_series_by_period(
            era5_ssw,
            era5_clim,
            era5_trend
        )

        seas5, seas5_counts = calc_seas5_member_lag_series_by_period(
            seas5_ssw,
            seas5_clim,
            seas5_trend
        )

        save_cache(
            OUT_NPZ,
            era5,
            seas5,
            era5_counts,
            seas5_counts
        )

        del era5_ssw, seas5_ssw
        del era5_clim, seas5_clim
        del era5_trend, seas5_trend
        gc.collect()

    plot_fig(era5, seas5)

    del era5, seas5, era5_counts, seas5_counts
    gc.collect()

    print("\nAll done.")


if __name__ == "__main__":
    main()
