# -*- coding: utf-8 -*-
"""
Fig.S1 10-year RAW + DETRENDED:

SEAS5 mean post-SSW T2m anomaly
+ hatching where ERA5 composite is outside SEAS5 95% member-event spread.

Layout:
    5 rows × 2 columns
    Left:  Raw
    Right: Detrended

Window:
    day +15 to +59

Baseline:
    1981–2010

SEAS5 spread:
    percentile across all independent (member, event) samples,
    NOT across 25 member-mean composites.
"""

import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


plt.rcParams["font.family"] = "Arial"
mpl.rcParams["hatch.linewidth"] = 0.5
mpl.rcParams["hatch.color"] = "black"


# ================================================================
# PATHS
# ================================================================
SEAS5_SSW_CSV_PATH = Path(
    r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv"
)

ERA5_SSW_CSV_PATH = Path(
    r"F:\data\paper_SSW_impacts_under_global_warming\figure\ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv"
)

ERA5_T2M_FILE = Path(
    r"F:\data\ERA5_data\ERA5_t2m_daily_1940_2024_no229.nc"
)

SEAS5_T2M_DIR = Path(
    r"F:\data\IFS_t2m_daily"
)

OUTPUT_DIR = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ_RAW = OUTPUT_DIR / "SI3_ERA5_outside_SEAS5_95spread_10yr_RAW_member.npz"
OUT_NPZ_DET = OUTPUT_DIR / "SI3_ERA5_outside_SEAS5_95spread_10yr_DET_member.npz"
OUT_FIG = OUTPUT_DIR / "SI3_ERA5_outside_SEAS5_95spread_10yr_RAW_DET_member.pdf"


# ================================================================
# SETTINGS
# ================================================================
START_YEAR = 1981
END_YEAR = 2024

BASELINE_START = 1981
BASELINE_END = 2010

N_MEMBERS = 25
MONTHS_NDJFM = [11, 12, 1, 2, 3]

D_START = 15
D_END = 59

LAT_THRESHOLD = 20

T2M_VAR_CANDIDATES = ["t2m", "2m_temperature"]

DECADE_PERIODS = [
    (1981, 1990, "1981-1990"),
    (1991, 2000, "1991-2000"),
    (2001, 2010, "2001-2010"),
    (2011, 2020, "2011-2020"),
    (2021, 2024, "2021-2024"),
]

REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": 220, "lon_max": 300},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min":   0, "lon_max":  40},
    "Siberia":      {"lat_min": 45, "lat_max": 70, "lon_min":  60, "lon_max": 120},
}


# ================================================================
# BASIC HELPERS
# ================================================================
def get_var_name(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify t2m variable, data_vars={list(ds.data_vars)}")


def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c:
        raise ValueError(f"No lat dim, dims={da.dims}")
    if not lon_c:
        raise ValueError(f"No lon dim, dims={da.dims}")
    return lat_c[0], lon_c[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time", "forecast_period"]:
        if name in da.dims:
            return name
    raise ValueError(f"No time dim, dims={da.dims}")


def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def _convert_lon_360_to_180(lon):
    return lon - 360 if lon > 180 else lon


def get_period_label(year):
    for y0, y1, label in DECADE_PERIODS:
        if y0 <= year <= y1:
            return label
    return None


def _add_region_boxes(ax):
    for _, rb in REGION_BOXES.items():
        lon_min = _convert_lon_360_to_180(rb["lon_min"])
        lon_max = _convert_lon_360_to_180(rb["lon_max"])

        for xs, ys in [
            ([lon_min, lon_max], [rb["lat_min"], rb["lat_min"]]),
            ([lon_min, lon_max], [rb["lat_max"], rb["lat_max"]]),
            ([lon_min, lon_min], [rb["lat_min"], rb["lat_max"]]),
            ([lon_max, lon_max], [rb["lat_min"], rb["lat_max"]]),
        ]:
            ax.plot(
                xs, ys,
                color="black",
                lw=1.3,
                transform=ccrs.PlateCarree(),
                zorder=6
            )


# ================================================================
# FIND SEAS5 FILE
# ================================================================
def find_seas5_file(year):
    patterns = [
        f"*{year}*t2m*.nc",
        f"*{year}*2m_temperature*.nc",
        f"*t2m*{year}*.nc",
        f"*2m_temperature*{year}*.nc",
        f"*{year}*.nc",
    ]

    for pat in patterns:
        files = sorted(SEAS5_T2M_DIR.rglob(pat))
        if files:
            return files[0]

    raise FileNotFoundError(f"Cannot find SEAS5 T2m file for {year} under {SEAS5_T2M_DIR}")


# ================================================================
# LOAD ERA5 ONE WINTER
# ================================================================
def load_era5_one_winter(init_year):
    date_start = pd.Timestamp(f"{init_year}-11-01")
    date_end = pd.Timestamp(f"{init_year + 1}-05-31")

    ds = xr.open_dataset(ERA5_T2M_FILE)
    var = get_var_name(ds, T2M_VAR_CANDIDATES)
    da = ds[var]

    lat_name, lon_name = get_lat_lon_dim_names(da)
    time_name = get_time_dim_name(da)

    da = da.sel({time_name: slice(date_start, date_end)})

    if da.sizes.get(time_name, 0) == 0:
        ds.close()
        return None

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    lon0 = da[lon_name].values
    if np.nanmax(lon0) > 180:
        lon_new = np.where(lon0 > 180, lon0 - 360, lon0)
        sort_idx = np.argsort(lon_new)
        da = da.isel({lon_name: sort_idx})
        da = da.assign_coords({lon_name: lon_new[sort_idx]})

    da = da.transpose(time_name, lat_name, lon_name).load().astype(np.float32)

    if float(da.mean()) > 100:
        da = da - 273.15

    times = pd.to_datetime(da[time_name].values).normalize()
    monthday = np.array([month_day_key(t) for t in times])

    out = {
        "data": da.values[None, ...],
        "times": times,
        "monthday": monthday,
        "lat": da[lat_name].values.copy(),
        "lon": da[lon_name].values.copy(),
        "time_to_idx": {pd.Timestamp(t): i for i, t in enumerate(times)}
    }

    ds.close()
    del ds, da
    gc.collect()

    return out


# ================================================================
# LOAD SEAS5 ONE WINTER
# ================================================================
def load_seas5_one_winter(init_year):
    date_start = pd.Timestamp(f"{init_year}-11-01")
    date_end = pd.Timestamp(f"{init_year + 1}-05-31")

    f = find_seas5_file(init_year)
    print(f"  SEAS5 {init_year}: {f.name}")

    ds = xr.open_dataset(f)
    var = get_var_name(ds, T2M_VAR_CANDIDATES)
    da = ds[var]

    lat_name, lon_name = get_lat_lon_dim_names(da)
    time_name = get_time_dim_name(da)

    if "number" not in da.dims:
        raise ValueError(f"SEAS5 file has no 'number' dimension: dims={da.dims}")

    da = da.sel({time_name: slice(date_start, date_end)})

    if da.sizes.get(time_name, 0) == 0:
        ds.close()
        return None

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    lon0 = da[lon_name].values
    if np.nanmax(lon0) > 180:
        lon_new = np.where(lon0 > 180, lon0 - 360, lon0)
        sort_idx = np.argsort(lon_new)
        da = da.isel({lon_name: sort_idx})
        da = da.assign_coords({lon_name: lon_new[sort_idx]})

    da = da.isel(number=slice(0, N_MEMBERS))
    da = da.transpose("number", time_name, lat_name, lon_name).load().astype(np.float32)

    if float(da.mean()) > 100:
        da = da - 273.15

    number_vals = da["number"].values
    member_to_idx = {}

    for i, m in enumerate(number_vals):
        member_to_idx[int(m)] = i
        member_to_idx[i + 1] = i
        member_to_idx[i] = i

    times = pd.to_datetime(da[time_name].values).normalize()
    monthday = np.array([month_day_key(t) for t in times])

    out = {
        "data": da.values,
        "times": times,
        "monthday": monthday,
        "lat": da[lat_name].values.copy(),
        "lon": da[lon_name].values.copy(),
        "member_to_idx": member_to_idx,
        "time_to_idx": {pd.Timestamp(t): i for i, t in enumerate(times)}
    }

    ds.close()
    del ds, da
    gc.collect()

    return out


# ================================================================
# READ SSW EVENTS
# ================================================================
def read_era5_ssw():
    df = pd.read_csv(ERA5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["ssw_date"])
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)]
    df = df[df["ssw_date"].dt.month.isin(MONTHS_NDJFM)].copy()

    df["member"] = 0
    df = df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)

    print(f"ERA5 SSW events: {len(df)}")
    return df


def read_seas5_ssw():
    df = pd.read_csv(SEAS5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()

    df = df.dropna(subset=["ssw_date"])
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)]
    df = df[df["ssw_date"].dt.month.isin(MONTHS_NDJFM)].copy()

    if "member" not in df.columns:
        raise ValueError("SEAS5 SSW CSV must contain column: member")

    df["member"] = df["member"].astype(int)
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)

    print(f"SEAS5 SSW events: {len(df)}")
    return df


# ================================================================
# BASELINE CLIMATOLOGY: day-of-year ±5-day smoothing
# ================================================================
def build_smoothed_baseline(loader_func, label_name):
    print(f"\nBuilding {label_name} baseline climatology with ±5-day smoothing...")
    print(f"Baseline period: {BASELINE_START}-{BASELINE_END}")

    doy_to_fields = {}
    lat_vals = lon_vals = None

    for year in range(BASELINE_START, BASELINE_END + 1):
        r = loader_func(year)
        if r is None:
            continue

        if lat_vals is None:
            lat_vals = r["lat"]
            lon_vals = r["lon"]

        # ERA5: member dim = 1
        # SEAS5: member dim = 25
        ens_mean = r["data"].mean(axis=0).astype(np.float32)  # time, lat, lon

        doys = pd.to_datetime(r["times"]).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(int(doy), []).append(
                ens_mean[i].astype(np.float32)
            )

        del r, ens_mean, doys
        gc.collect()

    if lat_vals is None or lon_vals is None:
        raise RuntimeError(f"No data loaded for {label_name} baseline.")

    nlat, nlon = len(lat_vals), len(lon_vals)

    # 366 是为了兼容闰年 dayofyear
    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        stacked = np.stack(fields, axis=0)
        clim_raw[doy - 1] = np.nanmean(stacked, axis=0).astype(np.float32)
        del stacked

    del doy_to_fields
    gc.collect()

    print(f"  Applying ±5-day rolling smoothing for {label_name}...")

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
        clim_smooth[i] = np.nanmean(
            clim_pad[i:i + window],
            axis=0
        ).astype(np.float32)

    del clim_pad, clim_raw
    gc.collect()

    # 用非闰年 2001 映射回 month-day key
    # 因此不生成 02-29
    dates_ref = pd.date_range("2001-01-01", "2001-12-31")

    baseline_clim = {}

    for d in dates_ref:
        doy = d.dayofyear
        key = f"{d.month:02d}-{d.day:02d}"
        baseline_clim[key] = clim_smooth[doy - 1].astype(np.float32)

    del clim_smooth
    gc.collect()

    print(f"  Done: {label_name} baseline keys = {len(baseline_clim)}")
    return baseline_clim, lat_vals, lon_vals


def build_era5_baseline():
    return build_smoothed_baseline(
        loader_func=load_era5_one_winter,
        label_name="ERA5"
    )


def build_seas5_baseline():
    return build_smoothed_baseline(
        loader_func=load_seas5_one_winter,
        label_name="SEAS5"
    )

def make_anomaly(data, monthday, clim):
    T = data.shape[1]
    nlat = data.shape[2]
    nlon = data.shape[3]

    clim_arr = np.full((T, nlat, nlon), np.nan, dtype=np.float32)

    for i, key in enumerate(monthday):
        if key in clim:
            clim_arr[i] = clim[key]

    return data - clim_arr[None, ...]


# ================================================================
# DETREND SLOPES
# ================================================================
def build_era5_detrend_slopes(clim):
    print("\nBuilding ERA5 detrend slopes...")

    md = {}

    for year in range(START_YEAR, END_YEAR + 1):
        r = load_era5_one_winter(year)
        if r is None:
            continue

        anom = make_anomaly(r["data"], r["monthday"], clim)
        ens_mean = anom.mean(axis=0)

        for i, key in enumerate(r["monthday"]):
            md.setdefault(key, []).append((year, ens_mean[i].astype(np.float32)))

        del r, anom, ens_mean
        gc.collect()

    slopes = {}

    for key, entries in md.items():
        years = np.array([e[0] for e in entries], dtype=np.float64)
        yr_mean = years.mean()
        yr_c = years - yr_mean

        fields = np.stack([e[1] for e in entries], axis=0).astype(np.float64)

        denom = np.sum(yr_c ** 2)
        if denom == 0:
            a = np.nanmean(fields, axis=0)
            b = np.zeros_like(a)
        else:
            a = np.nanmean(fields, axis=0)
            b = np.nansum(yr_c[:, None, None] * fields, axis=0) / denom

        slopes[key] = {
            "a": a.astype(np.float32),
            "b": b.astype(np.float32),
            "yr_mean": np.float32(yr_mean),
        }

        del years, yr_c, fields, a, b
        gc.collect()

    del md
    gc.collect()

    print(f"ERA5 detrend keys: {len(slopes)}")
    return slopes


def build_seas5_detrend_slopes(clim):
    print("\nBuilding SEAS5 detrend slopes...")

    md = {}

    for year in range(START_YEAR, END_YEAR + 1):
        r = load_seas5_one_winter(year)
        if r is None:
            continue

        anom = make_anomaly(r["data"], r["monthday"], clim)
        ens_mean = anom.mean(axis=0)

        for i, key in enumerate(r["monthday"]):
            md.setdefault(key, []).append((year, ens_mean[i].astype(np.float32)))

        del r, anom, ens_mean
        gc.collect()

    slopes = {}

    for key, entries in md.items():
        years = np.array([e[0] for e in entries], dtype=np.float64)
        yr_mean = years.mean()
        yr_c = years - yr_mean

        fields = np.stack([e[1] for e in entries], axis=0).astype(np.float64)

        denom = np.sum(yr_c ** 2)
        if denom == 0:
            a = np.nanmean(fields, axis=0)
            b = np.zeros_like(a)
        else:
            a = np.nanmean(fields, axis=0)
            b = np.nansum(yr_c[:, None, None] * fields, axis=0) / denom

        slopes[key] = {
            "a": a.astype(np.float32),
            "b": b.astype(np.float32),
            "yr_mean": np.float32(yr_mean),
        }

        del years, yr_c, fields, a, b
        gc.collect()

    del md
    gc.collect()

    print(f"SEAS5 detrend keys: {len(slopes)}")
    return slopes


def apply_detrend_inplace(anom, monthday, year, slopes):
    for i, key in enumerate(monthday):
        if key not in slopes:
            continue

        a = slopes[key]["a"]
        b = slopes[key]["b"]
        yr_mean = slopes[key]["yr_mean"]

        trend_val = a + b * (year - yr_mean)
        anom[:, i] = anom[:, i] - trend_val[None, :, :]

    return anom


# ================================================================
# ERA5 COMPOSITES BY PERIOD
# ================================================================
def calc_era5_composites_by_period(ssw_df, clim, detrend=False, slopes=None):
    mode = "detrended" if detrend else "raw"
    print(f"\nCalculating ERA5 {mode} composites by 10-year period...")

    period_samples = {label: [] for _, _, label in DECADE_PERIODS}
    lat_vals = lon_vals = None

    for year in range(START_YEAR, END_YEAR + 1):
        r = load_era5_one_winter(year)
        if r is None:
            continue

        if lat_vals is None:
            lat_vals = r["lat"]
            lon_vals = r["lon"]

        anom = make_anomaly(r["data"], r["monthday"], clim)

        if detrend:
            if slopes is None:
                raise ValueError("slopes must be provided when detrend=True")
            anom = apply_detrend_inplace(anom, r["monthday"], year, slopes)

        label = get_period_label(year)
        if label is None:
            del r, anom
            gc.collect()
            continue

        dfy = ssw_df[ssw_df["init_year"] == year]

        for _, row in dfy.iterrows():
            center = pd.Timestamp(row["ssw_date"]).normalize()

            if center not in r["time_to_idx"]:
                continue

            c_idx = r["time_to_idx"][center]
            i0 = c_idx + D_START
            i1 = c_idx + D_END + 1

            if i0 < 0 or i1 > anom.shape[1]:
                continue

            sample = np.nanmean(anom[0, i0:i1], axis=0).astype(np.float32)
            period_samples[label].append(sample)

        del r, anom
        gc.collect()

    results = {}

    for _, _, label in DECADE_PERIODS:
        samples = period_samples[label]

        if len(samples) == 0:
            raise RuntimeError(f"No ERA5 samples for period {label}")

        arr = np.stack(samples, axis=0).astype(np.float32)

        results[label] = {
            "comp": np.nanmean(arr, axis=0).astype(np.float32),
            "n": arr.shape[0],
        }

        print(f"ERA5 {mode} {label}: n={arr.shape[0]}")

        del arr, samples
        gc.collect()

    return results, lat_vals, lon_vals


# ================================================================
# SEAS5 MEMBER-EVENT SAMPLES BY PERIOD
# ================================================================
def calc_seas5_samples_by_period(ssw_df, clim, detrend=False, slopes=None):
    mode = "detrended" if detrend else "raw"
    print(f"\nCalculating SEAS5 {mode} 25 member-mean composites by 10-year period...")

    # period -> member -> event samples
    period_member_samples = {
        label: {m: [] for m in range(N_MEMBERS)}
        for _, _, label in DECADE_PERIODS
    }
    
    # ===== 新增：记录每个period的实际事件总数 =====
    period_event_counts = {label: 0 for _, _, label in DECADE_PERIODS}

    lat_vals = lon_vals = None

    for year in range(START_YEAR, END_YEAR + 1):
        r = load_seas5_one_winter(year)
        if r is None:
            continue

        if lat_vals is None:
            lat_vals = r["lat"]
            lon_vals = r["lon"]

        anom = make_anomaly(r["data"], r["monthday"], clim)

        if detrend:
            if slopes is None:
                raise ValueError("slopes must be provided when detrend=True")
            anom = apply_detrend_inplace(anom, r["monthday"], year, slopes)

        label = get_period_label(year)
        if label is None:
            del r, anom
            gc.collect()
            continue

        dfy = ssw_df[ssw_df["init_year"] == year]

        for _, row in dfy.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()

            if member not in r["member_to_idx"]:
                continue
            if center not in r["time_to_idx"]:
                continue

            m_idx = r["member_to_idx"][member]
            c_idx = r["time_to_idx"][center]

            i0 = c_idx + D_START
            i1 = c_idx + D_END + 1

            if i0 < 0 or i1 > anom.shape[1]:
                continue

            # 每个 event 先算 day15-59 平均场
            sample = np.nanmean(anom[m_idx, i0:i1], axis=0).astype(np.float32)

            # 存到对应 period + member
            period_member_samples[label][m_idx].append(sample)
            
            # ===== 统计实际事件总数 =====
            period_event_counts[label] += 1

        del r, anom
        gc.collect()

    results = {}

    for _, _, label in DECADE_PERIODS:
        member_comps = []
        member_counts = []

        for m in range(N_MEMBERS):
            samples = period_member_samples[label][m]

            if len(samples) == 0:
                print(f"[WARN] {mode} {label}: member {m+1:02d} has no events, skip.")
                continue

            arr = np.stack(samples, axis=0).astype(np.float32)

            # 每个 member 内部先对所有 events 平均
            member_comp = np.nanmean(arr, axis=0).astype(np.float32)

            member_comps.append(member_comp)
            member_counts.append(arr.shape[0])

            del arr, samples, member_comp
            gc.collect()

        if len(member_comps) < 2:
            raise RuntimeError(
                f"Not enough SEAS5 member composites for {mode} {label}: "
                f"n_members={len(member_comps)}"
            )

        member_comps = np.stack(member_comps, axis=0).astype(np.float32)

        results[label] = {
            # SEAS5 mean = 25 个 member composite 的平均
            "mean": np.nanmean(member_comps, axis=0).astype(np.float32),

            # 95% spread = 25 个 member-mean composites 的范围
            "lower": np.nanpercentile(member_comps, 2.5, axis=0).astype(np.float32),
            "upper": np.nanpercentile(member_comps, 97.5, axis=0).astype(np.float32),

            # ===== 修改：n 是实际事件总数，不是member数 =====
            "n": period_event_counts[label],  # 改这里！！！

            # 额外保存每个 member 有几个 event，方便检查
            "event_counts_per_member": np.array(member_counts, dtype=np.int16),
        }

        print(
            f"SEAS5 {mode} {label}: "
            f"total events={period_event_counts[label]}, "
            f"member composites n={member_comps.shape[0]}, "
            f"events per member min={np.min(member_counts)}, "
            f"max={np.max(member_counts)}, "
            f"mean={np.mean(member_counts):.2f}"
        )

        del member_comps, member_counts
        gc.collect()

    del period_member_samples
    gc.collect()

    return results, lat_vals, lon_vals

# ================================================================
# INTERPOLATION
# ================================================================
def interp_era5_to_seas5_if_needed(era5_field, era5_lat, era5_lon, seas5_lat, seas5_lon):
    if era5_field.shape == (len(seas5_lat), len(seas5_lon)):
        return era5_field.astype(np.float32)

    era5_da = xr.DataArray(
        era5_field,
        coords={"lat": era5_lat, "lon": era5_lon},
        dims=("lat", "lon")
    )

    era5_interp = era5_da.interp(
        lat=seas5_lat,
        lon=seas5_lon
    ).values.astype(np.float32)

    return era5_interp


# ================================================================
# BUILD FINAL MODE RESULT
# ================================================================
def build_mode_result(mode_name, out_npz, era5_ssw, seas5_ssw,
                      era5_clim, seas5_clim,
                      era5_slopes=None, seas5_slopes=None):
    detrend = mode_name.upper() == "DETRENDED"

    if out_npz.exists():
        print(f"\nLoading cached {mode_name}: {out_npz}")
        data = np.load(out_npz)

        lat_vals = data["lat"]
        lon_vals = data["lon"]

        results = {}
        for _, _, label in DECADE_PERIODS:
            results[label] = {
                "seas5_mean": data[f"{label}__seas5_mean"],
                "seas5_lower": data[f"{label}__seas5_lower"],
                "seas5_upper": data[f"{label}__seas5_upper"],
                "era5_interp": data[f"{label}__era5_interp"],
                "outside_mask": data[f"{label}__outside_mask"],
                "era5_n": int(data[f"{label}__era5_n"]),
                "seas5_n": int(data[f"{label}__seas5_n"]),  # 现在存的是实际事件数
            }

        return results, lat_vals, lon_vals

    era5_results, era5_lat, era5_lon = calc_era5_composites_by_period(
        era5_ssw,
        era5_clim,
        detrend=detrend,
        slopes=era5_slopes
    )

    seas5_results, lat_vals, lon_vals = calc_seas5_samples_by_period(
        seas5_ssw,
        seas5_clim,
        detrend=detrend,
        slopes=seas5_slopes
    )

    results = {}

    for _, _, label in DECADE_PERIODS:
        era5_interp = interp_era5_to_seas5_if_needed(
            era5_results[label]["comp"],
            era5_lat,
            era5_lon,
            lat_vals,
            lon_vals
        )

        seas5_mean = seas5_results[label]["mean"]
        seas5_lower = seas5_results[label]["lower"]
        seas5_upper = seas5_results[label]["upper"]

        outside_mask = (
            (era5_interp < seas5_lower) |
            (era5_interp > seas5_upper)
        ).astype(np.int8)

        results[label] = {
            "seas5_mean": seas5_mean.astype(np.float32),
            "seas5_lower": seas5_lower.astype(np.float32),
            "seas5_upper": seas5_upper.astype(np.float32),
            "era5_interp": era5_interp.astype(np.float32),
            "outside_mask": outside_mask,
            "era5_n": era5_results[label]["n"],      # ERA5实际事件数
            "seas5_n": seas5_results[label]["n"],    # SEAS5实际事件总数
        }

        print(
            f"{mode_name} {label}: "
            f"ERA5 n={era5_results[label]['n']}, "
            f"SEAS5 n={seas5_results[label]['n']}, "
            f"outside fraction={np.nanmean(outside_mask.astype(float)) * 100:.2f}%"
        )

    save_dict = {
        "lat": lat_vals,
        "lon": lon_vals,
    }

    for _, _, label in DECADE_PERIODS:
        save_dict[f"{label}__seas5_mean"] = results[label]["seas5_mean"]
        save_dict[f"{label}__seas5_lower"] = results[label]["seas5_lower"]
        save_dict[f"{label}__seas5_upper"] = results[label]["seas5_upper"]
        save_dict[f"{label}__era5_interp"] = results[label]["era5_interp"]
        save_dict[f"{label}__outside_mask"] = results[label]["outside_mask"]
        save_dict[f"{label}__era5_n"] = np.array(results[label]["era5_n"])
        save_dict[f"{label}__seas5_n"] = np.array(results[label]["seas5_n"])

    np.savez(out_npz, **save_dict)
    print(f"Saved cache: {out_npz}")

    del era5_results, seas5_results
    gc.collect()

    return results, lat_vals, lon_vals

# ================================================================
# PLOT RAW + DETRENDED
# ================================================================
# ================================================================
# PLOT RAW + DETRENDED
# ================================================================
def plot_raw_det_maps(raw_results, det_results, lat_vals, lon_vals):
    print(f"\nPlotting: {OUT_FIG}")

    lat_full = lat_vals.copy()
    lon = lon_vals.copy()
    nlat = len(lat_full)

    lat_threshold_idx = np.argmax(lat_full < LAT_THRESHOLD)
    if lat_threshold_idx == 0 and lat_full[0] < LAT_THRESHOLD:
        lat_threshold_idx = nlat

    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    nrows = len(DECADE_PERIODS)
    ncols = 2

    fig = plt.figure(figsize=(13, 2.5 * nrows))
    cs_ref = None

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)',
                    '(f)', '(g)', '(h)', '(i)', '(j)']

    for r, (_, _, label) in enumerate(DECADE_PERIODS):
        for c, (mode_results, col_title) in enumerate(
            zip([raw_results, det_results], ["Raw T2m", "Detrended T2m"])):
            
            ax = fig.add_subplot(
                nrows, ncols, r * ncols + c + 1,
                projection=ccrs.PlateCarree()
            )

            seas5_mean = mode_results[label]["seas5_mean"].copy()
            outside_mask = mode_results[label]["outside_mask"].copy()

            seas5_mean[lat_threshold_idx:] = np.nan

            mask = outside_mask.astype(float)
            mask[lat_threshold_idx:] = np.nan
            mask[mask == 0] = np.nan

            z, x_ = add_cyclic_point(seas5_mean, coord=lon)
            mask_c, _ = add_cyclic_point(mask, coord=lon)
            lon_grid, lat_grid = np.meshgrid(x_, lat_full)

            cs = ax.pcolormesh(
                lon_grid,
                lat_grid,
                z,
                transform=ccrs.PlateCarree(),
                cmap=plt.cm.RdBu_r,
                norm=norm,
                shading="auto"
            )
            cs_ref = cs

            ax.contourf(
                lon_grid,
                lat_grid,
                mask_c,
                levels=[0.5, 1.5],
                hatches=["////"],
                colors="none",
                transform=ccrs.PlateCarree()
            )

            ax.coastlines(linewidth=0.8)
            ax.set_extent([-180, 180, LAT_THRESHOLD, 90], crs=ccrs.PlateCarree())

            gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4, linewidth=0.4)
            gl.xlocator = plt.MultipleLocator(60)
            gl.ylocator = ticker.FixedLocator([30, 50, 70])
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (c == 0)
            gl.bottom_labels = (r == nrows - 1)
            gl.xlabel_style = {"size": 10}
            gl.ylabel_style = {"size": 10}

            if r == 0:
                _add_region_boxes(ax)

            panel_idx = r * ncols + c
            ax.text(
                0.02, 0.98, panel_labels[panel_idx],
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5),
                zorder=10
            )

            if r == 0:
                ax.text(
                    0.5, 1.25, col_title,
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    fontsize=13, fontweight='bold'
                )

        
            era5_n = mode_results[label]['era5_n']
            seas5_n = mode_results[label]['seas5_n']
            ax.set_title(
                f"{label} (n={seas5_n})",
                fontsize=12, fontweight="bold", pad=4
            )

            del seas5_mean, outside_mask, mask, z, mask_c, lon_grid, lat_grid
            gc.collect()

    fig.subplots_adjust(right=0.88, hspace=-0.6, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.3, 0.015, 0.4])
    cbar = fig.colorbar(cs_ref, cax=cbar_ax)
    cbar.set_label("T2m anomaly (K)", fontsize=12, fontweight="bold")

    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_FIG}")

# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("Fig.S1 10-year RAW + DETRENDED")
    print("SEAS5 mean + ERA5 outside SEAS5 95% member spread")
    print("=" * 80)

    # 如果 RAW 和 DET npz 都已经存在，直接读缓存画图
    if OUT_NPZ_RAW.exists() and OUT_NPZ_DET.exists():
        print("\nBoth RAW and DETRENDED caches exist. Skip all calculations.")

        raw_results, lat_vals, lon_vals = build_mode_result(
            mode_name="RAW",
            out_npz=OUT_NPZ_RAW,
            era5_ssw=None,
            seas5_ssw=None,
            era5_clim=None,
            seas5_clim=None,
            era5_slopes=None,
            seas5_slopes=None
        )

        det_results, _, _ = build_mode_result(
            mode_name="DETRENDED",
            out_npz=OUT_NPZ_DET,
            era5_ssw=None,
            seas5_ssw=None,
            era5_clim=None,
            seas5_clim=None,
            era5_slopes=None,
            seas5_slopes=None
        )

        plot_raw_det_maps(raw_results, det_results, lat_vals, lon_vals)

        del raw_results, det_results
        gc.collect()

        print("\nAll done.")
        return

    # 只有缺少 npz 时，才读数据和计算
    era5_ssw = read_era5_ssw()
    seas5_ssw = read_seas5_ssw()

    era5_clim, _, _ = build_era5_baseline()
    seas5_clim, lat_vals, lon_vals = build_seas5_baseline()

    raw_results, lat_vals, lon_vals = build_mode_result(
        mode_name="RAW",
        out_npz=OUT_NPZ_RAW,
        era5_ssw=era5_ssw,
        seas5_ssw=seas5_ssw,
        era5_clim=era5_clim,
        seas5_clim=seas5_clim,
        era5_slopes=None,
        seas5_slopes=None
    )

    if OUT_NPZ_DET.exists():
        era5_slopes = None
        seas5_slopes = None
    else:
        era5_slopes = build_era5_detrend_slopes(era5_clim)
        seas5_slopes = build_seas5_detrend_slopes(seas5_clim)

    det_results, _, _ = build_mode_result(
        mode_name="DETRENDED",
        out_npz=OUT_NPZ_DET,
        era5_ssw=era5_ssw,
        seas5_ssw=seas5_ssw,
        era5_clim=era5_clim,
        seas5_clim=seas5_clim,
        era5_slopes=era5_slopes,
        seas5_slopes=seas5_slopes
    )

    plot_raw_det_maps(raw_results, det_results, lat_vals, lon_vals)

    del era5_ssw, seas5_ssw
    del era5_clim, seas5_clim
    del raw_results, det_results
    gc.collect()

    print("\nAll done.")


if __name__ == "__main__":
    main()