# -*- coding: utf-8 -*-
"""
SEAS5 vs ERA5 SSW U10 amplitude + frequency trend comparison

Amplitude method:
- SEAS5 input: yearly pre-processed daily files (already daily mean, full season merged)
- ERA5 and SEAS5 both use fixed 1981-1985 climatology as baseline
- Raw anomaly computed first; detrended anomaly computed second
- ERA5 detrending: linear trend removed per calendar day (month-day) across years
- SEAS5 detrending: ensemble-mean anomaly trend removed per lead day across years
- Event metric: cumulative negative U10 anomaly over day 0 to +5 after SSW onset
- Bootstrap: randomly select one ensemble member per year, fit linear trend, repeat 5000 times

Frequency method:
- ERA5 frequency: annual binary occurrence (0/1) -> rolling 10-year sum -> linear trend
- SEAS5 frequency: bootstrap synthetic annual 0/1 time series by randomly selecting
  one member per year -> rolling 10-year sum -> linear trend distribution
"""

import gc
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ================================================================
# USER SETTINGS
# ================================================================
SEAS5_SSW_CSV_PATH = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
ERA5_SSW_CSV_PATH  = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure\ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv")

SEAS5_U10_DIR = Path(r"F:\data\IFS_U10")
ERA5_U_FILE   = Path(r"F:\data\ERA5_data\ERA5_u_daily_1940_2025_10_no229.nc")

OUTPUT_DIR = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1981
END_YEAR   = 2024

N_MEMBERS      = 25
BASELINE_START = 1981
BASELINE_END   = 2010

ERA5BASELINE_START = 1981
ERA5BASELINE_END   = 2010

TARGET_LAT            = 60.0
INTENSITY_WINDOW_DAYS = 5
MONTHS_DJF            = [11,12, 1, 2,3]
N_BOOT                = 5000
RANDOM_SEED           = 42
ROLLING_WINDOW_YEARS  = 10

# NPZ cache file paths
NPZ_ERA5_RAW   = OUTPUT_DIR / f"SI2_cache_ERA5_anom_raw0_5_{BASELINE_END}.npz"
NPZ_ERA5_DET   = OUTPUT_DIR / f"SI2_cache_ERA5_anom_det0_5_{BASELINE_END}.npz"
NPZ_SEAS5_RAW  = OUTPUT_DIR / f"SI2_cache_SEAS5_anom_raw0_5_{BASELINE_END}.npz"
NPZ_SEAS5_DET  = OUTPUT_DIR / f"SI2_cache_SEAS5_anom_det0_5_{BASELINE_END}.npz"
NPZ_SEAS5_META = OUTPUT_DIR / f"SI2_cache_SEAS5_meta0_5_{BASELINE_END}.npz"

SUMMARY_CSV = OUTPUT_DIR / f"SI2_SSW_amplitude&frequency_ERA5_vs_SEAS5_0_5_rolling_{BASELINE_END}.csv"
FIG_OUT     = OUTPUT_DIR / f"SI2_SSW_amplitude&frequency_ERA5_vs_SEAS5_0_5_rollin_{BASELINE_END}.pdf"


# ================================================================
# HELPERS
# ================================================================
def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def read_ssw_events_seas5():
    df = pd.read_csv(SEAS5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    for c in ["init_year", "member", "ssw_date"]:
        if c not in df.columns:
            raise ValueError(f"SEAS5 SSW CSV missing column: {c}")
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["member"] = df["member"].astype(int)
    df["month"]  = df["ssw_date"].dt.month
    df = df[df["month"].isin(MONTHS_DJF)].copy()
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
    print(f"[SEAS5] Loaded {len(df)} SSW events from CSV.")
    return df


def read_ssw_events_era5():
    df = pd.read_csv(ERA5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    for c in ["init_year", "ssw_date"]:
        if c not in df.columns:
            raise ValueError(f"ERA5 SSW CSV missing column: {c}")
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["month"] = df["ssw_date"].dt.month
    df = df[df["month"].isin(MONTHS_DJF)].copy()
    df = df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)
    print(f"[ERA5] Loaded {len(df)} SSW events from CSV.")
    return df


def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c: raise ValueError(f"No lat dim, dims={da.dims}")
    if not lon_c: raise ValueError(f"No lon dim, dims={da.dims}")
    return lat_c[0], lon_c[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time", "forecast_period"]:   
        if name in da.dims:
            return name
    for c in da.coords:
        if "time" in c.lower() or "valid" in c.lower() or "period" in c.lower(): 
            return c
    raise ValueError(f"No time dim, dims={da.dims}, coords={list(da.coords)}")

def safe_member_label(number_coord, idx):
    try:    return int(number_coord[idx])
    except: return idx + 1


def ols_slope_ignore_nan(years, values):
    years  = np.asarray(years,  dtype=float)
    values = np.asarray(values, dtype=float)
    valid  = np.isfinite(years) & np.isfinite(values)
    x, y   = years[valid], values[valid]
    if len(x) < 2:
        return np.nan, np.nan
    slope, intercept, r, p, se = stats.linregress(x, y)
    return float(slope), float(p)


# ================================================================
# ERA5 CACHE: SAVE / LOAD
# ================================================================
def save_era5_anom(path, era5_anom_df):
    np.savez_compressed(
        path,
        time     = era5_anom_df["time"].astype(str).values,
        u10      = era5_anom_df["u10"].values,
        u10_anom = era5_anom_df["u10_anom"].values,
        year     = era5_anom_df["year"].values,
        monthday = era5_anom_df["monthday"].values.astype(str),
    )
    print(f"   Saved ERA5 cache: {path.name}")


def load_era5_anom(path):
    d  = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "time":     pd.to_datetime(d["time"]),
        "u10":      d["u10"].astype(np.float32),
        "u10_anom": d["u10_anom"].astype(np.float32),
        "year":     d["year"].astype(int),
        "monthday": d["monthday"].astype(str),
    })
    print(f"   Loaded ERA5 cache: {path.name}")
    return df


# ================================================================
# SEAS5 CACHE: SAVE / LOAD
# ================================================================
def save_seas5_anom(path, anom_by_year):
    save_dict = {"years": np.array(sorted(anom_by_year.keys()), dtype=int)}
    for y, arr in anom_by_year.items():
        save_dict[f"y{y}"] = arr.astype(np.float32)
    np.savez_compressed(path, **save_dict)
    print(f"   Saved SEAS5 anomaly cache: {path.name}")


def load_seas5_anom(path):
    d    = np.load(path, allow_pickle=True)
    years = d["years"].tolist()
    anom  = {y: d[f"y{y}"].astype(np.float32) for y in years}
    print(f"   Loaded SEAS5 anomaly cache: {path.name}")
    return anom


def save_seas5_meta(path, all_data):
    save_dict = {"years": np.array(sorted(all_data.keys()), dtype=int)}
    for y, info in all_data.items():
        times_bytes          = np.array([str(t)[:26] for t in info["times"]], dtype="S26")
        save_dict[f"times_{y}"]   = times_bytes
        save_dict[f"members_{y}"] = np.array(info["member_labels"], dtype=int)
    np.savez_compressed(path, **save_dict)
    print(f"   Saved SEAS5 meta cache: {path.name}")


def load_seas5_meta(path):
    d     = np.load(path, allow_pickle=True)
    years = d["years"].tolist()
    all_data = {}
    for y in years:
        times_str  = d[f"times_{y}"].astype(str)
        times      = pd.to_datetime(times_str)
        mem_labels = d[f"members_{y}"].astype(int).tolist()
        all_data[y] = {
            "times":         times,
            "member_labels": mem_labels,
            "member_to_idx": {m: i for i, m in enumerate(mem_labels)},
            "time_to_idx":   {pd.Timestamp(t): j for j, t in enumerate(times)},
        }
    print(f"   Loaded SEAS5 meta cache: {path.name}")
    return all_data


# ================================================================
# ERA5 DAILY U10 PROCESSING
# ================================================================
def load_era5_u10_daily():
    print("\nLoading ERA5 daily U10...")
    ds  = xr.open_dataset(ERA5_U_FILE)
    da  = ds["u"].sel(level=10.0)

    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))

    lat_vals = da["lat"].values
    lat_idx  = int(np.argmin(np.abs(lat_vals - TARGET_LAT)))
    lat_used = float(lat_vals[lat_idx])

    da    = da.isel(lat=lat_idx).mean(dim="lon").load().astype(np.float32)
    times = pd.to_datetime(da["time"].values).normalize()

    df = pd.DataFrame({
        "time":     times,
        "u10":      da.values.astype(np.float32),
        "year":     times.year,
        "monthday": [month_day_key(t) for t in times],
    })

    ds.close()
    del ds, da, times
    gc.collect()
    print(f"  ERA5 latitude used = {lat_used:.2f}N")
    return df


# def build_era5_fixed_baseline_climatology(era5_df):
#     print(f"Building ERA5 baseline climatology: {ERA5BASELINE_START}-{ERA5BASELINE_END}")
#     dfb  = era5_df[(era5_df["year"] >= ERA5BASELINE_START) & (era5_df["year"] <= ERA5BASELINE_END)].copy()
#     clim = dfb.groupby("monthday")["u10"].mean().to_dict()
#     del dfb
#     gc.collect()
#     print(f"  Done: {len(clim)} month-day keys")
#     return clim
def build_era5_fixed_baseline_climatology(era5_df):
    print(f"Building ERA5 rolling climatology ({BASELINE_START}-{BASELINE_END}, ±5 days)")

    dfb = era5_df[
        (era5_df["year"] >= BASELINE_START) &
        (era5_df["year"] <= BASELINE_END)
    ].copy()

    # --- Step 1: day-of-year
    dfb["doy"] = pd.to_datetime(dfb["time"]).dt.dayofyear

    # --- Step 2: raw climatology
    clim_raw = dfb.groupby("doy")["u10"].mean().values

    # 补到 366
    if len(clim_raw) < 366:
        clim_pad_full = np.full(366, np.nan)
        clim_pad_full[:len(clim_raw)] = clim_raw
        clim_raw = clim_pad_full

    # --- Step 3: rolling ±5天
    window = 11
    pad = window // 2

    clim_pad = np.concatenate([
        clim_raw[-pad:], clim_raw, clim_raw[:pad]
    ])

    clim_smooth = np.full(366, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)

    # --- Step 4: month-day mapping
    baseline_clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        baseline_clim[key] = clim_smooth[t.dayofyear - 1]

    print("  ERA5 rolling climatology done.")
    return baseline_clim


def build_era5_anomalies_raw(era5_df, baseline_clim):
    print("Building ERA5 raw anomalies...")
    out = era5_df.copy()
    out["u10_anom"] = np.nan
    for key, idx in out.groupby("monthday").groups.items():
        if key in baseline_clim:
            out.loc[idx, "u10_anom"] = out.loc[idx, "u10"].values - baseline_clim[key]
    return out


def build_era5_anomalies_detrended(era5_anom_df):
    print("Building ERA5 detrended anomalies...")
    out = era5_anom_df.copy()
    for key, idx in out.groupby("monthday").groups.items():
        if len(idx) < 2:
            continue
        yvals = out.loc[idx, "u10_anom"].values.astype(float)
        years = out.loc[idx, "year"].values.astype(float)
        valid = (
            np.isfinite(yvals)
            & (years >= START_YEAR)
            & (years <= END_YEAR)
        )
        if valid.sum() < 2:
            continue
        yr_mean   = years[valid].mean()
        yr_c      = years[valid] - yr_mean
        slope     = np.sum(yr_c * yvals[valid]) / np.sum(yr_c ** 2)
        intercept = np.mean(yvals[valid])
        trend     = intercept + slope * (years - yr_mean)
        out.loc[idx, "u10_anom"] = (yvals - trend).astype(np.float32)
    gc.collect()
    print("  ERA5 detrending done.")
    return out


def compute_era5_event_anomalies(era5_events, era5_anom_df):
    print("Computing ERA5 event anomalies...")
    times    = pd.to_datetime(era5_anom_df["time"].values)
    u10_anom = era5_anom_df["u10_anom"].values
    records  = []

    for _, row in era5_events.iterrows():
        center = pd.Timestamp(row["ssw_date"]).normalize()
        mask   = (times >= center) & (times <= center + pd.Timedelta(days=INTENSITY_WINDOW_DAYS, hours=23))
        idx    = np.where(mask)[0]
        if len(idx) == 0:
            continue
        window_vals = u10_anom[idx]
        if not np.isfinite(window_vals).any():
            continue
        metric = float(np.nansum(np.maximum(-window_vals, 0)))
        records.append({
            "init_year":      int(row["init_year"]),
            "ssw_date":       center,
            "u10_event_anom": metric,
        })

    del times, u10_anom
    gc.collect()
    return pd.DataFrame(records)


def aggregate_era5_annual(event_df):
    return event_df.groupby("init_year")[["u10_event_anom"]].mean().reset_index()


# ================================================================
# SEAS5 DATA LOADING — 逐年加载，用完即释放
# ================================================================
def _load_one_seas5_year(year):
    fp = SEAS5_U10_DIR / f"SEAS5_u10hPa_NH_{year}11_system51_m25.nc"
    if not fp.exists():
        print(f"  Missing file: {fp.name}")
        return None

    try:
        ds  = xr.open_dataset(fp)
        var = list(ds.data_vars)[0]
        da  = ds[var].load()
        ds.close()
        del ds

        # squeeze 掉所有 size==1 的非 number 维
        for dim in list(da.dims):
            if dim != "number" and da.sizes[dim] == 1:
                da = da.squeeze(dim, drop=True)

        if "number" not in da.dims:
            raise ValueError(f"Missing 'number' dim, dims={da.dims}")
        da = da.transpose("number", *[d for d in da.dims if d != "number"])

        if "pressure_level" in da.dims:
            pidx = int(np.argmin(np.abs(da["pressure_level"].values - 10)))
            da = da.isel(pressure_level=pidx)
        elif "level" in da.dims:
            pidx = int(np.argmin(np.abs(da["level"].values - 10)))
            da = da.isel(level=pidx)

        da = da.isel(number=slice(0, N_MEMBERS))

        lat_name, lon_name = get_lat_lon_dim_names(da)
        time_name          = get_time_dim_name(da)

        if da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})

        lon0 = da[lon_name].values
        if np.nanmax(lon0) > 180:
            lon_new  = np.where(lon0 > 180, lon0 - 360, lon0)
            sort_idx = np.argsort(lon_new)
            da       = da.isel({lon_name: sort_idx})
            da       = da.assign_coords({lon_name: lon_new[sort_idx]})
            del lon_new, sort_idx
        del lon0

        lat_vals = da[lat_name].values
        lat_idx  = int(np.argmin(np.abs(lat_vals - TARGET_LAT)))
        lat_used = float(lat_vals[lat_idx])

        da = da.isel({lat_name: lat_idx}).mean(dim=lon_name).squeeze(drop=True)

        other_dims = [d for d in da.dims if d not in ["number", time_name]]
        if other_dims:
            raise ValueError(f"Unexpected dims after squeeze: {da.dims}")

        da        = da.transpose("number", time_name).astype(np.float32)
        arr_daily = da.values.astype(np.float32)

        # ---- 时间处理：区分 timedelta64 和 datetime64 ----
        raw_time = da[time_name].values
        if np.issubdtype(raw_time.dtype, np.timedelta64):
            # forecast_period 是相对偏移，需加上初始化日期 {year}-11-01
            init_date = pd.Timestamp(f"{year}-11-01")
            times = pd.to_datetime(
                [init_date + pd.Timedelta(td) for td in raw_time]
            ).normalize()
        else:
            times = pd.to_datetime(raw_time).normalize()

        member_labels = [safe_member_label(da["number"].values, i)
                         for i in range(da.sizes["number"])]

        del da, raw_time
        gc.collect()

        return arr_daily, times, member_labels, lat_used

    except Exception as e:
        print(f"  Failed {year}: {e}")
        gc.collect()
        return None
# ================================================================
# 分步计算并保存 SEAS5 缓存（逐年，内存最小化）
# ================================================================
def compute_and_save_seas5_caches():
    """
    逐年加载 SEAS5 U10，一次性完成：
      1. 收集 baseline 统计 -> 建立 climatology
      2. 计算 raw anomaly   -> 保存 meta + raw npz
      3. 收集 ensemble-mean per lead-day per year -> 建立 detrended anomaly
      4. 保存 det npz
    整个过程中每年的大数组用完即释放。
    """
    years_list = list(range(START_YEAR, END_YEAR + 1))

    # # ---- Pass 1: 建 baseline climatology（只读 baseline 年份）----
    # print(f"\n[Pass 1] Building SEAS5 baseline climatology: {BASELINE_START}-{BASELINE_END}")
    # baseline_years = [y for y in range(BASELINE_START, BASELINE_END + 1)]
    # clim_acc   = {}   # lead_day -> list of values
    # clim_tlen  = None

    # for year in baseline_years:
    #     result = _load_one_seas5_year(year)
    #     if result is None:
    #         continue
    #     arr, times, member_labels, lat_used = result
    #     tlen = arr.shape[1]
    #     if clim_tlen is None:
    #         clim_tlen = tlen
    #     usable = min(tlen, clim_tlen)
    #     for ti in range(usable):
    #         clim_acc.setdefault(ti, []).extend(arr[:, ti].tolist())
    #     del arr, times, member_labels
    #     gc.collect()

    # if clim_tlen is None or not clim_acc:
    #     raise RuntimeError("No SEAS5 baseline data loaded.")

    # seas5_clim = np.array([np.nanmean(clim_acc[ti]) for ti in range(clim_tlen)],
    #                       dtype=np.float32)
    # del clim_acc
    # gc.collect()
    # print(f"  Climatology length = {clim_tlen}")
    # ---- Pass 1: 建 SEAS5 rolling climatology（calendar day）
    print(f"\n[Pass 1] Building SEAS5 rolling climatology: {BASELINE_START}-{BASELINE_END}")

    doy_to_fields = {}

    for year in range(BASELINE_START, BASELINE_END + 1):

        result = _load_one_seas5_year(year)
        if result is None:
            continue

        arr, times, member_labels, lat_used = result

        ens_mean = np.nanmean(arr, axis=0)  # (time)

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(ens_mean[i])

        del arr, times, member_labels
        gc.collect()

    # --- raw climatology
    clim_raw = np.full(366, np.nan)

    for doy, vals in doy_to_fields.items():
        clim_raw[doy - 1] = np.nanmean(vals)

    del doy_to_fields
    gc.collect()

    # --- rolling ±5 days
    window = 11
    pad = window // 2

    clim_pad = np.concatenate([
        clim_raw[-pad:], clim_raw, clim_raw[:pad]
    ])

    clim_smooth = np.full(366, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)

    # --- month-day mapping
    seas5_clim = {}

    for t in pd.date_range("2001-01-01", "2001-12-31"):
        seas5_clim[month_day_key(t)] = clim_smooth[t.dayofyear - 1]

    print("  SEAS5 climatology done.")

    # # ---- Pass 2: 逐年计算 raw anomaly，同时收集 ensemble-mean 用于去趋势 ----
    # print("\n[Pass 2] Computing SEAS5 raw anomalies (year by year)...")
    # anom_raw_by_year = {}      # year -> np.float32 (n_mem, tlen)  — 暂存，用于 pass3
    # ens_mean_by_year = {}      # year -> np.float32 (common_tlen,)  — 供去趋势用
    # meta_by_year     = {}      # year -> {times, member_labels}      — 供 save_seas5_meta 用
    # common_tlen      = None

    # for year in years_list:
    #     result = _load_one_seas5_year(year)
    #     if result is None:
    #         continue
    #     arr, times, member_labels, lat_used = result
    #     nmem, tlen = arr.shape

    #     usable = min(tlen, clim_tlen)
    #     anom = np.full((nmem, tlen), np.nan, dtype=np.float32)
    #     anom[:, :usable] = arr[:, :usable] - seas5_clim[:usable][None, :]

    #     # 收集 ensemble-mean（取 usable 部分）
    #     em = np.nanmean(anom[:, :usable], axis=0).astype(np.float32)

    #     if common_tlen is None:
    #         common_tlen = usable
    #     else:
    #         common_tlen = min(common_tlen, usable)

    #     anom_raw_by_year[year] = anom
    #     ens_mean_by_year[year] = em
    #     meta_by_year[year]     = {"times": times, "member_labels": member_labels}

    #     print(f"  {year}: tlen={tlen}, usable={usable}, lat={lat_used:.2f}N")
    #     del arr, times, member_labels, anom, em
    #     gc.collect()

    # del seas5_clim
    # gc.collect()

    # # 保存 meta
    # all_data_meta = {}
    # for y, info in meta_by_year.items():
    #     ml = info["member_labels"]
    #     all_data_meta[y] = {
    #         "times":         info["times"],
    #         "member_labels": ml,
    #     }
    # save_seas5_meta(NPZ_SEAS5_META, all_data_meta)
    # del meta_by_year, all_data_meta
    # gc.collect()

    # # 保存 raw anomaly
    # save_seas5_anom(NPZ_SEAS5_RAW, anom_raw_by_year)
    # ---- Pass 2: 逐年计算 raw anomaly（calendar-day）----
    print("\n[Pass 2] Computing SEAS5 raw anomalies (calendar-day)...")

    anom_raw_by_year = {}
    ens_mean_by_year = {}
    meta_by_year     = {}

    for year in years_list:

        result = _load_one_seas5_year(year)
        if result is None:
            continue

        arr, times, member_labels, lat_used = result

        nmem, tlen = arr.shape

        anom = np.full((nmem, tlen), np.nan, dtype=np.float32)

        # ✅ 用 calendar-day climatology
        for ti in range(tlen):
            key = month_day_key(times[ti])
            cf  = seas5_clim.get(key)

            if cf is None:
                continue

            anom[:, ti] = arr[:, ti] - cf

        # ✅ ensemble mean（整个序列）
        em = np.nanmean(anom, axis=0).astype(np.float32)

        anom_raw_by_year[year] = anom
        ens_mean_by_year[year] = em
        meta_by_year[year]     = {
            "times": times,
            "member_labels": member_labels
        }

        print(f"  {year}: tlen={tlen}, lat={lat_used:.2f}N")

        del arr, times, member_labels, anom, em
        gc.collect()

    # # ---- Pass 3: 去趋势（基于 ensemble-mean 逐 lead-day 线性去趋势）----
    # print(f"\n[Pass 3] Detrending SEAS5 anomalies (common_tlen={common_tlen})...")
    # years_sorted = sorted(anom_raw_by_year.keys())
    # yr_vals  = np.array(years_sorted, dtype=np.float64)
    # yr_mean  = yr_vals.mean()
    # yr_c     = yr_vals - yr_mean

    # # 预先计算每个 lead-day 的去趋势偏移量（标量数组，极小内存）
    # trend_offset = {}   # year -> np.float32 (common_tlen,)
    # for ti in range(common_tlen):
    #     ens_mean_ti = np.array(
    #         [ens_mean_by_year[y][ti] if ti < len(ens_mean_by_year[y]) else np.nan
    #          for y in years_sorted],
    #         dtype=np.float64
    #     )
    #     valid = np.isfinite(ens_mean_ti)
    #     if valid.sum() < 2:
    #         for y in years_sorted:
    #             trend_offset.setdefault(y, np.zeros(common_tlen, dtype=np.float32))
    #         continue
    #     b = np.sum(yr_c[valid] * ens_mean_ti[valid]) / np.sum(yr_c[valid] ** 2)
    #     a = np.nanmean(ens_mean_ti[valid])
    #     for yi, y in enumerate(years_sorted):
    #         if y not in trend_offset:
    #             trend_offset[y] = np.zeros(common_tlen, dtype=np.float32)
    #         trend_offset[y][ti] = np.float32(a + b * (y - yr_mean))

    #     if (ti + 1) % 20 == 0 or ti == 0:
    #         print(f"  lead day {ti+1}/{common_tlen}")

    # del ens_mean_by_year, yr_vals, yr_mean, yr_c
    # gc.collect()

    # # 逐年施加去趋势偏移，生成 det anomaly，用完 raw 即释放
    # anom_det_by_year = {}
    # for y in years_sorted:
    #     raw  = anom_raw_by_year[y]
    #     offs = trend_offset.get(y, np.zeros(common_tlen, dtype=np.float32))
    #     det  = raw.copy()
    #     det[:, :common_tlen] -= offs[None, :]
    #     anom_det_by_year[y] = det.astype(np.float32)
    #     del raw, offs, det
    #     gc.collect()

    # del anom_raw_by_year, trend_offset
    # gc.collect()

    # save_seas5_anom(NPZ_SEAS5_DET, anom_det_by_year)
    # del anom_det_by_year
    # gc.collect()
    # ---- Pass 3: calendar-day detrending ----
    print("\n[Pass 3] Detrending SEAS5 anomalies (calendar-day)...")

    # ✅ Step 1: 收集 calendar-day 的 anomaly（ensemble mean）
    md_series = {}

    for year, arr in anom_raw_by_year.items():

        info = meta_by_year[year]
        times = info["times"]

        for ti, t in enumerate(times):

            key = month_day_key(t)

            vals = arr[:, ti]

            if np.isfinite(vals).any():
                md_series.setdefault(key, []).append(
                    (year, np.nanmean(vals))
                )

    # ✅ Step 2: 算趋势
    md_coeffs = {}

    for key, entries in md_series.items():

        if len(entries) < 2:
            continue

        years = np.array([e[0] for e in entries], dtype=float)
        vals  = np.array([e[1] for e in entries], dtype=float)

        valid = np.isfinite(vals)
        if valid.sum() < 2:
            continue

        years = years[valid]
        vals  = vals[valid]

        yr_mean = years.mean()
        yr_c    = years - yr_mean

        slope = np.sum(yr_c * vals) / np.sum(yr_c**2)
        intercept = np.mean(vals)

        md_coeffs[key] = (intercept, slope, yr_mean)

    # ✅ Step 3: 应用 detrend
    anom_det_by_year = {}

    for year, arr in anom_raw_by_year.items():

        info = meta_by_year[year]
        times = info["times"]

        det = arr.copy()

        for ti, t in enumerate(times):

            key = month_day_key(t)

            if key not in md_coeffs:
                continue

            a, b, yr_mean = md_coeffs[key]

            # # 只去 slope（你的设定正确）
            # det[:, ti] -= b * (year - yr_mean)
            # 去完整线性趋势（截距 + 斜率）
            det[:, ti] -= (a + b * (year - yr_mean))

        anom_det_by_year[year] = det.astype(np.float32)

        print(f"  detrended {year}")
        # ✅ 保存 raw
        save_seas5_anom(NPZ_SEAS5_RAW, anom_raw_by_year)

        # ✅ 保存 det
        save_seas5_anom(NPZ_SEAS5_DET, anom_det_by_year)

        # ✅ 保存 meta
        all_data_meta = {}
        for y, info in meta_by_year.items():
            all_data_meta[y] = {
                "times": info["times"],
                "member_labels": info["member_labels"]
            }

        save_seas5_meta(NPZ_SEAS5_META, all_data_meta)

        print("  All SEAS5 caches saved.")

        del arr, det
        gc.collect()



# ================================================================
# SEAS5 EVENT ANOMALIES
# ================================================================
def compute_seas5_event_anomalies(ssw_df, all_data_meta, anom_by_year):
    print("Computing SEAS5 event anomalies...")
    records = []

    for year in sorted(anom_by_year.keys()):
        dfy = ssw_df[ssw_df["init_year"] == year]
        if dfy.empty:
            continue

        info          = all_data_meta[year]
        arr           = anom_by_year[year]
        member_to_idx = info["member_to_idx"]
        time_to_idx   = info["time_to_idx"]
        tlen          = arr.shape[1]

        for _, row in dfy.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()
            if member not in member_to_idx or center not in time_to_idx:
                continue

            mi = member_to_idx[member]
            ti = time_to_idx[center]
            i0 = ti
            i1 = min(ti + INTENSITY_WINDOW_DAYS + 1, tlen)
            if i1 <= i0:
                continue

            window_vals = arr[mi, i0:i1]
            if not np.isfinite(window_vals).any():
                continue

            metric = float(np.nansum(np.maximum(-window_vals, 0)))
            records.append({
                "init_year":      year,
                "member":         member,
                "ssw_date":       center,
                "u10_event_anom": metric,
            })

        gc.collect()

    print(f"  Total SEAS5 event records: {len(records)}")
    return pd.DataFrame(records)


def aggregate_seas5_annual_member(event_df):
    return (
        event_df.groupby(["init_year", "member"])[["u10_event_anom"]]
        .mean().reset_index()
    )


# ================================================================
# FREQUENCY AND OCCURRENCE
# ================================================================
def build_era5_occurrence_annual(ssw_df_era5):
    years     = np.arange(START_YEAR, END_YEAR + 1)
    has_event = ssw_df_era5.groupby("init_year").size().to_dict()
    return pd.DataFrame({
        "init_year":      years,
        "ssw_occurrence": [1.0 if y in has_event else 0.0 for y in years],
    })


def build_seas5_occurrence_annual_member(ssw_df_seas5):
    years     = np.arange(START_YEAR, END_YEAR + 1)
    members   = np.arange(1, N_MEMBERS + 1)
    has_event = set(zip(ssw_df_seas5["init_year"].astype(int),
                        ssw_df_seas5["member"].astype(int)))
    records = [
        {"init_year": y, "member": m,
         "ssw_occurrence": 1.0 if (y, m) in has_event else 0.0}
        for y in years for m in members
    ]
    return pd.DataFrame(records)


def build_seas5_occurrence_fraction_annual(seas5_occ_member_df):
    return (
        seas5_occ_member_df.groupby("init_year")["ssw_occurrence"]
        .mean().reset_index(name="ssw_occ_fraction")
    )


def build_rolling_frequency_from_binary_series(years, binary_values, window_years=10):
    df = pd.DataFrame({
        "init_year":  np.asarray(years, dtype=int),
        "occurrence": np.asarray(binary_values, dtype=float),
    }).sort_values("init_year").reset_index(drop=True)
    df["rolling_freq"] = (
        df["occurrence"]
        .rolling(window=window_years, min_periods=window_years)
        .sum()
    )
    return df.dropna(subset=["rolling_freq"]).reset_index(drop=True)


def bootstrap_memberwise_trends(seas5_annual_member_df, value_col, n_boot=5000, seed=42):
    rng         = np.random.default_rng(seed)
    years_all   = np.arange(START_YEAR, END_YEAR + 1)
    members_all = np.arange(1, N_MEMBERS + 1)

    lookup = {}
    for _, row in seas5_annual_member_df.iterrows():
        lookup[(int(row["init_year"]), int(row["member"]))] = row[value_col]

    trend_slopes = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        yrs, vals = [], []
        for y in years_all:
            mem = int(rng.choice(members_all))
            val = lookup.get((y, mem), np.nan)
            if np.isfinite(val):
                yrs.append(y); vals.append(val)
        if len(yrs) >= 2:
            trend_slopes[b], _ = ols_slope_ignore_nan(yrs, vals)
    return trend_slopes


def bootstrap_seas5_occurrence_fraction_trends(seas5_occ_member_df, n_boot=5000, seed=42):
    rng         = np.random.default_rng(seed)
    years_all   = np.arange(START_YEAR, END_YEAR + 1)
    members_all = np.arange(1, N_MEMBERS + 1)

    lookup = {}
    for _, row in seas5_occ_member_df.iterrows():
        lookup[(int(row["init_year"]), int(row["member"]))] = row["ssw_occurrence"]

    trend_slopes = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        yrs, vals = [], []
        for y in years_all:
            sampled  = rng.choice(members_all, size=N_MEMBERS, replace=True)
            mem_vals = np.array([lookup.get((y, int(m)), np.nan) for m in sampled], dtype=float)
            if np.isfinite(mem_vals).any():
                yrs.append(y); vals.append(float(np.nanmean(mem_vals)))
        if len(yrs) >= 2:
            trend_slopes[b], _ = ols_slope_ignore_nan(yrs, vals)
    return trend_slopes


def bootstrap_seas5_rolling_frequency_trends(seas5_occ_member_df, window_years=10,
                                              n_boot=5000, seed=42):
    rng         = np.random.default_rng(seed)
    years_all   = np.arange(START_YEAR, END_YEAR + 1)
    members_all = np.arange(1, N_MEMBERS + 1)

    lookup = {}
    for _, row in seas5_occ_member_df.iterrows():
        lookup[(int(row["init_year"]), int(row["member"]))] = row["ssw_occurrence"]

    trend_slopes = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        occ_series = np.array(
            [lookup.get((y, int(rng.choice(members_all))), np.nan) for y in years_all],
            dtype=float
        )
        if np.isfinite(occ_series).sum() < window_years:
            continue
        roll_df = build_rolling_frequency_from_binary_series(
            years_all, occ_series, window_years=window_years
        )
        if len(roll_df) < 2:
            continue
        trend_slopes[b], _ = ols_slope_ignore_nan(roll_df["init_year"], roll_df["rolling_freq"])
    return trend_slopes


# ================================================================
# PLOT SETTINGS
# ================================================================
plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

C_HIST = "#1C6AB1"
C_ERA5 = "#ED4043"
C_CI90 = "#9EC3E6"
C_CI95 = "#000000"


def plot_trend_distribution(ax, model_slopes, era5_slope, era5_p, title, xlabel, trend_unit,
                             xlim=None, show_ylabel=True):
    vals = model_slopes[np.isfinite(model_slopes)]

    p025 = np.nanpercentile(vals, 2.5)
    p05  = np.nanpercentile(vals, 5.0)
    p95  = np.nanpercentile(vals, 95.0)
    p975 = np.nanpercentile(vals, 97.5)

    n, bins, patches = ax.hist(
        vals, bins=45, color=C_HIST, alpha=0.82,
        edgecolor="white", linewidth=0.35, zorder=2,
    )
    ymax = max(n.max(), 1)

    ax.axvspan(p025, p975, alpha=0.10, color=C_CI95, zorder=1)
    ax.axvspan(p05,  p95,  alpha=0.16, color=C_CI90, zorder=1)

    ax.axvline(0,          color="black", lw=0.9, ls="-",  zorder=3)
    ax.axvline(p05,        color=C_CI90,  lw=1.2, ls="--", zorder=4)
    ax.axvline(p95,        color=C_CI90,  lw=1.2, ls="--", zorder=4)
    ax.axvline(p025,       color=C_CI95,  lw=1.0, ls=":",  zorder=4)
    ax.axvline(p975,       color=C_CI95,  lw=1.0, ls=":",  zorder=4)
    ax.axvline(era5_slope, color=C_ERA5,  lw=2.0, ls="--", zorder=5)
    seas5_mean = float(np.nanmean(vals))

    ax.axvline(
        seas5_mean,
        color="#2CA02C",
        lw=1.8,
        ls="-.",
        zorder=5
    )

    is_right = era5_slope >= np.nanmedian(vals)
    x_span   = (np.nanmax(vals) - np.nanmin(vals)) if len(vals) > 1 else 1.0
    x_offset = 0.15 * x_span if x_span > 0 else 0.2
    x_text   = era5_slope + (x_offset if is_right else -x_offset)
    arrow_y  = ymax * 0.68
    ha_text  = "left" if is_right else "right"

    ax.annotate(
        f"ERA5\n{era5_slope:+.2f} {trend_unit}\np = {era5_p:.2f}",
        xy=(era5_slope, arrow_y),
        xytext=(x_text, arrow_y * 1.06),
        fontsize=6.5, color=C_ERA5, ha=ha_text, va="center",fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_ERA5, lw=0.9, mutation_scale=6),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_ERA5, lw=0.6, alpha=0.92),
        zorder=6,
    )
    # ======================================================
    # SEAS5 mean annotation
    # ======================================================
    C_SEAS5 = "#2CA02C"

    is_right_s5 = seas5_mean >= np.nanmedian(vals)

    if is_right_s5 == is_right:
        s5_offset = x_offset * 2.2
    else:
        s5_offset = x_offset

    x_text_s5 = seas5_mean + (
        s5_offset if is_right_s5 else -s5_offset
    )

    ha_s5 = "left" if is_right_s5 else "right"

    ax.annotate(
        f"SEAS5 mean\n{seas5_mean:+.2f} {trend_unit}",
        xy=(seas5_mean, ymax * 0.48),
        xytext=(x_text_s5, ymax * 0.54),

        fontsize=6.5,
        color=C_SEAS5,
        ha=ha_s5,
        va="center",fontweight="bold",

        arrowprops=dict(
            arrowstyle="-|>",
            color=C_SEAS5,
            lw=0.9,
            mutation_scale=6,
        ),

        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec=C_SEAS5,
            lw=0.6,
            alpha=0.92,
        ),

        zorder=6,
    )

    if xlim is None:
        xmin = min(np.nanmin(vals), era5_slope) - 0.2 * x_span
        xmax = max(np.nanmax(vals), era5_slope) + 0.2 * x_span
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            xmin, xmax = -1, 1
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(*xlim)

    ax.set_ylim(0, ymax * 1.32)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax.set_xlabel(xlabel, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("Bootstrap count", labelpad=4)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9, pad=5)

    legend_handles = [
        mpatches.Patch(color=C_HIST, alpha=0.82, label="SEAS5 bootstrap"),
        mpatches.Patch(color=C_CI90, alpha=0.32, label="5-95 % range"),
        mpatches.Patch(color=C_CI95, alpha=0.20, label="2.5-97.5 % range"),
        Line2D([0], [0], color=C_ERA5,  lw=1.8, ls="--", label="ERA5 trend"),
        Line2D([0], [0], color="#2CA02C",lw=1.8,ls="-.",label="SEAS5 mean"),
        Line2D([0], [0], color="black", lw=0.9, ls="-",  label="Zero line"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        frameon=True, framealpha=0.92, edgecolor="0.68",
        handlelength=1.6, handletextpad=0.45,
        borderpad=0.45, labelspacing=0.32,
    )
    ax.yaxis.grid(True, ls="--", lw=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


# ================================================================
# SUMMARY
# ================================================================
def summarize_bootstrap_case(vals, slope, pval, label):
    vals = vals[np.isfinite(vals)]
    p05  = np.nanpercentile(vals, 5)
    p95  = np.nanpercentile(vals, 95)
    p025 = np.nanpercentile(vals, 2.5)
    p975 = np.nanpercentile(vals, 97.5)
    return {
        "case":             label,
        "era5_slope":       slope,
        "era5_pvalue":      pval,
        "seas5_boot_n":     len(vals),
        "seas5_p05":        p05,
        "seas5_p95":        p95,
        "seas5_p025":       p025,
        "seas5_p975":       p975,
        "era5_in_5_95":     (slope >= p05)  and (slope <= p95),
        "era5_in_2p5_97p5": (slope >= p025) and (slope <= p975),
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 84)
    print("SEAS5 vs ERA5: SSW U10 amplitude + frequency trend distribution (10-yr rolling)")
    print(f"Fixed baseline climatology = {BASELINE_START}-{BASELINE_END}")
    print("=" * 84)

    ssw_df_seas5 = read_ssw_events_seas5()
    ssw_df_era5  = read_ssw_events_era5()

    # --- SSW frequency / occurrence series ---
    print("\nComputing SSW occurrence / frequency series...")
    era5_occ_annual   = build_era5_occurrence_annual(ssw_df_era5)
    era5_roll_freq    = build_rolling_frequency_from_binary_series(
        era5_occ_annual["init_year"].values,
        era5_occ_annual["ssw_occurrence"].values,
        window_years=ROLLING_WINDOW_YEARS,
    )
    seas5_occ_member          = build_seas5_occurrence_annual_member(ssw_df_seas5)
    seas5_occ_fraction_annual = build_seas5_occurrence_fraction_annual(seas5_occ_member)

    era5_freq_slope,  era5_freq_p  = ols_slope_ignore_nan(
        era5_roll_freq["init_year"], era5_roll_freq["rolling_freq"])
    seas5_frac_slope, seas5_frac_p = ols_slope_ignore_nan(
        seas5_occ_fraction_annual["init_year"], seas5_occ_fraction_annual["ssw_occ_fraction"])

    print("\nFrequency trends:")
    print(f"  ERA5 rolling-{ROLLING_WINDOW_YEARS}yr: slope = {era5_freq_slope:+.5f}, p = {era5_freq_p:.3f}")
    print(f"  SEAS5 fraction (check):   slope = {seas5_frac_slope:+.5f}, p = {seas5_frac_p:.3f}")

    # --- ERA5 raw anomaly ---
    if NPZ_ERA5_RAW.exists():
        print("\n[Cache] Loading ERA5 raw anomaly...")
        era5_anom_raw = load_era5_anom(NPZ_ERA5_RAW)
    else:
        print("\n[Compute] ERA5 raw anomaly...")
        era5_daily    = load_era5_u10_daily()
        era5_clim     = build_era5_fixed_baseline_climatology(era5_daily)
        era5_anom_raw = build_era5_anomalies_raw(era5_daily, era5_clim)
        save_era5_anom(NPZ_ERA5_RAW, era5_anom_raw)
        del era5_daily, era5_clim
        gc.collect()

    # --- ERA5 detrended anomaly ---
    if NPZ_ERA5_DET.exists():
        print("\n[Cache] Loading ERA5 detrended anomaly...")
        era5_anom_det = load_era5_anom(NPZ_ERA5_DET)
    else:
        print("\n[Compute] ERA5 detrended anomaly...")
        era5_anom_det = build_era5_anomalies_detrended(era5_anom_raw)
        save_era5_anom(NPZ_ERA5_DET, era5_anom_det)

    era5_events_raw = compute_era5_event_anomalies(ssw_df_era5, era5_anom_raw)
    era5_events_det = compute_era5_event_anomalies(ssw_df_era5, era5_anom_det)
    era5_annual_raw = aggregate_era5_annual(era5_events_raw)
    era5_annual_det = aggregate_era5_annual(era5_events_det)

    del era5_anom_raw, era5_anom_det
    gc.collect()

    # --- SEAS5 所有缓存：若任一缺失则重新计算（逐年，内存最小）---
    seas5_caches_exist = (NPZ_SEAS5_RAW.exists() and
                          NPZ_SEAS5_DET.exists() and
                          NPZ_SEAS5_META.exists())
    if not seas5_caches_exist:
        print("\n[Compute] SEAS5 caches missing. Running year-by-year computation...")
        compute_and_save_seas5_caches()
    else:
        print("\n[Cache] All SEAS5 caches found.")

    # --- 读取 SEAS5 缓存（raw 用完后释放，再读 det）---
    print("\n[Cache] Loading SEAS5 raw anomaly...")
    seas5_anom_raw = load_seas5_anom(NPZ_SEAS5_RAW)
    print("\n[Cache] Loading SEAS5 meta...")
    seas5_meta = load_seas5_meta(NPZ_SEAS5_META)

    seas5_events_raw = compute_seas5_event_anomalies(ssw_df_seas5, seas5_meta, seas5_anom_raw)
    del seas5_anom_raw
    gc.collect()

    print("\n[Cache] Loading SEAS5 detrended anomaly...")
    seas5_anom_det = load_seas5_anom(NPZ_SEAS5_DET)

    seas5_events_det = compute_seas5_event_anomalies(ssw_df_seas5, seas5_meta, seas5_anom_det)
    del seas5_anom_det, seas5_meta
    gc.collect()

    seas5_annual_member_raw = aggregate_seas5_annual_member(seas5_events_raw)
    seas5_annual_member_det = aggregate_seas5_annual_member(seas5_events_det)

    # --- ERA5 amplitude trend slopes ---
    era5_raw_slope, era5_raw_p = ols_slope_ignore_nan(
        era5_annual_raw["init_year"], era5_annual_raw["u10_event_anom"])
    era5_det_slope, era5_det_p = ols_slope_ignore_nan(
        era5_annual_det["init_year"], era5_annual_det["u10_event_anom"])

    print("\nERA5 amplitude trends:")
    print(f"  Raw:       slope = {era5_raw_slope:+.5f} m/s/yr, p = {era5_raw_p:.3f}")
    print(f"  Detrended: slope = {era5_det_slope:+.5f} m/s/yr, p = {era5_det_p:.3f}")

    # --- Bootstrap SEAS5 trend distributions ---
    print("\nBootstrapping SEAS5 trend distributions...")
    seas5_raw_boot  = bootstrap_memberwise_trends(
        seas5_annual_member_raw, value_col="u10_event_anom",
        n_boot=N_BOOT, seed=RANDOM_SEED)
    seas5_det_boot  = bootstrap_memberwise_trends(
        seas5_annual_member_det, value_col="u10_event_anom",
        n_boot=N_BOOT, seed=RANDOM_SEED + 1)
    seas5_freq_boot = bootstrap_seas5_rolling_frequency_trends(
        seas5_occ_member, window_years=ROLLING_WINDOW_YEARS,
        n_boot=N_BOOT, seed=RANDOM_SEED + 2)
    seas5_frac_boot = bootstrap_seas5_occurrence_fraction_trends(
        seas5_occ_member, n_boot=N_BOOT, seed=RANDOM_SEED + 3)
    
    # Convert trends from per year to per decade
    era5_raw_slope  *= 10.0
    era5_det_slope  *= 10.0
    era5_freq_slope *= 10.0

    seas5_raw_boot  *= 10.0
    seas5_det_boot  *= 10.0
    seas5_freq_boot *= 10.0

    # --- Save summary CSV ---
    summary_df = pd.DataFrame([
        summarize_bootstrap_case(seas5_raw_boot,  era5_raw_slope,  era5_raw_p,  "amplitude_raw"),
        summarize_bootstrap_case(seas5_det_boot,  era5_det_slope,  era5_det_p,  "amplitude_detrended"),
        summarize_bootstrap_case(seas5_freq_boot, era5_freq_slope, era5_freq_p, "frequency_rolling_bootstrap"),
        summarize_bootstrap_case(seas5_frac_boot, seas5_frac_slope, seas5_frac_p, "seas5_fraction_check"),
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary CSV: {SUMMARY_CSV}")

    # --- Plot ---
    print(f"\nPlotting -> {FIG_OUT}")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), constrained_layout=False)
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.99, bottom=0.18, top=0.87)

    amp_xlim = (-30, 30)

    plot_trend_distribution(
        axes[0], seas5_raw_boot, era5_raw_slope, era5_raw_p,
        title="(a) Raw",
        xlabel=r"Trend in SSW U10 amplitude (m s$^{-1}$ decade$^{-1}$)",
        trend_unit=r"m s$^{-1}$ decade$^{-1}$",
        xlim=amp_xlim, show_ylabel=True,
    )
    plot_trend_distribution(
        axes[1], seas5_det_boot, era5_det_slope, era5_det_p,
        title="(b) Detrended",
        xlabel=r"Trend in SSW U10 amplitude (m s$^{-1}$ decade$^{-1}$)",
        trend_unit=r"m s$^{-1}$ decade$^{-1}$",
        xlim=amp_xlim, show_ylabel=False,
    )
    plot_trend_distribution(
        axes[2], seas5_freq_boot, era5_freq_slope, era5_freq_p,
        title=f"(c) SSW frequency",
        xlabel=r"Trend in 10-year rolling SSW frequency (events decade$^{-1}$)",
        trend_unit=r"events decade$^{-1}$",
        xlim=None, show_ylabel=False,
    )

    plt.savefig(FIG_OUT, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {FIG_OUT}")

    gc.collect()
    print("All done.")


if __name__ == "__main__":
    main()