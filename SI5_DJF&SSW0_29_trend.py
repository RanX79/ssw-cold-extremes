# -*- coding: utf-8 -*-
"""
SEAS5 vs ERA5: DJF-mean + SSW post-onset day 0-29 mean U at 60N
- Select pressure level (10 or 100 hPa) via PLOT_LEVEL setting
- Row 1: DJF-mean U trend distribution
- Row 2: SSW day 0-29 mean U trend distribution
- 2 panels per row: Raw and Detrended
"""

import gc
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

# ================================================================
# SETTINGS
# ================================================================
PLOT_LEVEL = 100   # <--- SET TO 10 OR 100

ERA5_U_FILE = Path(r"F:\data\ERA5_data\ERA5_u_daily_1940_2025_100_no229.nc")

SEAS5_U_DIRS = {
    10:  Path(r"F:\data\IFS_U10_daily"),
    100: Path(r"F:\data\IFS_U100_daily"),
}
SEAS5_FILE_TMPLS = {
    10:  "SEAS5_u10hPa_NH_{year}11_system51_m25_daily.nc",
    100: "SEAS5_u100hPa_NH_{year}11_system51_m25_daily.nc",
}

SEAS5_SSW_CSV_PATH = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
ERA5_SSW_CSV_PATH  = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure\ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv")

START_YEAR       = 1981
END_YEAR         = 2024
N_MEMBERS        = 25
BASELINE_START   = 1981
BASELINE_END     = 2010
TARGET_LAT       = 60.0
DJF_MONTHS       = [12, 1, 2]
POST_ONSET_DAYS  = 29
MONTHS_NDJFM     = [11, 12, 1, 2, 3]
N_BOOT           = 5000
RANDOM_SEED      = 42

OUTPUT_DIR = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT = OUTPUT_DIR / f"SI5_DJF_SSW0_29_U{PLOT_LEVEL}_trend_bootstrap_{BASELINE_END}.pdf"

# NPZ caches - DJF
DJF_NPZ = {
    (10,  "raw"): OUTPUT_DIR / f"SI5_DJF_U10_raw_annual_{BASELINE_END}.npz",
    (10,  "det"): OUTPUT_DIR / f"SI5_DJF_U10_det_annual_{BASELINE_END}.npz",
    (100, "raw"): OUTPUT_DIR / f"SI5_DJF_U100_raw_annual_{BASELINE_END}.npz",
    (100, "det"): OUTPUT_DIR / f"SI5_DJF_U100_det_annual_{BASELINE_END}.npz",
}
DJF_ERA5_NPZ = {
    10:  OUTPUT_DIR / "SI5_DJF_ERA5_U10_annual.npz",
    100: OUTPUT_DIR / "SI5_DJF_ERA5_U100_annual.npz",
}

# NPZ caches - SSW
SSW_NPZ = {
    (10,  "raw"): OUTPUT_DIR / f"SI5_SSW0_29_U10_raw_annual_{BASELINE_END}.npz",
    (10,  "det"): OUTPUT_DIR / f"SI5_SSW0_29_U10_det_annual_{BASELINE_END}.npz",
    (100, "raw"): OUTPUT_DIR / f"SI5_SSW0_29_U100_raw_annual_{BASELINE_END}.npz",
    (100, "det"): OUTPUT_DIR / f"SI5_SSW0_29_U100_det_annual_{BASELINE_END}.npz",
}
SSW_ERA5_NPZ = {
    10:  OUTPUT_DIR / f"SI5_SSW0_29_ERA5_U10_annual_{BASELINE_END}.npz",
    100: OUTPUT_DIR / f"SI5_SSW0_29_ERA5_U100_annual_{BASELINE_END}.npz",
}

C_HIST = "#1C6AB1"
C_ERA5 = "#ED4043"
C_CI90 = "#9EC3E6"
C_CI95 = "#000000"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 10,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ================================================================
# HELPERS
# ================================================================
def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 2:
        return np.nan, np.nan
    s, _, _, p, _ = stats.linregress(x[v], y[v])
    return float(s), float(p)


def get_lat_name(da):
    c = [d for d in da.dims if "lat" in d.lower()]
    return c[0] if c else None


def get_time_name(da):
    for n in ["valid_time", "time"]:
        if n in da.dims: return n
    raise ValueError(f"No time dim: {da.dims}")


# ================================================================
# READ SSW EVENT CSVs
# ================================================================
def read_ssw_era5():
    df = pd.read_csv(ERA5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["month"] = df["ssw_date"].dt.month
    df = df[df["month"].isin(MONTHS_NDJFM)].copy()
    df = df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)
    print(f"[ERA5 SSW] {len(df)} events loaded.")
    return df


def read_ssw_seas5():
    df = pd.read_csv(SEAS5_SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["member"] = df["member"].astype(int)
    df["month"]  = df["ssw_date"].dt.month
    df = df[df["month"].isin(MONTHS_NDJFM)].copy()
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
    print(f"[SEAS5 SSW] {len(df)} events loaded.")
    return df


# ================================================================
# SEAS5: ONE YEAR LOAD
# ================================================================
def load_seas5_year_u(year, lev):
    fp = SEAS5_U_DIRS[lev] / SEAS5_FILE_TMPLS[lev].format(year=year)
    if not fp.exists():
        print(f"  Missing: {fp.name}"); return None
    try:
        ds  = xr.open_dataset(fp)
        var = list(ds.data_vars)[0]
        da  = ds[var].load(); ds.close()

        if "forecast_reference_time" in da.dims and da.sizes["forecast_reference_time"] == 1:
            da = da.squeeze("forecast_reference_time", drop=True)
        if "number" not in da.dims:
            raise ValueError("No number dim")
        da = da.transpose("number", *[d for d in da.dims if d != "number"])
        da = da.isel(number=slice(0, N_MEMBERS))

        lat_name  = get_lat_name(da)
        time_name = get_time_name(da)
        lon_name  = [d for d in da.dims if "lon" in d.lower()][0]

        if da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})

        lat_vals = da[lat_name].values
        lat_idx  = int(np.argmin(np.abs(lat_vals - TARGET_LAT)))
        da = da.isel({lat_name: lat_idx}).mean(dim=lon_name)
        da = da.transpose("number", time_name)

        raw_time = da[time_name].values
        if np.issubdtype(raw_time.dtype, np.timedelta64):
            init_date = pd.Timestamp(f"{year}-11-01")
            times = pd.to_datetime(
                [init_date + pd.Timedelta(td) for td in raw_time]).normalize()
        else:
            times = pd.to_datetime(raw_time).normalize()

        arr           = da.values.astype(np.float32)
        member_labels = [int(da["number"].values[i]) for i in range(arr.shape[0])]
        del da; gc.collect()
        return {"arr": arr, "times": times, "member_labels": member_labels}
    except Exception as e:
        print(f"  Failed year={year} lev={lev}: {e}")
        gc.collect(); return None


# ================================================================
# DJF: ERA5
# ================================================================
def compute_era5_djf_annual(lev):
    print(f"\n[ERA5] Loading U{lev}hPa...")
    ds  = xr.open_dataset(ERA5_U_FILE)
    da  = ds["u"].sel(level=float(lev))
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    lat_vals = da["lat"].values
    lat_idx  = int(np.argmin(np.abs(lat_vals - TARGET_LAT)))
    da   = da.isel(lat=lat_idx).mean(dim="lon").load().astype(np.float32)
    times = pd.to_datetime(da["time"].values).normalize()
    u_vals = da.values.copy()
    ds.close(); del da, ds; gc.collect()

    df = pd.DataFrame({
        "time":     times,
        "u":        u_vals,
        "year":     times.year,
        "month":    times.month,
        "monthday": [month_day_key(t) for t in times],
    })

    df_clim = df[
        (df["year"] >= BASELINE_START) &
        (df["year"] <= BASELINE_END)
    ]
    df_clim = df_clim.copy()
    df_clim["doy"] = pd.to_datetime(df_clim["time"]).dt.dayofyear
    clim_raw = df_clim.groupby("doy")["u"].mean().values
    if len(clim_raw) < 366:
        tmp = np.full(366, np.nan)
        tmp[:len(clim_raw)] = clim_raw
        clim_raw = tmp
    window = 11
    pad = window // 2
    clim_pad = np.concatenate([clim_raw[-pad:], clim_raw, clim_raw[:pad]])
    clim_smooth = np.full(366, np.nan)
    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)
    clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        clim[month_day_key(t)] = clim_smooth[t.dayofyear - 1]

    df["u_anom_raw"] = df.apply(
        lambda r: r["u"] - clim.get(r["monthday"], np.nan), axis=1
    ).astype(np.float32)

    df = df[
        (df["year"] >= START_YEAR) &
        (df["year"] <= END_YEAR)
    ].copy()

    df["u_anom_det"] = df["u_anom_raw"].copy()
    for key, idx in df.groupby("monthday").groups.items():
        sub = df.loc[idx]
        yv  = sub["u_anom_raw"].values.astype(float)
        yr  = sub["year"].values.astype(float)
        v   = np.isfinite(yv)
        if v.sum() < 2: continue
        yr_mean = yr[v].mean()
        yr_c    = yr[v] - yr_mean
        slope   = np.sum(yr_c * yv[v]) / np.sum(yr_c ** 2)
        intercept = yv[v].mean()
        trend   = intercept + slope * (yr - yr_mean)
        df.loc[idx, "u_anom_det"] = (yv - trend).astype(np.float32)

    def winter_year(row):
        if row["month"] == 12:
            return row["year"] + 1
        elif row["month"] in [1, 2]:
            return row["year"]
        return np.nan

    df["winter_year"] = df.apply(winter_year, axis=1)
    df_djf = df[df["month"].isin(DJF_MONTHS)].copy()

    raw_annual = (df_djf.groupby("winter_year")["u_anom_raw"]
                  .mean().rename("u_raw"))
    det_annual = (df_djf.groupby("winter_year")["u_anom_det"]
                  .mean().rename("u_det"))
    result = pd.concat([raw_annual, det_annual], axis=1).reset_index()
    result = result.rename(columns={"winter_year": "year"})
    result = result[(result["year"] >= START_YEAR) & (result["year"] <= END_YEAR)]
    print(f"  ERA5 U{lev} DJF years: {len(result)}")
    return result


def save_era5_djf_npz(path, df):
    np.savez_compressed(
        path,
        year  = df["year"].values.astype(int),
        u_raw = df["u_raw"].values.astype(np.float32),
        u_det = df["u_det"].values.astype(np.float32),
    )
    print(f"  Saved ERA5 DJF: {path.name}")


def load_era5_djf_npz(path):
    d  = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "year":  d["year"].astype(int),
        "u_raw": d["u_raw"].astype(np.float32),
        "u_det": d["u_det"].astype(np.float32),
    })
    print(f"  Loaded ERA5 DJF: {path.name}")
    return df


# ================================================================
# DJF: SEAS5
# ================================================================
def compute_seas5_djf_annual(lev):
    print(f"\n[SEAS5] Computing U{lev}hPa DJF annual means...")

    print(f"  Building rolling climatology {BASELINE_START}-{BASELINE_END}...")
    doy_to_vals = {}
    for year in range(BASELINE_START, BASELINE_END + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        ens_mean = data["arr"].mean(axis=0)
        times = data["times"]
        doys = pd.to_datetime(times).dayofyear.values
        for i, doy in enumerate(doys):
            doy_to_vals.setdefault(doy, []).append(ens_mean[i])
        del data, ens_mean; gc.collect()

    clim_raw = np.full(366, np.nan)
    for doy, vals in doy_to_vals.items():
        clim_raw[doy-1] = np.nanmean(vals)
    del doy_to_vals; gc.collect()

    window = 11
    pad = window // 2
    clim_pad = np.concatenate([clim_raw[-pad:], clim_raw, clim_raw[:pad]])
    clim_smooth = np.full(366, np.nan)
    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)
    clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        clim[month_day_key(t)] = clim_smooth[t.dayofyear - 1]
    print(f"  Climatology keys: {len(clim)} (smoothed)")

    raw_records   = []
    em_by_day_yr  = {}

    for year in range(START_YEAR, END_YEAR + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        arr    = data["arr"]
        times  = data["times"]
        mlabels= data["member_labels"]

        anom = np.full_like(arr, np.nan)
        for i, t in enumerate(times):
            cf = clim.get(month_day_key(t))
            if cf is not None:
                anom[:, i] = arr[:, i] - cf

        for i, t in enumerate(times):
            m = t.month
            if m == 12 and t.year == year:
                wy = year + 1
            elif m in [1, 2] and t.year == year + 1:
                wy = year + 1
            else:
                continue
            for mi, mem in enumerate(mlabels):
                raw_records.append({
                    "year": wy, "member": mem,
                    "u_raw": float(anom[mi, i])
                })
            em = float(np.nanmean(anom[:, i]))
            key = (t.month, t.day)
            em_by_day_yr.setdefault(key, []).append((year, em))

        del data, arr, anom; gc.collect()
        print(f"  {year} done")

    raw_df = pd.DataFrame(raw_records)
    del raw_records; gc.collect()

    print("  Building detrend offsets...")
    detrend_offset = {}
    for key, entries in em_by_day_yr.items():
        if len(entries) < 2: continue
        yr_arr = np.array([e[0] for e in entries], float)
        em_arr = np.array([e[1] for e in entries], float)
        v = np.isfinite(em_arr)
        if v.sum() < 2: continue
        yr_mean = yr_arr[v].mean()
        yr_c    = yr_arr[v] - yr_mean
        slope   = np.sum(yr_c * em_arr[v]) / np.sum(yr_c ** 2)
        intercept = em_arr[v].mean()
        for yi, y in enumerate(yr_arr.astype(int)):
            detrend_offset[(y, key[0], key[1])] = float(
                intercept + slope * (y - yr_mean))
    del em_by_day_yr; gc.collect()

    det_records = []
    for year in range(START_YEAR, END_YEAR + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        arr    = data["arr"]
        times  = data["times"]
        mlabels= data["member_labels"]

        anom = np.full_like(arr, np.nan)
        for i, t in enumerate(times):
            cf = clim.get(month_day_key(t))
            if cf is not None:
                anom[:, i] = arr[:, i] - cf

        for i, t in enumerate(times):
            m = t.month
            if m == 12 and t.year == year:
                wy = year + 1
            elif m in [1, 2] and t.year == year + 1:
                wy = year + 1
            else:
                continue
            offset = detrend_offset.get((year, t.month, t.day), 0.0)
            for mi, mem in enumerate(mlabels):
                det_records.append({
                    "year": wy, "member": mem,
                    "u_det": float(anom[mi, i]) - offset
                })

        del data, arr, anom; gc.collect()
        print(f"  Detrend {year} done")

    det_df = pd.DataFrame(det_records)
    del det_records; gc.collect()

    raw_annual = raw_df.groupby(["year", "member"])["u_raw"].mean().reset_index()
    det_annual = det_df.groupby(["year", "member"])["u_det"].mean().reset_index()
    merged = pd.merge(raw_annual, det_annual, on=["year", "member"])
    merged = merged[(merged["year"] >= START_YEAR) & (merged["year"] <= END_YEAR)]
    print(f"  SEAS5 U{lev} DJF records: {len(merged)}")
    return merged


def save_seas5_djf_npz(path, df):
    np.savez_compressed(
        path,
        year   = df["year"].values.astype(int),
        member = df["member"].values.astype(int),
        u_raw  = df["u_raw"].values.astype(np.float32),
        u_det  = df["u_det"].values.astype(np.float32),
    )
    print(f"  Saved: {path.name}")


def load_seas5_djf_npz(path):
    d  = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "year":   d["year"].astype(int),
        "member": d["member"].astype(int),
        "u_raw":  d["u_raw"].astype(np.float32),
        "u_det":  d["u_det"].astype(np.float32),
    })
    print(f"  Loaded: {path.name}")
    return df


# ================================================================
# SSW: ERA5
# ================================================================
def load_era5_daily_u(lev):
    print(f"\n[ERA5] Loading U{lev}hPa daily...")
    ds  = xr.open_dataset(ERA5_U_FILE)
    da  = ds["u"].sel(level=float(lev))
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    lat_vals = da["lat"].values
    lat_idx  = int(np.argmin(np.abs(lat_vals - TARGET_LAT)))
    da   = da.isel(lat=lat_idx).mean(dim="lon").load().astype(np.float32)
    times = pd.to_datetime(da["time"].values).normalize()
    u_vals = da.values.copy()
    ds.close(); del da, ds; gc.collect()

    df = pd.DataFrame({
        "time":     times,
        "u":        u_vals,
        "year":     times.year,
        "month":    times.month,
        "monthday": [month_day_key(t) for t in times],
    })
    print(f"  ERA5 U{lev}: {len(df)} daily records")
    return df


def build_era5_raw_anomaly(era5_df):
    df_clim = era5_df[
        (era5_df["year"] >= BASELINE_START) &
        (era5_df["year"] <= BASELINE_END)
    ].copy()
    df_clim["doy"] = pd.to_datetime(df_clim["time"]).dt.dayofyear
    clim_raw = df_clim.groupby("doy")["u"].mean().values
    if len(clim_raw) < 366:
        tmp = np.full(366, np.nan)
        tmp[:len(clim_raw)] = clim_raw
        clim_raw = tmp
    window = 11
    pad = window // 2
    clim_pad = np.concatenate([clim_raw[-pad:], clim_raw, clim_raw[:pad]])
    clim_smooth = np.full(366, np.nan)
    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)
    clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        clim[month_day_key(t)] = clim_smooth[t.dayofyear - 1]
    out = era5_df.copy()
    out["u_anom"] = out.apply(
        lambda r: r["u"] - clim.get(r["monthday"], np.nan), axis=1
    ).astype(np.float32)
    return out


def build_era5_detrended_anomaly(era5_anom_df):
    out = era5_anom_df.copy()
    out["u_anom_det"] = out["u_anom"].copy()
    for key, idx in out.groupby("monthday").groups.items():
        sub = out.loc[idx]
        yv = sub["u_anom"].values.astype(float)
        yr = sub["year"].values.astype(float)
        fit_mask = np.isfinite(yv)
        if fit_mask.sum() < 2: continue
        yr_mean = yr[fit_mask].mean()
        yr_c    = yr[fit_mask] - yr_mean
        slope = np.sum(yr_c * yv[fit_mask]) / np.sum(yr_c ** 2)
        intercept = yv[fit_mask].mean()
        trend = intercept + slope * (yr - yr_mean)
        out.loc[idx, "u_anom_det"] = (yv - trend).astype(np.float32)
    gc.collect()
    return out


def compute_era5_ssw_annual(era5_anom_df, ssw_era5, anom_col="u_anom"):
    times    = pd.to_datetime(era5_anom_df["time"].values)
    u_arr    = era5_anom_df[anom_col].values
    time_idx = {pd.Timestamp(t): i for i, t in enumerate(times)}

    records = []
    for _, row in ssw_era5.iterrows():
        center = pd.Timestamp(row["ssw_date"]).normalize()
        i0 = time_idx.get(center)
        if i0 is None: continue
        i1 = min(i0 + POST_ONSET_DAYS + 1, len(u_arr))
        window = u_arr[i0:i1]
        if not np.isfinite(window).any(): continue
        records.append({
            "init_year": int(row["init_year"]),
            "u_mean":    float(np.nanmean(window)),
        })

    if not records:
        return pd.DataFrame(columns=["init_year", "u_mean"])

    df_ev = pd.DataFrame(records)
    annual = df_ev.groupby("init_year")["u_mean"].mean().reset_index()
    annual = annual[(annual["init_year"] >= START_YEAR) & (annual["init_year"] <= END_YEAR)]
    print(f"  ERA5 SSW annual ({anom_col}): {len(annual)} years with events")
    return annual


def save_era5_ssw_npz(path, df):
    np.savez_compressed(
        path,
        year  = df["init_year"].values.astype(int),
        u_raw = df["u_raw"].values.astype(np.float32),
        u_det = df["u_det"].values.astype(np.float32),
    )
    print(f"  Saved ERA5 SSW: {path.name}")


def load_era5_ssw_npz(path):
    d  = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "init_year": d["year"].astype(int),
        "u_raw":     d["u_raw"].astype(np.float32),
        "u_det":     d["u_det"].astype(np.float32),
    })
    print(f"  Loaded ERA5 SSW: {path.name}")
    return df


# ================================================================
# SSW: SEAS5
# ================================================================
def compute_seas5_ssw_annual(lev, ssw_seas5):
    print(f"\n[SEAS5] Computing U{lev}hPa SSW day 0-29 annual means...")

    print(f"  Building rolling climatology {BASELINE_START}-{BASELINE_END}...")
    doy_to_vals = {}
    for year in range(BASELINE_START, BASELINE_END + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        ens_mean = data["arr"].mean(axis=0)
        times = data["times"]
        doys = pd.to_datetime(times).dayofyear.values
        for i, doy in enumerate(doys):
            doy_to_vals.setdefault(doy, []).append(ens_mean[i])
        del data, ens_mean; gc.collect()

    clim_raw = np.full(366, np.nan)
    for doy, vals in doy_to_vals.items():
        clim_raw[doy - 1] = np.nanmean(vals)
    del doy_to_vals; gc.collect()

    window = 11
    pad = window // 2
    clim_pad = np.concatenate([clim_raw[-pad:], clim_raw, clim_raw[:pad]])
    clim_smooth = np.full(366, np.nan)
    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win)
    clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        clim[month_day_key(t)] = clim_smooth[t.dayofyear - 1]
    print(f"  Climatology keys: {len(clim)} (smoothed)")

    raw_records  = []
    em_by_day_yr = {}

    for year in range(START_YEAR, END_YEAR + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        arr     = data["arr"]
        times   = data["times"]
        mlabels = data["member_labels"]
        t2i = {t: i for i, t in enumerate(times)}

        anom = np.full_like(arr, np.nan)
        for i, t in enumerate(times):
            cf = clim.get(month_day_key(t))
            if cf is not None:
                anom[:, i] = arr[:, i] - cf

        for i, t in enumerate(times):
            em = float(np.nanmean(anom[:, i]))
            key = (t.month, t.day)
            if 1981 <= year <= 2024:
                em_by_day_yr.setdefault(key, []).append((year, em))

        dfy = ssw_seas5[ssw_seas5["init_year"] == year]
        for mi, mem in enumerate(mlabels):
            dfm = dfy[dfy["member"] == mem]
            if dfm.empty: continue
            vals = []
            for _, row in dfm.iterrows():
                center = pd.Timestamp(row["ssw_date"]).normalize()
                i0 = t2i.get(center)
                if i0 is None: continue
                i1 = min(i0 + POST_ONSET_DAYS + 1, arr.shape[1])
                w = anom[mi, i0:i1]
                if np.isfinite(w).any():
                    vals.append(float(np.nanmean(w)))
            if vals:
                raw_records.append({
                    "init_year": year,
                    "member":    mem,
                    "u_raw":     float(np.mean(vals)),
                })

        del data, arr, anom; gc.collect()
        print(f"  {year} done (raw)")

    raw_df = pd.DataFrame(raw_records)
    del raw_records; gc.collect()

    print("  Building detrend offsets...")
    detrend_offset = {}
    for key, entries in em_by_day_yr.items():
        if len(entries) < 2: continue
        yr_arr = np.array([e[0] for e in entries], float)
        em_arr = np.array([e[1] for e in entries], float)
        v = np.isfinite(em_arr)
        if v.sum() < 2: continue
        yr_mean   = yr_arr[v].mean()
        yr_c      = yr_arr[v] - yr_mean
        slope     = np.sum(yr_c * em_arr[v]) / np.sum(yr_c ** 2)
        intercept = em_arr[v].mean()
        for yi, y in enumerate(yr_arr.astype(int)):
            detrend_offset[(y, key[0], key[1])] = float(
                intercept + slope * (y - yr_mean))
    del em_by_day_yr; gc.collect()

    det_records = []
    for year in range(START_YEAR, END_YEAR + 1):
        data = load_seas5_year_u(year, lev)
        if data is None: continue
        arr     = data["arr"]
        times   = data["times"]
        mlabels = data["member_labels"]
        t2i     = {t: i for i, t in enumerate(times)}

        anom = np.full_like(arr, np.nan)
        for i, t in enumerate(times):
            cf = clim.get(month_day_key(t))
            if cf is not None:
                anom[:, i] = arr[:, i] - cf

        det_anom = anom.copy()
        for i, t in enumerate(times):
            offset = detrend_offset.get((year, t.month, t.day), 0.0)
            det_anom[:, i] = anom[:, i] - offset

        dfy = ssw_seas5[ssw_seas5["init_year"] == year]
        for mi, mem in enumerate(mlabels):
            dfm = dfy[dfy["member"] == mem]
            if dfm.empty: continue
            vals = []
            for _, row in dfm.iterrows():
                center = pd.Timestamp(row["ssw_date"]).normalize()
                i0 = t2i.get(center)
                if i0 is None: continue
                i1 = min(i0 + POST_ONSET_DAYS + 1, arr.shape[1])
                w = det_anom[mi, i0:i1]
                if np.isfinite(w).any():
                    vals.append(float(np.nanmean(w)))
            if vals:
                det_records.append({
                    "init_year": year,
                    "member":    mem,
                    "u_det":     float(np.mean(vals)),
                })

        del data, arr, anom, det_anom; gc.collect()
        print(f"  {year} done (det)")

    det_df = pd.DataFrame(det_records)
    del det_records; gc.collect()

    merged = pd.merge(raw_df, det_df, on=["init_year", "member"])
    merged = merged[(merged["init_year"] >= START_YEAR) & (merged["init_year"] <= END_YEAR)]
    print(f"  SEAS5 U{lev} SSW records: {len(merged)}")
    return merged


def save_seas5_ssw_npz(path, df):
    np.savez_compressed(
        path,
        year   = df["init_year"].values.astype(int),
        member = df["member"].values.astype(int),
        u_raw  = df["u_raw"].values.astype(np.float32),
        u_det  = df["u_det"].values.astype(np.float32),
    )
    print(f"  Saved: {path.name}")


def load_seas5_ssw_npz(path):
    d  = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "init_year": d["year"].astype(int),
        "member":    d["member"].astype(int),
        "u_raw":     d["u_raw"].astype(np.float32),
        "u_det":     d["u_det"].astype(np.float32),
    })
    print(f"  Loaded: {path.name}")
    return df


# ================================================================
# BOOTSTRAP
# ================================================================
def bootstrap_seas5_djf_trend(seas5_df, col, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng         = np.random.default_rng(seed)
    years_all   = np.arange(START_YEAR, END_YEAR + 1)
    members_all = seas5_df["member"].unique()
    lookup = {}
    for _, row in seas5_df.iterrows():
        lookup[(int(row["year"]), int(row["member"]))] = float(row[col])
    slopes = np.full(n_boot, np.nan)
    for b in range(n_boot):
        yrs, vals = [], []
        for y in years_all:
            mem = int(rng.choice(members_all))
            v   = lookup.get((y, mem), np.nan)
            if np.isfinite(v):
                yrs.append(y); vals.append(v)
        if len(yrs) >= 2:
            slopes[b], _ = ols_slope(yrs, vals)
    return slopes


def bootstrap_seas5_ssw_trend(seas5_df, col, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng         = np.random.default_rng(seed)
    years_all   = np.arange(START_YEAR, END_YEAR + 1)
    members_all = seas5_df["member"].unique()
    lookup = {}
    for _, row in seas5_df.iterrows():
        lookup[(int(row["init_year"]), int(row["member"]))] = float(row[col])
    slopes = np.full(n_boot, np.nan)
    for b in range(n_boot):
        yrs, vals = [], []
        for y in years_all:
            mem = int(rng.choice(members_all))
            v   = lookup.get((y, mem), np.nan)
            if np.isfinite(v):
                yrs.append(y); vals.append(v)
        if len(yrs) >= 2:
            slopes[b], _ = ols_slope(yrs, vals)
    return slopes


# ================================================================
# PLOT ONE PANEL
# ================================================================
def plot_panel(ax, model_slopes, era5_slope, era5_p, title,
               xlabel, xlim=None, show_ylabel=True):
    vals = model_slopes[np.isfinite(model_slopes)]
    p025, p05  = np.percentile(vals, [2.5, 5.0])
    p95,  p975 = np.percentile(vals, [95.0, 97.5])

    n, bins, _ = ax.hist(vals, bins=45, color=C_HIST, alpha=0.82,
                         edgecolor="white", linewidth=0.35, zorder=2)
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
    ax.axvline(seas5_mean, color="#2CA02C", lw=1.8, ls="-.", zorder=5)

    x_span = np.nanmax(vals) - np.nanmin(vals) if len(vals) > 1 else 1.0
    x_off  = 0.15 * x_span if x_span > 0 else 0.2

    is_right_era5 = era5_slope >= np.nanmedian(vals)
    x_text_era5   = era5_slope + (x_off if is_right_era5 else -x_off)
    ha_era5       = "left" if is_right_era5 else "right"

    ax.annotate(
         f"ERA5\n{era5_slope:+.2f} m/s/decade\np={era5_p:.3f}",
        xy=(era5_slope, ymax * 0.68),
        xytext=(x_text_era5, ymax * 0.74),
        fontsize=6.5, color=C_ERA5, ha=ha_era5, va="center",fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_ERA5, lw=0.9, mutation_scale=6),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_ERA5,
                  lw=0.6, alpha=0.92),
        zorder=6,
    )

    C_SEAS5     = "#2CA02C"
    is_right_s5 = seas5_mean >= np.nanmedian(vals)
    if is_right_s5 == is_right_era5:
        s5_off = x_off * 2.2
    else:
        s5_off = x_off
    x_text_s5 = seas5_mean + (s5_off if is_right_s5 else -s5_off)
    ha_s5     = "left" if is_right_s5 else "right"

    ax.annotate(
        f"SEAS5 mean\n{seas5_mean:+.2f} m/s/decade",
        xy=(seas5_mean, ymax * 0.48),
        xytext=(x_text_s5, ymax * 0.54),
        fontsize=6.5, color=C_SEAS5, ha=ha_s5, va="center",fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=C_SEAS5, lw=0.9, mutation_scale=6),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_SEAS5,
                  lw=0.6, alpha=0.92),
        zorder=6,
    )

    if xlim is None:
        xmin = min(np.nanmin(vals), era5_slope) - 0.2 * x_span
        xmax = max(np.nanmax(vals), era5_slope) + 0.2 * x_span
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(*xlim)

    ax.set_ylim(0, ymax * 1.35)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
    ax.set_xlabel(xlabel, labelpad=4,fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Bootstrap count", labelpad=4,fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=5)

    legend_handles = [
        mpatches.Patch(color=C_HIST, alpha=0.82, label="SEAS5 bootstrap"),
        mpatches.Patch(color=C_CI90, alpha=0.32, label="5–95% range"),
        mpatches.Patch(color=C_CI95, alpha=0.20, label="2.5–97.5% range"),
        Line2D([0], [0], color=C_ERA5,    lw=1.8, ls="--",  label="ERA5 trend"),
        Line2D([0], [0], color="#2CA02C", lw=1.8, ls="-.",  label="SEAS5 mean"),
        Line2D([0], [0], color="black",   lw=0.9, ls="-",   label="Zero line"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=True, framealpha=0.92, edgecolor="0.68",
              handlelength=1.6, handletextpad=0.45,
              borderpad=0.45, labelspacing=0.32)
    ax.yaxis.grid(True, ls="--", lw=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


# ================================================================
# MAIN
# ================================================================
def main():
    lev = PLOT_LEVEL
    print("=" * 70)
    print(f"DJF-mean + SSW day 0-29 U{lev}hPa at 60N: ERA5 vs SEAS5 bootstrap")
    print("=" * 70)

    ssw_era5  = read_ssw_era5()
    ssw_seas5 = read_ssw_seas5()

    # ----------------------------------------------------------------
    # DJF: ERA5
    # ----------------------------------------------------------------
    if DJF_ERA5_NPZ[lev].exists():
        era5_djf = load_era5_djf_npz(DJF_ERA5_NPZ[lev])
    else:
        era5_djf = compute_era5_djf_annual(lev)
        save_era5_djf_npz(DJF_ERA5_NPZ[lev], era5_djf)

    # DJF: SEAS5
    raw_path = DJF_NPZ[(lev, "raw")]
    det_path = DJF_NPZ[(lev, "det")]
    if raw_path.exists() and det_path.exists():
        raw_df = load_seas5_djf_npz(raw_path)
        det_df = load_seas5_djf_npz(det_path)
        seas5_djf = pd.merge(
            raw_df[["year", "member", "u_raw"]],
            det_df[["year", "member", "u_det"]],
            on=["year", "member"]
        )
    else:
        seas5_djf = compute_seas5_djf_annual(lev)
        np.savez_compressed(
            raw_path,
            year   = seas5_djf["year"].values.astype(int),
            member = seas5_djf["member"].values.astype(int),
            u_raw  = seas5_djf["u_raw"].values.astype(np.float32),
            u_det  = seas5_djf["u_det"].values.astype(np.float32),
        )
        np.savez_compressed(
            det_path,
            year   = seas5_djf["year"].values.astype(int),
            member = seas5_djf["member"].values.astype(int),
            u_raw  = seas5_djf["u_raw"].values.astype(np.float32),
            u_det  = seas5_djf["u_det"].values.astype(np.float32),
        )
        print(f"  Saved: {raw_path.name}, {det_path.name}")
    gc.collect()

    # ----------------------------------------------------------------
    # SSW: ERA5
    # ----------------------------------------------------------------
    if SSW_ERA5_NPZ[lev].exists():
        era5_ssw = load_era5_ssw_npz(SSW_ERA5_NPZ[lev])
    else:
        era5_df  = load_era5_daily_u(lev)
        anom_raw = build_era5_raw_anomaly(era5_df)
        anom_raw = anom_raw[
            (anom_raw["year"] >= START_YEAR) &
            (anom_raw["year"] <= END_YEAR)
        ].copy()
        anom_det = build_era5_detrended_anomaly(anom_raw)
        del era5_df; gc.collect()

        annual_raw = compute_era5_ssw_annual(anom_raw, ssw_era5, anom_col="u_anom")
        annual_det = compute_era5_ssw_annual(anom_det, ssw_era5, anom_col="u_anom_det")
        del anom_raw, anom_det; gc.collect()

        era5_ssw = pd.merge(
            annual_raw.rename(columns={"u_mean": "u_raw"}),
            annual_det.rename(columns={"u_mean": "u_det"}),
            on="init_year",
        )
        save_era5_ssw_npz(SSW_ERA5_NPZ[lev], era5_ssw)

    # SSW: SEAS5
    raw_path_ssw = SSW_NPZ[(lev, "raw")]
    det_path_ssw = SSW_NPZ[(lev, "det")]
    if raw_path_ssw.exists() and det_path_ssw.exists():
        raw_df = load_seas5_ssw_npz(raw_path_ssw)
        det_df = load_seas5_ssw_npz(det_path_ssw)
        seas5_ssw = pd.merge(
            raw_df[["init_year", "member", "u_raw"]],
            det_df[["init_year", "member", "u_det"]],
            on=["init_year", "member"]
        )
    else:
        seas5_ssw = compute_seas5_ssw_annual(lev, ssw_seas5)
        save_seas5_ssw_npz(raw_path_ssw, seas5_ssw)
        save_seas5_ssw_npz(det_path_ssw, seas5_ssw)
        print(f"  Saved: {raw_path_ssw.name}, {det_path_ssw.name}")
    gc.collect()

    # ----------------------------------------------------------------
    # Bootstrap & ERA5 slopes
    # ----------------------------------------------------------------
    print("\nBootstrapping DJF...")
    era5_djf_raw_slope, era5_djf_raw_p = ols_slope(era5_djf["year"], era5_djf["u_raw"])
    era5_djf_det_slope, era5_djf_det_p = ols_slope(era5_djf["year"], era5_djf["u_det"])
    print(f"  ERA5 U{lev} DJF raw: {era5_djf_raw_slope:+.4f} m/s/yr  p={era5_djf_raw_p:.3f}")
    print(f"  ERA5 U{lev} DJF det: {era5_djf_det_slope:+.4f} m/s/yr  p={era5_djf_det_p:.3f}")
    boot_djf_raw = bootstrap_seas5_djf_trend(seas5_djf, "u_raw", seed=RANDOM_SEED + lev)
    boot_djf_det = bootstrap_seas5_djf_trend(seas5_djf, "u_det", seed=RANDOM_SEED + lev + 1)

    print("\nBootstrapping SSW...")
    era5_ssw_raw_slope, era5_ssw_raw_p = ols_slope(era5_ssw["init_year"], era5_ssw["u_raw"])
    era5_ssw_det_slope, era5_ssw_det_p = ols_slope(era5_ssw["init_year"], era5_ssw["u_det"])
    print(f"  ERA5 U{lev} SSW raw: {era5_ssw_raw_slope:+.4f} m/s/yr  p={era5_ssw_raw_p:.3f}")
    print(f"  ERA5 U{lev} SSW det: {era5_ssw_det_slope:+.4f} m/s/yr  p={era5_ssw_det_p:.3f}")
    boot_ssw_raw = bootstrap_seas5_ssw_trend(seas5_ssw, "u_raw", seed=RANDOM_SEED + lev)
    boot_ssw_det = bootstrap_seas5_ssw_trend(seas5_ssw, "u_det", seed=RANDOM_SEED + lev + 1)
    # ==========================================================
    # Convert trends from yr^-1 to decade^-1
    # ==========================================================
    era5_djf_raw_slope *= 10.0
    era5_djf_det_slope *= 10.0

    era5_ssw_raw_slope *= 10.0
    era5_ssw_det_slope *= 10.0

    boot_djf_raw *= 10.0
    boot_djf_det *= 10.0

    boot_ssw_raw *= 10.0
    boot_ssw_det *= 10.0

    # ----------------------------------------------------------------
    # Plot: 2x2
    # ----------------------------------------------------------------
    xlim_map = {10: (-6, 6), 100: (-3, 3)}
    xl = xlim_map[lev]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(wspace=0.28, hspace=0.38,
                        left=0.07, right=0.98,
                        bottom=0.10, top=0.93)

    # Row 1: DJF
    plot_panel(
        axes[0, 0], boot_djf_raw,
        era5_djf_raw_slope, era5_djf_raw_p,
        title=f"(a) DJF raw anomalies",
        xlabel="Trend (m s$^{-1}$ decade$^{-1}$)",
        xlim=xl, show_ylabel=True,
    )
    plot_panel(
        axes[0, 1], boot_djf_det,
        era5_djf_det_slope, era5_djf_det_p,
        title=f"(b) DJF detrended anomalies",
        xlabel="Trend (m s$^{-1}$ decade$^{-1}$)",
        xlim=xl, show_ylabel=False,
    )

    # Row 2: SSW day 0-29
    plot_panel(
        axes[1, 0], boot_ssw_raw,
        era5_ssw_raw_slope, era5_ssw_raw_p,
        title=f"(c) Post-SSW raw anomalies",
        xlabel="Trend (m s$^{-1}$ decade$^{-1}$)",
        xlim=xl, show_ylabel=True,
    )
    plot_panel(
        axes[1, 1], boot_ssw_det,
        era5_ssw_det_slope, era5_ssw_det_p,
        title=f"(d) Post-SSW detrended anomalies",
        xlabel="Trend (m s$^{-1}$ decade$^{-1}$)",
        xlim=xl, show_ylabel=False,
    )

    # fig.suptitle(
    #     f"SEAS5 vs ERA5: U{lev} hPa at 60°N trend distribution",
    #     fontsize=12, fontweight="bold"
    # )

    plt.savefig(FIG_OUT, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {FIG_OUT}")
    gc.collect()
    print("All done.")


if __name__ == "__main__":
    main()