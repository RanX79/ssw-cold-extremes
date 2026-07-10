# -*- coding: utf-8 -*-
"""
1) p10 figure: left = Raw, right = Detrended
2) mean figure: left = Raw, right = Detrended


Settings:
- U strength uses day -10 to +10 mean anomaly at 60N
- T2m uses only day +15 to +59
- NPZ cache: load if exists, otherwise compute and save
"""


import gc
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


from pathlib import Path
from scipy.stats import linregress
from scipy.stats import t as t_dist, f as f_dist
from matplotlib.lines import Line2D



# ================================================================
# GLOBAL STYLE
# ================================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.8,
    "axes.labelweight": "normal",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})


COL_EARLY = "#2B6CB0"
COL_LATE  = "#C53030"
COL_GRID  = "#D9D9D9"
COL_SPINE = "#222222"
COL_TEXT  = "#111111"
COL_ZERO  = "#8A8A8A"



# ================================================================
# USER SETTINGS
# ================================================================
SSW_CSV_PATH  = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
T2M_DAILY_DIR = Path(r"F:\data\IFS_t2m_daily")
OUTPUT_DIR    = Path(r"F:\data\paper_SSW_impacts_under_global _warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR     = 1981
END_YEAR       = 2024
N_MEMBERS      = 25
BASELINE_START = 1981
BASELINE_END   = 2010


U_DAY_START = 0
U_DAY_END   = 29


T_DAY_START = 0
T_DAY_END   = 29
U_LEVEL = 100


if U_LEVEL == 10:
    U_DIR       = Path(r"F:\data\IFS_U10_daily")
    U_FILE_TMPL = "SEAS5_u10hPa_NH_{year}11_system51_m25_daily.nc"
    U_LABEL     = "U10hPa"
elif U_LEVEL == 100:
    U_DIR       = Path(r"F:\data\IFS_U100_daily")
    U_FILE_TMPL = "SEAS5_u100hPa_NH_{year}11_system51_m25_daily.nc"
    U_LABEL     = "U100hPa"
else:
    raise ValueError(f"U_LEVEL must be 10 or 100, got {U_LEVEL}")


NPZ_CACHE = OUTPUT_DIR / f"SI6_regression_sensitivity_cache_{U_LABEL}_u{U_DAY_START}to{U_DAY_END}_t{T_DAY_START}to{T_DAY_END}_NDJFM_{BASELINE_END}.npz"




REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": 220, "lon_max": 300},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min":   0, "lon_max":  40},
    "EastAsia":      {"lat_min": 45, "lat_max": 70, "lon_min":  60, "lon_max": 120},
}
REGION_LABELS = {
    "NorthAmerica": "North America",
    "Europe":       "Europe",
    "EastAsia":      "EastAsia",
}
REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]


T2M_VAR_CANDIDATES = ["2m_temperature"]
U_LAT_TARGET = 60.0



# ================================================================
# HELPERS
# ================================================================
def style_ax(ax, add_minor=False):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COL_SPINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", colors=COL_TEXT, length=3.2, width=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color=COL_GRID, alpha=0.55)
    ax.set_axisbelow(True)
    if add_minor:
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())



def add_panel_label(ax, label):
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold", color=COL_TEXT)



def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"



def window_dates(center_date, day_start, day_end):
    center_date = pd.Timestamp(center_date)
    return pd.date_range(
        center_date + pd.Timedelta(days=day_start),
        center_date + pd.Timedelta(days=day_end)
    )



def get_var_name(ds):
    for v in T2M_VAR_CANDIDATES:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError("Cannot identify t2m variable")



def get_lat_lon_names(da):
    lat = [d for d in da.dims if "lat" in d.lower()]
    lon = [d for d in da.dims if "lon" in d.lower()]
    return lat[0], lon[0]



def get_time_name(da):
    for n in ["valid_time", "time"]:
        if n in da.dims:
            return n
    raise ValueError("Cannot find time dim")



def _lon360_to_180(lon):
    return lon - 360 if lon > 180 else lon



def cosine_weighted_mean(field2d, lat_vals, lon_vals, rb):
    lon_min = _lon360_to_180(rb["lon_min"])
    lon_max = _lon360_to_180(rb["lon_max"])
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    if lon_min > lon_max:
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)
    else:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    sub     = field2d[np.ix_(lat_mask, lon_mask)]
    weights = np.cos(np.deg2rad(lat_vals[lat_mask]))
    w2d     = weights[:, None] * np.ones(lon_mask.sum())[None, :]
    valid   = np.isfinite(sub)
    if valid.sum() == 0:
        return np.nan
    return float(np.nansum(sub * w2d * valid) / np.nansum(w2d * valid))



def u_zonal_mean_60n(u2d, lat_vals):
    lat_idx = int(np.argmin(np.abs(lat_vals - U_LAT_TARGET)))
    row   = u2d[lat_idx, :]
    valid = np.isfinite(row)
    if valid.sum() == 0:
        return np.nan
    return float(np.nanmean(row[valid]))



def format_pval(p):
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"



# ================================================================
# T2M — 只扫元数据，不 load 数据
# ================================================================
def load_all_winters():
    print("Scanning SEAS5 daily T2m metadata ...")
    all_data = {}
    lat_vals = lon_vals = None


    for year in range(START_YEAR, END_YEAR + 1):
        fp = T2M_DAILY_DIR / f"SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"
        if not fp.exists():
            print(f"  [WARN] T2m daily file missing: {fp}")
            continue
        try:
            ds  = xr.open_dataset(fp)
            var = get_var_name(ds)
            da  = ds[var]


            for dim in list(da.dims):
                if (da.sizes[dim] == 1
                        and "lat"   not in dim.lower()
                        and "lon"   not in dim.lower()
                        and "time"  not in dim.lower()
                        and "valid" not in dim.lower()
                        and dim     != "number"):
                    da = da.squeeze(dim, drop=True)


            da = da.transpose("number", *[d for d in da.dims if d != "number"])
            da = da.isel(number=slice(0, N_MEMBERS))


            lat_name, lon_name = get_lat_lon_names(da)
            time_name          = get_time_name(da)


            lat_flip         = da[lat_name].values[0] < da[lat_name].values[-1]
            lon0             = da[lon_name].values
            lon_need_convert = np.nanmax(lon0) > 180


            if lat_flip:
                lat_v = da[lat_name].values[::-1]
            else:
                lat_v = da[lat_name].values.copy()


            if lon_need_convert:
                lon_new  = np.where(lon0 > 180, lon0 - 360, lon0)
                sort_idx = np.argsort(lon_new)
                lon_v    = lon_new[sort_idx]
            else:
                sort_idx = None
                lon_v    = lon0.copy()


            times       = pd.to_datetime(da[time_name].values).normalize()
            member_vals = da["number"].values
            mlabels     = []
            for i in range(len(member_vals)):
                try:    mlabels.append(int(member_vals[i]))
                except: mlabels.append(i + 1)


            all_data[year] = {
                "fp":            fp,
                "var":           var,
                "lat_flip":      lat_flip,
                "lon_sort_idx":  sort_idx,
                "monthday":      np.array([month_day_key(t) for t in times]),
                "member_to_idx": {m: i for i, m in enumerate(mlabels)},
                "time_to_idx":   {t: j for j, t in enumerate(times)},
                "times":         times,
                "n_times":       len(times),
            }


            if lat_vals is None:
                lat_vals = lat_v
                lon_vals = lon_v


            ds.close()
            del da, times, member_vals, mlabels, lat_v, lon_v
            gc.collect()
            print(f"  Scanned T2m {year}")


        except Exception as e:
            print(f"  Failed T2m {year}: {e}")
            gc.collect()


    return all_data, lat_vals, lon_vals



def _load_year_t2m(info):
    ds  = xr.open_dataset(info["fp"])
    var = info["var"]
    da  = ds[var]


    for dim in list(da.dims):
        if (da.sizes[dim] == 1
                and "lat"   not in dim.lower()
                and "lon"   not in dim.lower()
                and "time"  not in dim.lower()
                and "valid" not in dim.lower()
                and dim     != "number"):
            da = da.squeeze(dim, drop=True)


    da = da.transpose("number", *[d for d in da.dims if d != "number"])
    da = da.isel(number=slice(0, N_MEMBERS))


    lat_name, lon_name = get_lat_lon_names(da)


    if info["lat_flip"]:
        da = da.isel({lat_name: slice(None, None, -1)})


    if info["lon_sort_idx"] is not None:
        lon0    = da[lon_name].values
        lon_new = np.where(lon0 > 180, lon0 - 360, lon0)
        da = da.isel({lon_name: info["lon_sort_idx"]}).assign_coords(
            {lon_name: lon_new[info["lon_sort_idx"]]}
        )


    arr = da.values.astype(np.float32)
    ds.close()
    del da
    gc.collect()
    return arr  # (n_mem, T, lat, lon)



# ================================================================
# BASELINE
# ================================================================
# def build_baseline(all_data):
#     print(f"Building T2m baseline {BASELINE_START}–{BASELINE_END} ...")
#     baseline_years = [y for y in range(BASELINE_START, BASELINE_END + 1) if y in all_data]
#     mmdd_fields = {}


#     for y in baseline_years:
#         t2m      = _load_year_t2m(all_data[y])
#         ens_mean = t2m.mean(axis=0)
#         del t2m
#         gc.collect()
#         for i, key in enumerate(all_data[y]["monthday"]):
#             mmdd_fields.setdefault(key, []).append(ens_mean[i])
#         del ens_mean
#         gc.collect()


#     baseline_clim = {
#         key: np.stack(fields).mean(axis=0).astype(np.float32)
#         for key, fields in mmdd_fields.items()
#     }
#     del mmdd_fields
#     gc.collect()
#     return baseline_clim

def build_baseline(all_data):
    print(f"Building T2m rolling climatology {BASELINE_START}–{BASELINE_END} (±5 days)...")

    doy_to_fields = {}

    # ----------------------------
    # Step 1: collect baseline
    # ----------------------------
    for year in range(BASELINE_START, BASELINE_END + 1):

        if year not in all_data:
            continue

        t2m = _load_year_t2m(all_data[year])
        ens_mean = t2m.mean(axis=0)
        times = all_data[year]["times"]

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(
                ens_mean[i].astype(np.float32)
            )

        del t2m, ens_mean
        gc.collect()

    # ----------------------------
    # Step 2: raw climatology
    # ----------------------------
    sample = next(iter(doy_to_fields.values()))[0]
    nlat, nlon = sample.shape

    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        clim_raw[doy - 1] = np.nanmean(
            np.stack(fields, axis=0), axis=0
        )

    del doy_to_fields
    gc.collect()

    # ----------------------------
    # Step 3: rolling smoothing ✅
    # ----------------------------
    window = 11
    pad = window // 2

    clim_pad = np.concatenate(
        [clim_raw[-pad:], clim_raw, clim_raw[:pad]],
        axis=0
    )

    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]

        if not np.isfinite(win).any():
            continue

        clim_smooth[i] = np.nanmean(win, axis=0)

    del clim_pad, clim_raw
    gc.collect()

    # ----------------------------
    # Step 4: map to month-day ✅
    # ----------------------------
    baseline_clim = {}

    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        baseline_clim[key] = clim_smooth[t.dayofyear - 1]

    del clim_smooth
    gc.collect()

    print(f"  Done: {len(baseline_clim)} calendar days (smoothed)")
    return baseline_clim

# ================================================================
# TREND SLOPES — T2m
# ================================================================
def compute_trend_slopes(all_data, baseline_clim):
    print("Computing T2m trend slopes ...")
    years    = sorted(all_data.keys())
    md_index = {}


    for year in years:
        t2m = _load_year_t2m(all_data[year])
        md  = all_data[year]["monthday"]
        n_mem, n_t, n_lat, n_lon = t2m.shape


        for t_idx, key in enumerate(md):
            if key not in baseline_clim:
                continue
            c_field = baseline_clim[key]
            acc = np.zeros((n_lat, n_lon), dtype=np.float64)
            for mi in range(n_mem):
                acc += t2m[mi, t_idx].astype(np.float64) - c_field.astype(np.float64)
            ens_mean_anom = (acc / n_mem).astype(np.float32)
            del acc
            md_index.setdefault(key, []).append((year, t_idx, ens_mean_anom))
            del ens_mean_anom


        del t2m
        gc.collect()
        print(f"  Trend accumulate {year}")


    trend_slopes = {}
    for key in list(md_index.keys()):
        entries = md_index.pop(key)
        if len(entries) < 2:
            del entries
            continue
        yr_arr    = np.array([e[0] for e in entries], dtype=np.float64)
        yr_c      = yr_arr - yr_arr.mean()
        ens_stack = np.stack([e[2] for e in entries]).astype(np.float64)
        del entries
        denom = (yr_c ** 2).sum()
        if denom == 0:
            del yr_arr, yr_c, ens_stack
            continue
        b = (yr_c[:, None, None] * ens_stack).sum(axis=0) / denom
        a = ens_stack.mean(axis=0)
        trend_slopes[key] = (a.astype(np.float32), b.astype(np.float32), float(yr_arr.mean()))
        del yr_arr, yr_c, ens_stack, b, a
        gc.collect()


    del md_index
    gc.collect()
    print(f"  Done: {len(trend_slopes)} calendar days")
    return trend_slopes



# ================================================================
# TREND SLOPES — U100 60°N  ← 新增
# ================================================================
def compute_u_trend_slopes(u_data_all, baseline_u):
    print(f"Computing {U_LABEL} trend slopes at 60N ...")
    md_index = {}

    for year in sorted(u_data_all.keys()):
        info     = u_data_all[year]
        ens_mean = info["u"].mean(axis=0)  # (T, lat, lon)
        for t_idx, key in enumerate(info["monthday"]):
            if key not in baseline_u:
                continue
            u_anom_day = ens_mean[t_idx] - baseline_u[key]
            u_60n      = u_zonal_mean_60n(u_anom_day, info["lat_vals"])
            if np.isfinite(u_60n):
                md_index.setdefault(key, []).append((year, u_60n))
        del ens_mean
        gc.collect()

    u_trend_slopes = {}
    for key, entries in md_index.items():
        if len(entries) < 2:
            continue
        yr_arr  = np.array([e[0] for e in entries], dtype=np.float64)
        val_arr = np.array([e[1] for e in entries], dtype=np.float64)
        yr_c    = yr_arr - yr_arr.mean()
        denom   = (yr_c ** 2).sum()
        if denom == 0:
            continue
        b = (yr_c * val_arr).sum() / denom
        a = val_arr.mean()
        u_trend_slopes[key] = (float(a), float(b), float(yr_arr.mean()))

    del md_index
    gc.collect()
    print(f"  Done: {len(u_trend_slopes)} calendar days for {U_LABEL} trend")
    return u_trend_slopes



# ================================================================
# EXTRACT T2M PER EVENT
# ================================================================
def extract_intensity_per_event(all_data, lat_vals, lon_vals, baseline_clim, trend_slopes):
    print(f"Extracting T2m over day +{T_DAY_START} to +{T_DAY_END} ...")


    ssw_df = pd.read_csv(SSW_CSV_PATH)
    ssw_df["ssw_date"] = pd.to_datetime(ssw_df["ssw_date"], errors="coerce").dt.normalize()
    ssw_df = ssw_df.dropna(subset=["ssw_date"])
    ssw_df["member"] = ssw_df["member"].astype(int)
    ssw_df = ssw_df[ssw_df["ssw_date"].dt.month.isin([11, 12, 1, 2, 3])].copy()


    records = {rn: [] for rn in REGION_ORDER}


    for year in sorted(all_data.keys()):
        dfp = ssw_df[ssw_df["init_year"] == year]
        if len(dfp) == 0:
            continue


        info = all_data[year]
        t2m  = _load_year_t2m(info)
        md   = info["monthday"]


        for _, row in dfp.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"])
            if member not in info["member_to_idx"]:
                continue
            if center not in info["time_to_idx"]:
                continue


            m_idx    = info["member_to_idx"][member]
            dates    = window_dates(center, T_DAY_START, T_DAY_END)
            idx_list = [info["time_to_idx"].get(pd.Timestamp(dt)) for dt in dates]
            if any(v is None for v in idx_list):
                continue


            raw_win = np.empty((len(idx_list), t2m.shape[2], t2m.shape[3]),
                               dtype=np.float32)
            for i, tidx in enumerate(idx_list):
                key     = md[tidx]
                c_field = baseline_clim.get(key)
                raw_win[i] = (t2m[m_idx, tidx] - c_field) if c_field is not None else np.nan


            md_slice = [month_day_key(dt) for dt in dates]
            det_win  = raw_win.copy()
            for ti, key in enumerate(md_slice):
                if key in trend_slopes:
                    a, b, yr_mean = trend_slopes[key]
                    det_win[ti] -= (a + b * (year - yr_mean)).astype(np.float32)


            for rn, rb in REGION_BOXES.items():
                raw_daily = np.array([
                    cosine_weighted_mean(raw_win[ti], lat_vals, lon_vals, rb)
                    for ti in range(len(idx_list))
                ])
                det_daily = np.array([
                    cosine_weighted_mean(det_win[ti], lat_vals, lon_vals, rb)
                    for ti in range(len(idx_list))
                ])
                if not np.isfinite(raw_daily).any() or not np.isfinite(det_daily).any():
                    continue
                records[rn].append({
                    "year":     year,
                    "member":   member,
                    "ssw_date": center.strftime("%Y-%m-%d"),
                    "raw_mean": float(np.nanmean(raw_daily)),
                    "det_mean": float(np.nanmean(det_daily)),
                    "raw_p10":  float(np.nanpercentile(raw_daily, 10)),
                    "det_p10":  float(np.nanpercentile(det_daily, 10)),
                })
                del raw_daily, det_daily
            del raw_win, det_win


        del t2m
        gc.collect()
        print(f"  T2m year {year} done")


    return {rn: pd.DataFrame(records[rn]) for rn in REGION_ORDER}



# ================================================================
# U LOADING
# ================================================================
def load_u_year(year):
    fp = U_DIR / U_FILE_TMPL.format(year=year)
    if not fp.exists():
        print(f"  {U_LABEL} file not found: {fp}")
        return None


    try:
        ds  = xr.open_dataset(fp)
        var = list(ds.data_vars)[0]
        da  = ds[var]


        if ("forecast_reference_time" in da.dims and
                da.sizes["forecast_reference_time"] == 1):
            da = da.squeeze("forecast_reference_time", drop=True)


        da = da.transpose("number", *[d for d in da.dims if d != "number"])
        da = da.isel(number=slice(0, N_MEMBERS))


        lat_name  = [d for d in da.dims if "lat" in d.lower()][0]
        time_name = get_time_name(da)


        if da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})


        lat_vals_full = da[lat_name].values


        lat_mask    = (lat_vals_full >= 55.0) & (lat_vals_full <= 65.0)
        lat_indices = np.where(lat_mask)[0]
        if len(lat_indices) == 0:
            lat_indices = [int(np.argmin(np.abs(lat_vals_full - U_LAT_TARGET)))]
        da = da.isel({lat_name: lat_indices})


        da = da.load()
        times       = pd.to_datetime(da[time_name].values).normalize()
        member_vals = da["number"].values
        mlabels     = []
        for i in range(len(member_vals)):
            try:    mlabels.append(int(member_vals[i]))
            except: mlabels.append(i + 1)


        result = {
            "u":             da.values.astype(np.float32),
            "lat_vals":      da[lat_name].values.copy(),
            "times":         times,
            "monthday":      np.array([month_day_key(t) for t in times]),
            "member_to_idx": {m: i for i, m in enumerate(mlabels)},
            "time_to_idx":   {t: j for j, t in enumerate(times)},
        }
        ds.close()
        del da, times, member_vals, mlabels
        gc.collect()
        return result


    except Exception as e:
        print(f"  Failed {U_LABEL} {year}: {e}")
        gc.collect()
        return None



# def build_u_baseline(u_data_all):
#     print(f"Building {U_LABEL} baseline {BASELINE_START}–{BASELINE_END} ...")
#     baseline_years = [y for y in range(BASELINE_START, BASELINE_END + 1) if y in u_data_all]
#     mmdd_fields = {}
#     for y in baseline_years:
#         ens_mean = u_data_all[y]["u"].mean(axis=0)
#         for i, key in enumerate(u_data_all[y]["monthday"]):
#             mmdd_fields.setdefault(key, []).append(ens_mean[i])
#         del ens_mean
#         gc.collect()
#     baseline_u = {
#         key: np.stack(fields).mean(axis=0).astype(np.float32)
#         for key, fields in mmdd_fields.items()
#     }
#     del mmdd_fields
#     gc.collect()
#     return baseline_u
def build_u_baseline(u_data_all):
    print(f"Building {U_LABEL} rolling climatology {BASELINE_START}–{BASELINE_END} (±5 days)...")

    doy_to_fields = {}

    # ----------------------------
    # Step 1: collect baseline
    # ----------------------------
    for year in range(BASELINE_START, BASELINE_END + 1):

        if year not in u_data_all:
            continue

        ens_mean = u_data_all[year]["u"].mean(axis=0)  # (time, lat, lon)
        times    = u_data_all[year]["times"]

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(
                ens_mean[i].astype(np.float32)
            )

        del ens_mean
        gc.collect()

    # ----------------------------
    # Step 2: raw climatology
    # ----------------------------
    sample = next(iter(doy_to_fields.values()))[0]
    nlat, nlon = sample.shape

    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        clim_raw[doy - 1] = np.nanmean(
            np.stack(fields, axis=0), axis=0
        )

    del doy_to_fields
    gc.collect()

    # ----------------------------
    # Step 3: rolling smoothing ✅
    # ----------------------------
    window = 11
    pad = window // 2

    clim_pad = np.concatenate(
        [clim_raw[-pad:], clim_raw, clim_raw[:pad]],
        axis=0
    )

    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]

        if not np.isfinite(win).any():
            continue

        clim_smooth[i] = np.nanmean(win, axis=0)

    del clim_pad, clim_raw
    gc.collect()

    # ----------------------------
    # Step 4: map to month-day ✅
    # ----------------------------
    baseline_u = {}

    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        baseline_u[key] = clim_smooth[t.dayofyear - 1]

    del clim_smooth
    gc.collect()

    print(f"  Done: {len(baseline_u)} calendar days (smoothed)")
    return baseline_u


# ================================================================
# EXTRACT U STRENGTH — 同时输出 raw 和 detrended  ← 修改
# ================================================================
def extract_u_strength(u_data_all, baseline_u, u_trend_slopes):
    print(f"Extracting {U_LABEL} anomaly averaged over day {U_DAY_START} to +{U_DAY_END} ...")


    ssw_df = pd.read_csv(SSW_CSV_PATH)
    ssw_df["ssw_date"] = pd.to_datetime(ssw_df["ssw_date"], errors="coerce").dt.normalize()
    ssw_df = ssw_df.dropna(subset=["ssw_date"])
    ssw_df["member"] = ssw_df["member"].astype(int)
    ssw_df = ssw_df[ssw_df["ssw_date"].dt.month.isin([11, 12, 1, 2, 3])].copy()


    rows = []
    for year, info in u_data_all.items():
        dfp = ssw_df[ssw_df["init_year"] == year]
        if len(dfp) == 0:
            continue
        for _, row in dfp.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"])
            if member not in info["member_to_idx"]:
                continue
            dates    = window_dates(center, U_DAY_START, U_DAY_END)
            idx_list = [info["time_to_idx"].get(pd.Timestamp(dt)) for dt in dates]
            if any(v is None for v in idx_list):
                continue
            m_idx       = info["member_to_idx"][member]
            u_list_raw  = []
            u_list_det  = []
            for c_idx in idx_list:
                key    = info["monthday"][c_idx]
                if key not in baseline_u:
                    continue
                u_anom    = info["u"][m_idx, c_idx] - baseline_u[key]
                u_60n_raw = u_zonal_mean_60n(u_anom, info["lat_vals"])
                del u_anom
                if not np.isfinite(u_60n_raw):
                    continue
                u_list_raw.append(u_60n_raw)
                # 去趋势
                u_60n_det = u_60n_raw
                if key in u_trend_slopes:
                    a, b, yr_mean = u_trend_slopes[key]
                    u_60n_det = u_60n_raw - (a + b * (year - yr_mean))
                u_list_det.append(u_60n_det)

            if len(u_list_raw) == 0:
                continue
            rows.append({
                "year":       year,
                "member":     member,
                "ssw_date":   center.strftime("%Y-%m-%d"),
                "u_anom":     float(np.mean(u_list_raw)),
                "u_anom_det": float(np.mean(u_list_det)),
            })
            del u_list_raw, u_list_det


    df_out = pd.DataFrame(rows)
    del rows
    gc.collect()
    print(f"  Extracted {len(df_out)} {U_LABEL} records.")
    return df_out



# ================================================================
# NPZ CACHE
# ================================================================
def df_to_recarray(df):
    df2 = df.copy()
    if "ssw_date" in df2.columns:
        df2["ssw_date"] = df2["ssw_date"].astype(str)
    return df2.to_records(index=False)



def recarray_to_df(arr):
    df = pd.DataFrame.from_records(arr)
    if "ssw_date" in df.columns:
        df["ssw_date"] = pd.to_datetime(df["ssw_date"])
    return df



def save_cache_npz(npz_path, result_dfs, u_strength_df):
    save_dict = {}
    for rn in REGION_ORDER:
        save_dict[f"result__{rn}"] = df_to_recarray(result_dfs[rn])
    save_dict["u_strength_df"] = df_to_recarray(u_strength_df)
    np.savez_compressed(npz_path, **save_dict)
    print(f"NPZ cache saved: {npz_path}")



def load_cache_npz(npz_path):
    print(f"Loading NPZ cache: {npz_path}")
    z             = np.load(npz_path, allow_pickle=True)
    result_dfs    = {rn: recarray_to_df(z[f"result__{rn}"]) for rn in REGION_ORDER}
    u_strength_df = recarray_to_df(z["u_strength_df"])
    z.close()
    return result_dfs, u_strength_df



# ================================================================
# CHOW TEST
# ================================================================
def chow_test(x1, y1, x2, y2):
    def ssr(x, y):
        slope, intercept, _, _, _ = linregress(x, y)
        resid = y - (intercept + slope * x)
        return np.sum(resid ** 2), len(x)
    ssr1, _         = ssr(x1, y1)
    ssr2, _         = ssr(x2, y2)
    ssr_pool, n_all = ssr(np.concatenate([x1, x2]), np.concatenate([y1, y2]))
    k = 2
    F = ((ssr_pool - (ssr1 + ssr2)) / k) / ((ssr1 + ssr2) / (n_all - 2 * k))
    p = 1 - f_dist.cdf(F, dfn=k, dfd=n_all - 2 * k)
    return F, p



# ================================================================
# DRAW ONE PANEL  ← 新增参数 use_det_u
# ================================================================
def draw_regression_panel(ax, merged_all, y_col, u_col="u_anom",
                           ylabel=None, panel_label=None, show_legend=False):
    merged_all = merged_all.copy()
    merged_all["year"] = merged_all["year"].astype(int)


    EARLY       = (1981, 1990, "1981–1990", COL_EARLY)
    LATE        = (2015, 2024, "2015–2024", COL_LATE)
    TWO_PERIODS = [EARLY, LATE]
    period_data = {}


    for p_start, p_end, p_label, pcolor in TWO_PERIODS:
        sub   = merged_all[(merged_all["year"] >= p_start) & (merged_all["year"] <= p_end)]
        x     = sub[u_col].values
        y     = sub[y_col].values
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        xv, yv = x[valid], y[valid]
        period_data[p_label] = (xv, yv, pcolor)


        slope, intercept, r_val, p_val, se = linregress(xv, yv)
        n_p    = len(xv)
        t_crit = t_dist.ppf(0.975, df=n_p - 2)
        ci_sl  = t_crit * se


        ax.scatter(xv, yv, s=14, alpha=0.55,
                   color=pcolor, edgecolors="none", zorder=2)


        cross_color = "blue" if p_label == "1981–1990" else "red"
        ax.plot(np.mean(xv), np.mean(yv),
                marker='+', markersize=20, markeredgewidth=3.0,
                color=cross_color, zorder=5)


        x_fit   = np.linspace(xv.min(), xv.max(), 200)
        x_mean  = xv.mean()
        ss_x    = np.sum((xv - x_mean) ** 2)
        resid   = yv - (intercept + slope * xv)
        s_res   = np.sqrt(np.sum(resid ** 2) / (n_p - 2))
        ci_band = t_crit * s_res * np.sqrt(1 / n_p + (x_fit - x_mean) ** 2 / ss_x)
        y_fit   = intercept + slope * x_fit
        ls      = "-" if p_val < 0.05 else "--"


        ax.fill_between(x_fit, y_fit - ci_band, y_fit + ci_band,
                        color=pcolor, alpha=0.10, zorder=1)
        ax.plot(x_fit, y_fit, color=pcolor, lw=1.7, ls=ls, zorder=3)


        txt_y = 0.05 if p_label == EARLY[2] else 0.27
        ax.text(
            0.97, txt_y,
            f"{p_label}\n"
            f"$\\beta$={slope:.2f}$\\pm${ci_sl:.2f}\n"
            f"$R^2$={r_val**2:.2f}, {format_pval(p_val)}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom", color=pcolor,
            bbox=dict(boxstyle="square,pad=0.20", facecolor="white",
                      edgecolor=pcolor, linewidth=0.8, alpha=0.92),
            zorder=10
        )
        del x_fit, x_mean, ss_x, resid, s_res, ci_band, y_fit


    if len(period_data) == 2:
        labels = list(period_data.keys())
        x1, y1, _ = period_data[labels[0]]
        x2, y2, _ = period_data[labels[1]]
        _, p_chow = chow_test(x1, y1, x2, y2)
        ax.text(0.97, 0.97, f"Chow: {format_pval(p_chow)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=COL_TEXT)


    ax.axhline(0, color=COL_ZERO, lw=0.7, ls="--", alpha=0.8)
    ax.axvline(0, color=COL_ZERO, lw=0.7, ls="--", alpha=0.8)
    style_ax(ax, add_minor=False)


    ax.set_xlim(-20, 20) if U_LEVEL == 100 else ax.set_xlim(-40, 5)


    vals = merged_all[y_col].dropna().values
    if len(vals) > 0:
        vmin = vals.min(); vmax = vals.max()
        pad  = 0.10 * (vmax - vmin) if vmax > vmin else 1.0
        ax.set_ylim(vmin - pad, vmax + pad)


    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if panel_label is not None:
        add_panel_label(ax, panel_label)


    if show_legend:
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_EARLY,
                   markeredgecolor='none', markersize=5, label='1981–1990'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor=COL_LATE,
                   markeredgecolor='none', markersize=5, label='2015–2024'),
            Line2D([0], [0], color='black', lw=1.7, ls='-',  label='p<0.05'),
            Line2D([0], [0], color='black', lw=1.7, ls='--', label='n.s.'),
            Line2D([0], [0], marker='+', color='black', linestyle='none',
                   markersize=12, markeredgewidth=2, label='Mean'),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True,
                  fontsize=8, bbox_to_anchor=(0.04, 1))



# ================================================================
# PLOT: 2 rows × 3 cols
# ================================================================
def plot_regression_dual(result_dfs, u_strength_df,
                         raw_col, det_col, output_png):
    print(f"\nPlotting: {raw_col} / {det_col}")


    u_str = u_strength_df.copy()
    u_str["ssw_date"] = u_str["ssw_date"].astype(str).str[:10]


    fig, axes    = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    col_titles   = ["North America", "Europe", "East Asia"]
    # row_titles   = ["Raw", "Detrended"]


    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=10, fontweight="bold", pad=5)
    # for i in range(2):
    #     axes[i, 0].text(-0.25, 0.5, row_titles[i],
    #                     transform=axes[i, 0].transAxes,
    #                     ha='center', va='center',
    #                     fontsize=10, fontweight='bold', rotation=90)


    for ri, rn in enumerate(REGION_ORDER):
        df_t = result_dfs[rn].copy()
        df_t["ssw_date"] = df_t["ssw_date"].astype(str).str[:10]


        merged_all = pd.merge(
            df_t[["year", "member", "ssw_date", raw_col, det_col]],
            u_str[["year", "member", "ssw_date", "u_anom", "u_anom_det"]],
            on=["year", "member", "ssw_date"], how="inner"
        ).dropna(subset=[raw_col, det_col, "u_anom", "u_anom_det"])


        print(f"  {rn}: {len(merged_all)} matched events")


        if merged_all.empty:
            axes[0, ri].set_visible(False)
            axes[1, ri].set_visible(False)
            continue


        # Raw 行：T2m raw，U100 raw
        draw_regression_panel(
            axes[0, ri], merged_all=merged_all, y_col=raw_col,
            u_col="u_anom",
            ylabel=f"Raw T2m anomaly" if ri == 0 else None,
            panel_label=panel_labels[ri], show_legend=(ri == 0)
        )
        # Detrended 行：T2m detrended，U100 detrended 
        draw_regression_panel(
            axes[1, ri], merged_all=merged_all, y_col=det_col,
            u_col="u_anom_det",
            ylabel=f"Detrended T2m anomaly" if ri == 0 else None,
            panel_label=panel_labels[3 + ri], show_legend=False
        )
        axes[1, ri].set_xlabel(
            f"Raw 100 hPa U anomaly"
        )
        axes[0, ri].set_xlabel(
            f"Detrended 100 hPa U anomaly"
        )
        del df_t, merged_all


    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12,
                        top=0.94, wspace=0.25, hspace=0.1)
    plt.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"Saved: {output_png}")



# ================================================================
# BUILD OR LOAD CACHE
# ================================================================
def build_or_load_cache():
    if NPZ_CACHE.exists():
        return load_cache_npz(NPZ_CACHE)


    print("No NPZ cache found. Computing from raw data...")


    # --- T2m ---
    all_data, lat_vals, lon_vals = load_all_winters()
    baseline_clim                = build_baseline(all_data)
    trend_slopes                 = compute_trend_slopes(all_data, baseline_clim)
    result_dfs                   = extract_intensity_per_event(
        all_data, lat_vals, lon_vals, baseline_clim, trend_slopes
    )
    del all_data, baseline_clim, trend_slopes, lat_vals, lon_vals
    gc.collect()


    # --- U ---
    u_data_all = {}
    for year in range(START_YEAR, END_YEAR + 1):
        data = load_u_year(year)
        if data is not None:
            u_data_all[year] = data
            print(f"  Loaded {U_LABEL} {year}")


    baseline_u      = build_u_baseline(u_data_all)
    u_trend_slopes  = compute_u_trend_slopes(u_data_all, baseline_u)  # ← 新增
    u_strength_df   = extract_u_strength(u_data_all, baseline_u, u_trend_slopes)  # ← 传入趋势
    del u_data_all, baseline_u, u_trend_slopes
    gc.collect()


    save_cache_npz(NPZ_CACHE, result_dfs, u_strength_df)
    return result_dfs, u_strength_df



# ================================================================
# MAIN
# ================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Running with {U_LABEL} (U_LEVEL={U_LEVEL}) ===")


    result_dfs, u_strength_df = build_or_load_cache()


    plot_regression_dual(
        result_dfs, u_strength_df,
        raw_col="raw_mean",
        det_col="det_mean",
        output_png=str(OUTPUT_DIR / f"SI6_SSW_{U_LABEL}_regression_dual_mean_NDJFM_{BASELINE_END}.pdf")
    )


    print(f"Done. ({U_LABEL})")
    gc.collect()



if __name__ == "__main__":
    main()