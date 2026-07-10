# -*- coding: utf-8 -*-
"""
Combined figure with cache (.npz)

Layout (3 rows, 2 columns, last row spans full width):
Row 1:
    (a) Monthly SSW frequency: SEAS5 vs ERA5
    (b) Post-SSW U10 amplitude distribution: SEAS5 vs ERA5
Row 2:
    (c) Post-SSW T2m amplitude over 3 regions - left: Raw
    (d) Post-SSW T2m amplitude over 3 regions - right: Detrended
Row 3:
    (e) SEAS5 vs ERA5 SSW Frequency in 10-year Sliding Windows (NDJFM) - full width
"""

import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================================================================
# GLOBAL STYLE
# ================================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# ================================================================
# SHARED SETTINGS
# ================================================================
START_YEAR = 1981
END_YEAR   = 2024
N_MEMBERS  = 25
MONTHS_DJF = [11, 12, 1, 2, 3]
BASELINE_START = 1981
BASELINE_END   = 2010

SEAS5_SSW_CSV = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
ERA5_SSW_CSV  = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure\ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv")

OUTPUT_DIR = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_OUT = OUTPUT_DIR / f"SI1_SEAS5_ERA5_frequency_U_t2m_sliding_{BASELINE_END}.pdf"
NPZ_U10 = OUTPUT_DIR / f"SI1_cache_u10_{BASELINE_END}.npz"
NPZ_T2M = OUTPUT_DIR / f"SI1_cache_t2m_{BASELINE_END}.npz"

COL_RED        = "#ED4043"
COL_BLUE       = "#1C6AB1"
COL_SEAS5      = "#1C6AB1"
COL_SEAS5_FILL = "#9EC3E6"
COL_ERA5       = "#E35235"
COL_ERA5_EDGE  = "#ED4043"
COL_GRID       = "#D9D9D9"
COL_SPINE      = "#000000"
COL_TEXT       = "#000000"

AXIS_TICK_FONTSIZE   = 12
AXIS_TICK_FONTWEIGHT = "bold"

ERA5_EVENT_ALPHA  = 0.55
ERA5_EVENT_SIZE   = 46
ERA5_DIAMOND_SIZE = 72
ERA5_DIAMOND_FACE = COL_ERA5
ERA5_DIAMOND_EDGE = COL_ERA5_EDGE
ERA5_DIAMOND_LW   = 1.0

# ================================================================
# CODE 1 SETTINGS
# ================================================================
MONTHS_PLOT  = [11, 12, 1, 2, 3]
MONTH_LABELS = ["Nov", "Dec", "Jan", "Feb", "Mar"]

# ================================================================
# CODE 2 SETTINGS — paths updated to daily file directory
# ================================================================
ERA5_U10_PATH       = Path(r"F:\data\ERA5_data\ERA5_u_daily_1940_2025_10_no229.nc")
SEAS5_U10_DAILY_DIR = Path(r"F:\data\IFS_U10_daily")
U10_FILE_PATTERN    = "SEAS5_u10hPa_NH_{year}11_system51_m25_daily.nc"
U_VAR_CANDIDATES    = ["u", "u10", "uwnd", "var131"]

DAY_START_U10      = 15
DAY_END_U10        = 59
MIN_VALID_FRAC_U10 = 0.8
N_DAYS_U10         = DAY_END_U10 - DAY_START_U10 + 1
MIN_DAYS_U10       = int(N_DAYS_U10 * MIN_VALID_FRAC_U10)

# ================================================================
# CODE 3 SETTINGS — paths updated to daily file directory
# ================================================================
ERA5_T2M_PATH       = Path(r"F:\data\ERA5_data\ERA5_t2m_daily_1940_2024_no229.nc")
SEAS5_T2M_DAILY_DIR = Path(r"F:\data\IFS_t2m_daily")
T2M_FILE_PATTERN    = "SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"
T2M_VAR_CANDIDATES  = ["t2m","2m_temperature"]

DAY_START_T      = 15
DAY_END_T        = 59
MIN_VALID_FRAC_T = 0.8
N_DAYS_T         = DAY_END_T - DAY_START_T + 1
MIN_DAYS_T       = int(N_DAYS_T * MIN_VALID_FRAC_T)

REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": -140, "lon_max": -60},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min":   0,  "lon_max":  40},
    "EastAsia":      {"lat_min": 45, "lat_max": 70, "lon_min":  60,  "lon_max": 120},
}

REGION_NAMES = {
    "NorthAmerica": "North America",
    "Europe":       "Europe",
    "EastAsia":      "East Asia",
}
REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]

# ================================================================
# CODE 4 (SLIDING WINDOW) SETTINGS
# ================================================================
SLIDING_WINDOW = 10
N_BOOT = 1000
BOOT_SEED = 100
MONTHS_NDJFM = [11, 12, 1, 2, 3]


# ================================================================
# GENERAL HELPERS
# ================================================================
def style_ax(ax, add_minor_y=False):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COL_SPINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=COL_TEXT, direction="out")
    ax.grid(axis='y', ls='--', lw=0.7, color=COL_GRID, alpha=0.7)
    ax.set_axisbelow(True)
    if add_minor_y:
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def add_panel_label(ax, label):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=13, fontweight='bold', color=COL_TEXT,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.2),
        zorder=20
    )


def set_bold_axis_ticklabels(ax, fontsize=None):
    if fontsize is None:
        fontsize = AXIS_TICK_FONTSIZE
    for lab in ax.get_xticklabels():
        lab.set_fontsize(fontsize)
        lab.set_fontweight(AXIS_TICK_FONTWEIGHT)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(fontsize)
        lab.set_fontweight(AXIS_TICK_FONTWEIGHT)


def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def window_dates(ssw_date, day_start, day_end):
    return pd.date_range(
        ssw_date + pd.Timedelta(days=day_start),
        ssw_date + pd.Timedelta(days=day_end)
    )


def cosine_region_mean(field2d, lat_vals, lon_vals, rb):
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    lon_mask = (lon_vals >= rb["lon_min"]) & (lon_vals <= rb["lon_max"])
    sub      = field2d[np.ix_(lat_mask, lon_mask)]
    weights  = np.cos(np.deg2rad(lat_vals[lat_mask]))
    w2d      = weights[:, None] * np.ones(lon_mask.sum())
    valid    = np.isfinite(sub)
    if valid.sum() == 0:
        return np.nan
    return float(np.nansum(sub * w2d * valid) / np.nansum(w2d * valid))


def fit_daily_trend(md_series):
    """Fit a linear trend (slope b, intercept a, mean year) for each calendar day key."""
    coeffs = {}
    for key, entries in md_series.items():
        if len(entries) < 2:
            continue
        yrs  = np.array([e[0] for e in entries], dtype=np.float64)
        vals = np.array([e[1] for e in entries], dtype=np.float64)
        yc   = yrs - yrs.mean()
        denom = (yc ** 2).sum()
        if denom < 1e-10:
            continue
        b = float((yc * vals).sum() / denom)
        coeffs[key] = (float(vals.mean()), b, float(yrs.mean()))
        del yrs, vals, yc
    return coeffs


def read_era5_ssw_events():
    df = pd.read_csv(ERA5_SSW_CSV)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["month"] = df["ssw_date"].dt.month
    df = df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)
    return df


def read_seas5_ssw_events():
    df = pd.read_csv(SEAS5_SSW_CSV)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["member"] = df["member"].astype(int)
    df["month"]  = df["ssw_date"].dt.month
    df = df[df["member"] < N_MEMBERS].copy()
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
    return df


# ================================================================
# DAILY FILE LOADER (same approach as fig3 code)
# ================================================================
def get_var_name(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify variable, data_vars={list(ds.data_vars)}")


def safe_member_label(number_coord, idx):
    try:    return int(number_coord[idx])
    except: return idx + 1


def load_one_year_daily(year, daily_dir, file_pattern, var_candidates,
                        flip_lat=True, wrap_lon=True):
    """
    Load a single-year pre-computed daily SEAS5 file.
    Returns a dict: {arr, times, monthday, member_labels,
                     member_to_idx, time_to_idx, lat_vals, lon_vals}
    arr shape: (member, time, lat, lon)
    """
    fp = daily_dir / file_pattern.format(year=year)
    if not fp.exists():
        print(f"  Missing: {fp.name}")
        return None
    try:
        ds  = xr.open_dataset(fp)
        var = get_var_name(ds, var_candidates)
        da  = ds[var].load()
        ds.close()
        del ds

        if "number" not in da.dims:
            raise ValueError(f"Missing 'number' dim: {da.dims}")
        # Ensure member dimension is first
        da = da.transpose("number", *[d for d in da.dims if d != "number"])
        da = da.isel(number=slice(0, N_MEMBERS))

        # Squeeze any size-1 non-spatial/temporal dimensions
        for dim in list(da.dims):
            if dim not in ("number",) and da.sizes[dim] == 1 and \
               "lat" not in dim.lower() and "lon" not in dim.lower() and \
               "time" not in dim.lower() and "valid" not in dim.lower():
                da = da.squeeze(dim, drop=True)

        # Detect lat / lon / time dimension names
        lat_c = [d for d in da.dims if "lat" in d.lower()]
        lon_c = [d for d in da.dims if "lon" in d.lower()]
        if not lat_c: raise ValueError(f"No lat dim: {da.dims}")
        if not lon_c: raise ValueError(f"No lon dim: {da.dims}")
        lat_name = lat_c[0]
        lon_name = lon_c[0]

        time_name = None
        for name in ["valid_time", "time"]:
            if name in da.dims:
                time_name = name
                break
        if time_name is None:
            raise ValueError(f"No time dim: {da.dims}")

        # Flip latitude to descending (N→S) if needed
        if flip_lat and da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})

        # Convert 0–360 longitude to -180–180 if needed
        if wrap_lon:
            lon0 = da[lon_name].values
            if np.nanmax(lon0) > 180:
                lon_new  = np.where(lon0 > 180, lon0 - 360, lon0)
                sort_idx = np.argsort(lon_new)
                da       = da.isel({lon_name: sort_idx})
                da       = da.assign_coords({lon_name: lon_new[sort_idx]})
                del lon_new, sort_idx

        times         = pd.to_datetime(da[time_name].values).normalize()
        member_labels = [safe_member_label(da["number"].values, i)
                         for i in range(da.sizes["number"])]
        arr      = da.values.astype(np.float32)
        lat_vals = da[lat_name].values.copy()
        lon_vals = da[lon_name].values.copy()
        del da
        gc.collect()

        return {
            "arr":           arr,
            "times":         times,
            "monthday":      np.array([month_day_key(t) for t in times]),
            "member_labels": member_labels,
            "member_to_idx": {m: i for i, m in enumerate(member_labels)},
            "time_to_idx":   {pd.Timestamp(t): j for j, t in enumerate(times)},
            "lat_vals":      lat_vals,
            "lon_vals":      lon_vals,
        }
    except Exception as e:
        print(f"  Failed year={year}: {e}")
        gc.collect()
        return None


# ================================================================
# ERA5 LOADERS (single-file approach retained for ERA5)
# ================================================================
def load_era5_u10_series():
    """Load ERA5 U10 at 60°N as a pandas Series indexed by date."""
    ds        = xr.open_dataset(ERA5_U10_PATH)
    era5_time = pd.to_datetime(ds['time'].values).normalize()
    ds        = ds.assign_coords(time=era5_time)
    u_var     = 'u' if 'u' in ds.data_vars else list(ds.data_vars)[0]
    series    = (
        ds[u_var]
        .sel(level=10.0, method='nearest')
        .sel(lat=60.0,   method='nearest')
        .mean(dim='lon')
        .to_series()
    )
    ds.close()
    del ds, era5_time
    gc.collect()
    return series


def load_era5_t2m_array():
    """
    Load the full ERA5 T2m daily file.
    Returns: (data_array, times, lat_vals, lon_vals)
    data_array shape: (time, lat, lon)
    """
    ds       = xr.open_dataset(ERA5_T2M_PATH)
    var      = next((v for v in T2M_VAR_CANDIDATES if v in ds.data_vars), list(ds.data_vars)[0])
    da       = ds[var]
    if 'forecast_reference_time' in da.dims and da.sizes['forecast_reference_time'] == 1:
        da = da.squeeze('forecast_reference_time', drop=True)

    lat_name  = next(d for d in da.dims if 'lat' in d.lower())
    lon_name  = next(d for d in da.dims if 'lon' in d.lower())
    time_name = next(
        (d for d in da.dims if 'time' in d.lower() or 'valid' in d.lower()),
        next((c for c in da.coords if 'time' in c.lower() or 'valid' in c.lower()), None)
    )

    # Flip to descending latitude if needed
    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    # Convert 0–360 to -180–180 if needed
    lon_vals = da[lon_name].values
    if np.nanmax(lon_vals) > 180:
        lon_new  = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
        sort_idx = np.argsort(lon_new)
        da       = da.isel({lon_name: sort_idx})
        da       = da.assign_coords({lon_name: lon_new[sort_idx]})
        del lon_new, sort_idx, lon_vals

    lat_vals   = da[lat_name].values.copy()
    lon_vals   = da[lon_name].values.copy()
    times      = pd.to_datetime(da[time_name].values).normalize()
    data_array = da.load().values.astype(np.float32)
    ds.close()
    del ds, da
    gc.collect()
    return data_array, times, lat_vals, lon_vals


# ================================================================
# CACHE SAVE / LOAD
# ================================================================
def save_cache_u10(npz_path, u10_era5, u10_seas5):
    np.savez(npz_path, u10_era5=u10_era5, u10_seas5=u10_seas5)
    print(f"  U10 cache saved → {npz_path}")


def load_cache_u10(npz_path):
    z   = np.load(npz_path, allow_pickle=False)
    e_v = z["u10_era5"].copy()
    s_v = z["u10_seas5"].copy()
    z.close()
    return e_v, s_v


def save_cache_t2m(npz_path, t2m_e_raw, t2m_e_det, t2m_s_raw, t2m_s_det):
    np.savez(
        npz_path,
        t2m_e_raw_na=t2m_e_raw["NorthAmerica"], t2m_e_raw_eu=t2m_e_raw["Europe"],
        t2m_e_raw_si=t2m_e_raw["EastAsia"],
        t2m_e_det_na=t2m_e_det["NorthAmerica"], t2m_e_det_eu=t2m_e_det["Europe"],
        t2m_e_det_si=t2m_e_det["EastAsia"],
        t2m_s_raw_na=t2m_s_raw["NorthAmerica"], t2m_s_raw_eu=t2m_s_raw["Europe"],
        t2m_s_raw_si=t2m_s_raw["EastAsia"],
        t2m_s_det_na=t2m_s_det["NorthAmerica"], t2m_s_det_eu=t2m_s_det["Europe"],
        t2m_s_det_si=t2m_s_det["EastAsia"],
    )
    print(f"  T2m cache saved → {npz_path}")


def load_cache_t2m(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    t2m_e_raw = {"NorthAmerica": z["t2m_e_raw_na"].copy(),
                 "Europe":       z["t2m_e_raw_eu"].copy(),
                 "EastAsia":      z["t2m_e_raw_si"].copy()}
    t2m_e_det = {"NorthAmerica": z["t2m_e_det_na"].copy(),
                 "Europe":       z["t2m_e_det_eu"].copy(),
                 "EastAsia":      z["t2m_e_det_si"].copy()}
    t2m_s_raw = {"NorthAmerica": z["t2m_s_raw_na"].copy(),
                 "Europe":       z["t2m_s_raw_eu"].copy(),
                 "EastAsia":      z["t2m_s_raw_si"].copy()}
    t2m_s_det = {"NorthAmerica": z["t2m_s_det_na"].copy(),
                 "Europe":       z["t2m_s_det_eu"].copy(),
                 "EastAsia":      z["t2m_s_det_si"].copy()}
    z.close()
    return t2m_e_raw, t2m_e_det, t2m_s_raw, t2m_s_det


# ================================================================
# COMPUTE CACHE — U10
# ================================================================
def compute_and_save_u10():
    """
    Compute post-SSW mean U10 amplitude arrays for ERA5 and SEAS5.
    SEAS5: stream one year at a time using load_one_year_daily().
    ERA5:  single-file approach retained.
    Results are saved to NPZ_U10 and all large arrays freed on exit.
    """
    print("Computing U10 arrays ...")

    # --- ERA5 U10 (single file) ---
    u_era5   = load_era5_u10_series()
    era5_ssw = read_era5_ssw_events()
    era5_ssw = era5_ssw[era5_ssw["month"].isin(MONTHS_DJF)].copy()

    era5_amps = []
    for _, row in era5_ssw.iterrows():
        vals = u_era5.reindex(window_dates(row['ssw_date'], DAY_START_U10, DAY_END_U10)).dropna()
        if len(vals) >= MIN_DAYS_U10:
            era5_amps.append(float(vals.mean()))
        del vals
    e_v = np.array(era5_amps, dtype=np.float32)
    del era5_amps, u_era5, era5_ssw
    gc.collect()

    # --- SEAS5 U10: stream year by year ---
    df_ifs = read_seas5_ssw_events()
    df_ifs = df_ifs[df_ifs["month"].isin(MONTHS_DJF)].copy()

    s5_amps = []
    for year in range(START_YEAR, END_YEAR + 1):
        dfp = df_ifs[df_ifs["init_year"] == year]
        if dfp.empty:
            continue

        data = load_one_year_daily(year, SEAS5_U10_DAILY_DIR, U10_FILE_PATTERN, U_VAR_CANDIDATES)
        if data is None:
            continue

        arr      = data["arr"]          # (member, time, lat, lon)
        m_to_idx = data["member_to_idx"]
        t_to_idx = data["time_to_idx"]
        lat_vals = data["lat_vals"]
        lon_vals = data["lon_vals"]

        # Find the latitude index closest to 60°N
        lat_idx = int(np.argmin(np.abs(lat_vals - 60.0)))

        for _, row in dfp.iterrows():
            member = int(row["member"])
            if member not in m_to_idx:
                continue
            mi = m_to_idx[member]

            win_d    = window_dates(row['ssw_date'], DAY_START_U10, DAY_END_U10)
            day_vals = []
            for dt in win_d:
                tidx = t_to_idx.get(pd.Timestamp(dt))
                if tidx is None:
                    continue
                # Zonal mean at 60°N for this member and day
                u_val = float(np.nanmean(arr[mi, tidx, lat_idx, :]))
                if np.isfinite(u_val):
                    day_vals.append(u_val)

            if len(day_vals) >= MIN_DAYS_U10:
                s5_amps.append(float(np.mean(day_vals)))
            del day_vals

        del data, arr, m_to_idx, t_to_idx, lat_vals, lon_vals
        gc.collect()
        print(f"  U10 year {year} done.")

    s_v = np.array(s5_amps, dtype=np.float32)
    del s5_amps, df_ifs
    gc.collect()

    print(f"  ERA5 U10 events: {len(e_v)}, SEAS5 U10 events: {len(s_v)}")
    save_cache_u10(NPZ_U10, e_v, s_v)
    del e_v, s_v
    gc.collect()


# ================================================================
# COMPUTE CACHE — T2m
# ================================================================
def compute_and_save_t2m():
    print("Computing T2m arrays ...")

    # 用统一 rolling climatology
    baseline_clim_era5, lat_era5, lon_era5 = build_rolling_climatology()

    data_era5, times_era5, _, _ = load_era5_t2m_array()

    # ---- 建 ERA5 detrend
    era5_md = {rn: {} for rn in REGION_ORDER}
    for i, dt in enumerate(times_era5):
        key  = month_day_key(dt)
        year = dt.year

        if not (START_YEAR <= year <= END_YEAR):
            continue

        cf = baseline_clim_era5.get(key)
        if cf is None:
            continue

        for rn in REGION_ORDER:
            v = cosine_region_mean(
                data_era5[i] - cf,
                lat_era5, lon_era5,
                REGION_BOXES[rn]
            )
            if np.isfinite(v):
                era5_md[rn].setdefault(key, []).append((year, v))

    era5_coeffs = {rn: fit_daily_trend(era5_md[rn]) for rn in REGION_ORDER}
    del era5_md
    gc.collect()

    # ---- ERA5 events
    era5_ssw = read_era5_ssw_events()
    era5_ssw = era5_ssw[era5_ssw["month"].isin(MONTHS_DJF)].copy()

    time_to_idx_era5 = {dt: i for i, dt in enumerate(times_era5)}

    era5_raw = {rn: [] for rn in REGION_ORDER}
    era5_det = {rn: [] for rn in REGION_ORDER}

    for _, row in era5_ssw.iterrows():
        dates  = window_dates(row['ssw_date'], DAY_START_T, DAY_END_T)

        rn_raw = {rn: [] for rn in REGION_ORDER}
        rn_det = {rn: [] for rn in REGION_ORDER}

        for dt in dates:
            tidx = time_to_idx_era5.get(pd.Timestamp(dt))
            if tidx is None:
                continue

            key  = month_day_key(dt)
            year = dt.year

            cf = baseline_clim_era5.get(key)
            if cf is None:
                continue

            for rn in REGION_ORDER:
                v = cosine_region_mean(
                    data_era5[tidx] - cf,
                    lat_era5, lon_era5,
                    REGION_BOXES[rn]
                )

                if not np.isfinite(v):
                    continue

                rn_raw[rn].append(v)

                coeff = era5_coeffs[rn].get(key)
                if coeff is not None:
                    a, b, yr_mean = coeff
                    rn_det[rn].append(v - (a + b * (year - yr_mean)))
                else:
                    rn_det[rn].append(v)

        for rn in REGION_ORDER:
            if len(rn_raw[rn]) >= MIN_DAYS_T:
                era5_raw[rn].append(float(np.mean(rn_raw[rn])))
            if len(rn_det[rn]) >= MIN_DAYS_T:
                era5_det[rn].append(float(np.mean(rn_det[rn])))

        del rn_raw, rn_det

    e_raw = {rn: np.array(era5_raw[rn], dtype=np.float32) for rn in REGION_ORDER}
    e_det = {rn: np.array(era5_det[rn], dtype=np.float32) for rn in REGION_ORDER}

    del era5_raw, era5_det
    gc.collect()

    # ---- SEAS5（完全一致 climatology）
    baseline_clim_seas5 = {}
    print("\nBuilding SEAS5 rolling climatology...")

    doy_to_fields_s5 = {}

    for year in range(BASELINE_START, BASELINE_END + 1):

        data = load_one_year_daily(year, SEAS5_T2M_DAILY_DIR, T2M_FILE_PATTERN, T2M_VAR_CANDIDATES)
        if data is None:
            continue

        ens_mean = data["arr"].mean(axis=0)
        times    = data["times"]

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields_s5.setdefault(doy, []).append(ens_mean[i])

    # 转 climatology
    sample = next(iter(doy_to_fields_s5.values()))[0]
    nlat_s5, nlon_s5 = sample.shape

    clim_raw_s5 = np.full((366, nlat_s5, nlon_s5), np.nan)

    for doy, fields in doy_to_fields_s5.items():
        clim_raw_s5[doy-1] = np.nanmean(np.stack(fields, axis=0), axis=0)

    # rolling
    window = 11
    pad = window // 2

    clim_pad = np.concatenate([clim_raw_s5[-pad:], clim_raw_s5, clim_raw_s5[:pad]], axis=0)

    clim_smooth_s5 = np.full_like(clim_raw_s5, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth_s5[i] = np.nanmean(win, axis=0)

    # monthday
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        baseline_clim_seas5[month_day_key(t)] = clim_smooth_s5[t.dayofyear-1]

    print("  SEAS5 climatology done.")
    # ---- 建 SEAS5 detrend 系数（用 SEAS5 自己的趋势）
    print("  Building SEAS5 detrend coefficients...")
    seas5_md = {rn: {} for rn in REGION_ORDER}

    for year in range(START_YEAR, END_YEAR + 1):
        data = load_one_year_daily(year, SEAS5_T2M_DAILY_DIR, T2M_FILE_PATTERN, T2M_VAR_CANDIDATES)
        if data is None:
            continue

        arr      = data["arr"]
        t_to_idx = data["time_to_idx"]
        lat_vals = data["lat_vals"]
        lon_vals = data["lon_vals"]

        for dt, tidx in t_to_idx.items():
            key  = month_day_key(dt)
            yr   = dt.year
            if not (START_YEAR <= yr <= END_YEAR):
                continue
            cf = baseline_clim_seas5.get(key)
            if cf is None:
                continue
            # 集合平均后做区域平均
            ens_mean_field = arr[:, tidx].mean(axis=0)
            for rn in REGION_ORDER:
                v = cosine_region_mean(ens_mean_field - cf, lat_vals, lon_vals, REGION_BOXES[rn])
                if np.isfinite(v):
                    seas5_md[rn].setdefault(key, []).append((yr, v))

        del data, arr, t_to_idx, lat_vals, lon_vals
        gc.collect()

    seas5_coeffs = {rn: fit_daily_trend(seas5_md[rn]) for rn in REGION_ORDER}
    del seas5_md
    gc.collect()
    print("  SEAS5 detrend coefficients done.")

    df_ifs = read_seas5_ssw_events()
    df_ifs = df_ifs[df_ifs["month"].isin(MONTHS_DJF)].copy()

    s5_raw_all = {rn: [] for rn in REGION_ORDER}
    s5_det_all = {rn: [] for rn in REGION_ORDER}

    for year in range(START_YEAR, END_YEAR + 1):

        dfp = df_ifs[df_ifs["init_year"] == year]
        if dfp.empty:
            continue

        data = load_one_year_daily(
            year, SEAS5_T2M_DAILY_DIR, T2M_FILE_PATTERN, T2M_VAR_CANDIDATES
        )
        if data is None:
            continue

        arr      = data["arr"]
        m_to_idx = data["member_to_idx"]
        t_to_idx = data["time_to_idx"]
        lat_vals = data["lat_vals"]
        lon_vals = data["lon_vals"]

        for _, row in dfp.iterrows():

            member = int(row["member"])
            if member not in m_to_idx:
                continue

            mi    = m_to_idx[member]
            dates = window_dates(row['ssw_date'], DAY_START_T, DAY_END_T)

            rn_raw = {rn: [] for rn in REGION_ORDER}
            rn_det = {rn: [] for rn in REGION_ORDER}

            for dt in dates:

                tidx = t_to_idx.get(pd.Timestamp(dt))
                if tidx is None:
                    continue

                key = month_day_key(dt)
                cf  = baseline_clim_seas5.get(key)
                if cf is None:
                    continue

                for rn in REGION_ORDER:

                    cf = baseline_clim_seas5.get(key)
                    if cf is None:
                        continue

                    v = cosine_region_mean(
                        arr[mi, tidx] - cf,
                        lat_vals, lon_vals,
                        REGION_BOXES[rn]
                    )

                    if not np.isfinite(v):
                        continue

                    rn_raw[rn].append(v)

                    coeff = seas5_coeffs[rn].get(key)
                    if coeff is not None:
                        a, b, yr_mean = coeff
                        rn_det[rn].append(v - (a + b * (year - yr_mean)))
                    else:
                        rn_det[rn].append(v)

            for rn in REGION_ORDER:
                if len(rn_raw[rn]) >= MIN_DAYS_T:
                    s5_raw_all[rn].append(float(np.mean(rn_raw[rn])))
                if len(rn_det[rn]) >= MIN_DAYS_T:
                    s5_det_all[rn].append(float(np.mean(rn_det[rn])))

        print(f"  T2m year {year} done.")

    s_raw = {rn: np.array(s5_raw_all[rn], dtype=np.float32) for rn in REGION_ORDER}
    s_det = {rn: np.array(s5_det_all[rn], dtype=np.float32) for rn in REGION_ORDER}
    del seas5_coeffs
    gc.collect()
    save_cache_t2m(NPZ_T2M, e_raw, e_det, s_raw, s_det)

    print("T2m cache complete ✅")


# ================================================================
# ENSURE CACHES EXIST (sequential: U10 first, then T2m)
# ================================================================
def ensure_caches():
    """
    Compute and save each cache file independently.
    U10 is fully computed and freed before T2m computation begins,
    minimising peak memory usage.
    """
    if not NPZ_U10.exists():
        compute_and_save_u10()
        gc.collect()
    else:
        print(f"U10 cache found: {NPZ_U10}")

    if not NPZ_T2M.exists():
        compute_and_save_t2m()
        gc.collect()
    else:
        print(f"T2m cache found: {NPZ_T2M}")


# ================================================================
# CODE 1 — MONTHLY SSW FREQUENCY
# ================================================================
def compute_counts(df_sub, years_list):
    counts = {}
    for m in MONTHS_PLOT:
        counts[m] = len(df_sub[
            (df_sub["month"] == m) & (df_sub["init_year"].isin(years_list))
        ])
    return counts


def compute_member_spread(df_sub, years_list):
    """Return mean, 10th/90th pctile, min, max across members."""
    n_years  = len(years_list)
    members  = sorted(df_sub["member"].unique())
    freq_mat = np.full((len(members), len(MONTHS_PLOT)), np.nan)
    for mi, mem in enumerate(members):
        dfm = df_sub[(df_sub["member"] == mem) & (df_sub["init_year"].isin(years_list))]
        for ki, m in enumerate(MONTHS_PLOT):
            freq_mat[mi, ki] = len(dfm[dfm["month"] == m]) / n_years * 10
        del dfm
    mean_vals = np.nanmean(freq_mat, axis=0)
    p10       = np.nanpercentile(freq_mat, 10, axis=0)
    p90       = np.nanpercentile(freq_mat, 90, axis=0)
    vmin      = np.nanmin(freq_mat, axis=0)
    vmax      = np.nanmax(freq_mat, axis=0)
    del freq_mat
    gc.collect()
    return mean_vals, p10, p90, vmin, vmax


def compute_era5_freq(era5_df, years_list):
    n_years = len(years_list)
    freqs   = {}
    for m in MONTHS_PLOT:
        n = len(era5_df[(era5_df["month"] == m) & (era5_df["init_year"].isin(years_list))])
        freqs[m] = n / n_years * 10
    return freqs

def build_rolling_climatology():
    print("\nBuilding ERA5 rolling climatology (1981–2010, ±5 days)...")

    data, times, lat_vals, lon_vals = load_era5_t2m_array()

    doy_to_fields = {}

    # ---- Step 1: collect 1981–2010 daily fields
    for i, t in enumerate(times):
        if not (BASELINE_START <= t.year <= BASELINE_END):
            continue
        doy = t.dayofyear
        doy_to_fields.setdefault(doy, []).append(data[i])

    # ---- Step 2: raw climatology
    nlat, nlon = len(lat_vals), len(lon_vals)
    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        clim_raw[doy - 1] = np.nanmean(np.stack(fields, axis=0), axis=0)

    del doy_to_fields
    gc.collect()

    # ---- Step 3: rolling ±5 days
    window = 11
    pad = window // 2

    clim_pad = np.concatenate([
        clim_raw[-pad:], clim_raw, clim_raw[:pad]
    ], axis=0)

    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]
        if np.isfinite(win).any():
            clim_smooth[i] = np.nanmean(win, axis=0)

    del clim_pad, clim_raw
    gc.collect()

    # ---- Step 4: map to monthday
    baseline_clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        baseline_clim[key] = clim_smooth[t.dayofyear - 1]

    del clim_smooth, data, times
    gc.collect()

    print(f"  Done: {len(baseline_clim)} keys (smoothed)")
    return baseline_clim, lat_vals, lon_vals

# -------------------------------------------------------------------------------
def draw_SSW_frequency(ax):
    df_ifs  = read_seas5_ssw_events()
    df_ifs  = df_ifs[df_ifs["month"].isin(MONTHS_PLOT)].copy()
    df_era5 = read_era5_ssw_events()
    df_era5 = df_era5[df_era5["month"].isin(MONTHS_PLOT)].copy()

    all_years  = list(range(START_YEAR, END_YEAR + 1))
    means, p10, p90, vmin, vmax = compute_member_spread(df_ifs, all_years)
    era5_freqs = compute_era5_freq(df_era5, all_years)
    era5_vals  = np.array([era5_freqs[m] for m in MONTHS_PLOT])
    ifs_counts  = compute_counts(df_ifs, all_years)
    era5_counts = compute_counts(df_era5, all_years)

    del df_ifs, df_era5, era5_freqs
    gc.collect()

    x     = np.arange(len(MONTHS_PLOT))
    box_w = 0.34

    for i in range(len(MONTHS_PLOT)):
        # Draw 10–90th percentile box
        rect = Rectangle(
            (x[i] - box_w / 2, p10[i]),
            box_w, p90[i] - p10[i],
            facecolor=COL_SEAS5_FILL, edgecolor=COL_SEAS5,
            linewidth=1.1, alpha=0.45, zorder=3
        )
        ax.add_patch(rect)
        
        # ===== 完全照抄图e的min-max样式 =====
        # 垂直细线
        ax.vlines(
            x[i],
            vmin[i],
            vmax[i],
            color=COL_SEAS5,
            linewidth=1.5,
            zorder=5
        )
        # 上下两端短横线
        ax.hlines(
            [vmin[i], vmax[i]],
            x[i] - box_w * 0.28,
            x[i] + box_w * 0.28,
            color=COL_SEAS5,
            linewidth=1.4,
            zorder=5
        )
        # 中间加粗的mean横线
        ax.scatter(
            x[i],
            means[i],
            marker="_",
            s=260,
            color=COL_SEAS5,
            linewidths=2.3,
            zorder=6
        )
        
        # Annotate event counts
        top = vmax[i] + 0.12
        ax.text(x[i] - 0.05, top, f"SEAS5 n={ifs_counts[MONTHS_PLOT[i]]}",
                ha='center', va='bottom', fontsize=10, color=COL_SEAS5)
        ax.text(x[i] + 0.12, era5_vals[i] + 0.12, f"ERA5 n={era5_counts[MONTHS_PLOT[i]]}",
                ha='center', va='bottom', fontsize=10, color=COL_ERA5_EDGE)

    # ERA5 mean diamonds
    ax.scatter(x, era5_vals, marker='D', s=ERA5_DIAMOND_SIZE,
               color=ERA5_DIAMOND_FACE, edgecolors=ERA5_DIAMOND_EDGE,
               linewidths=ERA5_DIAMOND_LW, zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("SSW frequency (events / decade)")
    ax.set_xlabel("Month of SSW onset")
   
    style_ax(ax, add_minor_y=True)
    set_bold_axis_ticklabels(ax)

    # ===== 图例完全照抄图e =====
    leg_handles = [
        mpatches.Patch(facecolor=COL_SEAS5_FILL, edgecolor=COL_SEAS5, alpha=0.45,
                       label="SEAS5 10-90th pctl"),
        Line2D([0], [0], color=COL_SEAS5, lw=1.5, marker='_', markersize=10,
               markeredgewidth=2.0, label="SEAS5 min-max + mean"),
        Line2D([0], [0], marker='D', color='w', markersize=7.0,
               markerfacecolor=ERA5_DIAMOND_FACE, markeredgecolor=ERA5_DIAMOND_EDGE,
               markeredgewidth=ERA5_DIAMOND_LW, label="ERA5 mean"),
    ]
    ax.legend(handles=leg_handles, framealpha=0.95, loc='upper left', bbox_to_anchor=(0.04, 1))
    add_panel_label(ax, "(a)")

    del means, p10, p90, vmin, vmax, era5_vals, ifs_counts, era5_counts, x
    gc.collect()

# ================================================================
# CODE 2 — POST-SSW U10 DISTRIBUTION
# ================================================================
def draw_U10_distribution(ax, e_v, s_v):
    ks_stat, ks_p = stats.ks_2samp(e_v, s_v)
    sig = "**" if ks_p < 0.01 else ("*" if ks_p < 0.05 else "ns")
    xr_ = np.linspace(min(e_v.min(), s_v.min()) - 3,
                      max(e_v.max(), s_v.max()) + 3, 400)

    ax.hist(s_v, bins=38, density=True, color=COL_BLUE, alpha=0.32,
            edgecolor='white', linewidth=0.6, label=f"SEAS5 all members (n={len(s_v)})")
    ax.hist(e_v, bins=15, density=True, color=COL_RED,  alpha=0.44,
            edgecolor='white', linewidth=0.6, label=f"ERA5 (n={len(e_v)})")

    # Overlay KDE curves
    kde_s = gaussian_kde(s_v)(xr_)
    kde_e = gaussian_kde(e_v)(xr_)
    ax.plot(xr_, kde_s, color=COL_BLUE, lw=2.4, label="SEAS5 KDE")
    ax.plot(xr_, kde_e, color=COL_RED,  lw=2.4, label="ERA5 KDE")
    del kde_s, kde_e, xr_

    # Mean lines
    ax.axvline(np.mean(e_v), color=COL_RED,  ls=':', lw=1.9)
    ax.axvline(np.mean(s_v), color=COL_BLUE, ls=':', lw=1.9)

    handles, labels = ax.get_legend_handles_labels()
    labels.append(f"KS test p = {ks_p:.3f} ({sig})")
    handles.append(Line2D([], [], color='none'))
    ax.legend(handles, labels, loc='upper right', framealpha=0.95,fontsize=9)

    ax.set_xlabel("Post-SSW mean U10 (m s$^{-1}$)")
    ax.set_ylabel("Probability density")
   
    style_ax(ax, add_minor_y=False)
    set_bold_axis_ticklabels(ax)
    add_panel_label(ax, "(b)")

    del handles, labels
    gc.collect()


# ================================================================
# CODE 3 — POST-SSW T2m PANELS
# ================================================================
def draw_combined_t2m(ax, s5_dict, e5_dict, row_label, ylabel):
    rng         = np.random.default_rng(42)
    x_positions = np.arange(len(REGION_ORDER))
    box_w       = 0.24

    for i, rn in enumerate(REGION_ORDER):
        s5      = np.array(s5_dict[rn], dtype=float)
        e5      = np.array(e5_dict[rn], dtype=float)
        xm      = x_positions[i]
        p10     = np.percentile(s5, 10)
        p50     = np.percentile(s5, 50)
        p90     = np.percentile(s5, 90)
        mean_s5 = np.mean(s5)
        vmin    = np.min(s5)
        vmax    = np.max(s5)

        # SEAS5 individual member–event dots (jittered)
        jitter_s = rng.uniform(-box_w * 0.34, box_w * 0.34, len(s5))
        ax.scatter(xm + jitter_s, s5, marker='o', s=12, color=COL_SEAS5,
                   alpha=0.16, edgecolors='none', zorder=2)
        del jitter_s

        # 10–90th percentile box
        rect = Rectangle(
            (xm - box_w / 2, p10), box_w, p90 - p10,
            facecolor=COL_SEAS5_FILL, edgecolor=COL_SEAS5,
            linewidth=1.1, alpha=0.45, zorder=3
        )
        ax.add_patch(rect)
        
        # Median line
        ax.hlines(p50, xm - box_w / 2, xm + box_w / 2,
                  colors=COL_SEAS5, linewidth=2.2, zorder=5)
        
        # ===== 完全照抄图e的min-max样式 =====
        # 垂直细线
        ax.vlines(
            xm,
            vmin,
            vmax,
            color=COL_SEAS5,
            linewidth=1.5,
            zorder=5
        )
        # 上下两端短横线
        ax.hlines(
            [vmin, vmax],
            xm - box_w * 0.28,
            xm + box_w * 0.28,
            color=COL_SEAS5,
            linewidth=1.4,
            zorder=5
        )
        # 中间加粗的mean横线
        ax.scatter(
            xm,
            mean_s5,
            marker="_",
            s=260,
            color=COL_SEAS5,
            linewidths=2.3,
            zorder=6
        )

        # ERA5 individual event dots (jittered)
        jitter_e = rng.uniform(-box_w * 0.26, box_w * 0.26, len(e5))
        ax.scatter(xm + jitter_e, e5, marker='o', s=ERA5_EVENT_SIZE,
                   color=COL_ERA5, alpha=ERA5_EVENT_ALPHA,
                   edgecolors='white', linewidths=0.5, zorder=8)
        del jitter_e
        # ERA5 mean diamond
        ax.scatter(xm, np.mean(e5), marker='D', s=ERA5_DIAMOND_SIZE,
                   color=ERA5_DIAMOND_FACE, edgecolors=ERA5_DIAMOND_EDGE,
                   linewidths=ERA5_DIAMOND_LW, zorder=9)

        del s5, e5

    ax.set_xticks(x_positions)
    ax.set_xticklabels([REGION_NAMES[rn] for rn in REGION_ORDER])
    ax.set_ylabel(ylabel)
   
    ax.set_xlim(-0.55, len(REGION_ORDER) - 0.45)
    style_ax(ax, add_minor_y=False)
    set_bold_axis_ticklabels(ax)
    del x_positions
    gc.collect()


def draw_t2m_panel(ax, s5_dict, e5_dict, row_label, ylabel,
                     panel_label=None, add_legend=False):
    draw_combined_t2m(ax, s5_dict, e5_dict, row_label, ylabel)

    if add_legend:
        n_s5 = len(s5_dict[REGION_ORDER[0]])
        n_e5 = len(e5_dict[REGION_ORDER[0]])
        # ===== 图例完全照抄图e =====
        leg_handles = [
            Line2D([0], [0], marker='o', color='w', markersize=5.8,
                   markerfacecolor=COL_SEAS5, alpha=0.35, label="SEAS5 member×event"),
            mpatches.Patch(facecolor=COL_SEAS5_FILL, edgecolor=COL_SEAS5, alpha=0.45,
                           label=f"SEAS5 10-90th pctl (n={n_s5})"),
            Line2D([0], [0], color=COL_SEAS5, lw=1.5, marker='_', markersize=10,
                   markeredgewidth=2.0, label="SEAS5 min–max + mean"),
            Line2D([0], [0], marker='o', color='w', markersize=7.0,
                   markerfacecolor=COL_ERA5, alpha=ERA5_EVENT_ALPHA,
                   label=f"ERA5 events (n={n_e5})"),
            Line2D([0], [0], marker='D', color='w', markersize=7.0,
                   markerfacecolor=ERA5_DIAMOND_FACE, markeredgecolor=ERA5_DIAMOND_EDGE,
                   markeredgewidth=ERA5_DIAMOND_LW, label="ERA5 mean"),
        ]
        ax.legend(handles=leg_handles, loc='upper left', framealpha=0.95,
                  fontsize=8.4, bbox_to_anchor=(0.04, 1))
        del leg_handles

    if panel_label is not None:
        add_panel_label(ax, panel_label)


# ================================================================
# CODE 4 — SLIDING WINDOW BOOTSTRAP (from second code)
# ================================================================
def build_seas5_count_matrix(seas5_df):
    years = np.arange(START_YEAR, END_YEAR + 1)
    members = np.array(sorted(seas5_df["member"].unique()), dtype=int)

    count_mat = np.zeros((len(years), len(members)), dtype=float)

    grouped = seas5_df.groupby(["init_year", "member"]).size()

    for yi, yy in enumerate(years):
        for mi, mem in enumerate(members):
            count_mat[yi, mi] = grouped.get((yy, mem), 0)

    print("SEAS5 members used:", members)
    print("Number of SEAS5 members:", len(members))

    return years, members, count_mat


def build_era5_count_series(era5_df):
    years = np.arange(START_YEAR, END_YEAR + 1)
    grouped = era5_df.groupby("init_year").size()

    counts = np.array(
        [grouped.get(yy, 0) for yy in years],
        dtype=float
    )

    return years, counts


def compute_sliding_member_bootstrap(seas5_df, era5_df):
    rng = np.random.default_rng(BOOT_SEED)

    years, members, count_mat = build_seas5_count_matrix(seas5_df)
    _, era5_counts = build_era5_count_series(era5_df)

    windows = [
        (y, y + SLIDING_WINDOW - 1)
        for y in range(START_YEAR, END_YEAR - SLIDING_WINDOW + 2)
    ]

    x = []
    labels = []

    s5_mean = []
    s5_min = []
    s5_p10 = []
    s5_p90 = []
    s5_max = []

    era5_freq = []

    n_members = len(members)

    for ws, we in windows:
        idx = np.where((years >= ws) & (years <= we))[0]
        n_years = len(idx)

        random_members = rng.integers(
            low=0,
            high=n_members,
            size=(N_BOOT, n_years)
        )

        sampled_counts = count_mat[idx, :][
            np.arange(n_years)[None, :],
            random_members
        ]

        boot_freqs = sampled_counts.sum(axis=1) / n_years * 10.0

        s5_mean.append(np.nanmean(boot_freqs))
        s5_min.append(np.nanmin(boot_freqs))
        s5_p10.append(np.nanpercentile(boot_freqs, 10))
        s5_p90.append(np.nanpercentile(boot_freqs, 90))
        s5_max.append(np.nanmax(boot_freqs))

        era5_total = np.nansum(era5_counts[idx])
        era5_freq.append(era5_total / n_years * 10.0)

        x.append(ws + (SLIDING_WINDOW - 1) / 2.0)
        labels.append(f"{ws}–{we}")

    return {
        "x": np.array(x),
        "labels": labels,
        "s5_mean": np.array(s5_mean),
        "s5_min": np.array(s5_min),
        "s5_p10": np.array(s5_p10),
        "s5_p90": np.array(s5_p90),
        "s5_max": np.array(s5_max),
        "era5_freq": np.array(era5_freq),
    }


def draw_sliding_window(ax):
    """Draw the sliding window bootstrap figure (panel e)."""
    seas5_df = read_seas5_ssw_events()
    era5_df  = read_era5_ssw_events()
    
    print(f"  Sliding window - SEAS5 events: {len(seas5_df)}")
    print(f"  Sliding window - ERA5 events:  {len(era5_df)}")
    
    d = compute_sliding_member_bootstrap(seas5_df, era5_df)
    
    x = d["x"]
    s5_mean = d["s5_mean"]
    vmin = d["s5_min"]
    p10  = d["s5_p10"]
    p90  = d["s5_p90"]
    vmax = d["s5_max"]
    era5 = d["era5_freq"]
    box_w = 0.42

    for i in range(len(x)):
        rect = Rectangle(
            (x[i] - box_w / 2, p10[i]),
            box_w,
            p90[i] - p10[i],
            facecolor=COL_SEAS5_FILL,
            edgecolor=COL_SEAS5,
            linewidth=1.1,
            alpha=0.45,
            zorder=3
        )
        ax.add_patch(rect)

        ax.vlines(
            x[i],
            vmin[i],
            vmax[i],
            color=COL_SEAS5,
            linewidth=1.5,
            zorder=5
        )

        ax.hlines(
            [vmin[i], vmax[i]],
            x[i]-box_w*0.28,
            x[i]+box_w*0.28,
            color=COL_SEAS5,
            linewidth=1.4,
            zorder=5
        )

        ax.scatter(
            x[i],
            s5_mean[i],
            marker="_",
            s=260,
            color=COL_SEAS5,
            linewidths=2.3,
            zorder=6
        )

    ax.scatter(
        x,
        era5,
        marker="D",
        s=62,
        color=COL_ERA5,
        edgecolors=COL_ERA5,
        linewidths=0.8,
        zorder=8
    )

    outside = (era5 < vmin) | (era5 > vmax)
    if np.any(outside):
        ax.scatter(
            x[outside],
            era5[outside],
            marker="x",
            s=76,
            color="black",
            linewidths=1.6,
            zorder=9
        )

    # Show all tick labels but with rotation
    ax.set_xticks(x[::5])  # Show every 5th tick to avoid overcrowding
    ax.set_xticklabels(
        [d["labels"][i] for i in range(0, len(x), 5)],
        rotation=35,
        ha="right"
    )

    ax.set_ylabel("SSW frequency (events / 10 yrs)")
    ax.set_xlabel("10-year sliding window")

    ymax = max(12.0, np.nanmax(vmax) + 0.8, np.nanmax(era5) + 0.8)
    ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
    ax.set_ylim(-0.6, ymax)

    style_ax(ax, add_minor_y=False)
    set_bold_axis_ticklabels(ax)

    ax.text(
        0.01, 0.98, "(e)",  
        transform=ax.transAxes, 
        ha='left', 
        va='top',
        fontsize=13, 
        fontweight='bold', 
        color=COL_TEXT,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=1.2),
        zorder=20
    )
    

    # Legend
    handles = [
        Rectangle(
            (0, 0), 1, 1,
            facecolor=COL_SEAS5_FILL,
            edgecolor=COL_SEAS5,
            alpha=0.45,
            label="SEAS5 bootstrap 10-90th pctl"
        ),
        Line2D(
            [0], [0],
            color=COL_SEAS5,
            lw=1.5,
            marker="_",
            markersize=10,
            markeredgewidth=2.0,
            label="SEAS5 bootstrap min-max + mean"
        ),
        Line2D(
            [0], [0],
            marker="D",
            color="w",
            markerfacecolor=COL_ERA5,
            markeredgecolor=COL_ERA5,
            markersize=7,
            label="ERA5"
        ),
        Line2D(
            [0], [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=7,
            label="ERA5 outside SEAS5 min-max"
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.03, 1.0),
        framealpha=0.95,
        fontsize=9
    )
    
    del seas5_df, era5_df, d
    gc.collect()


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("Combined figure: 3 rows * 2 columns, last row spans full width")
    print("  (a) Monthly frequency  (b) U10 distribution")
    print("  (c) T2m raw           (d) T2m detrended")
    print("  (e) Sliding window frequency (full width)")
    print("=" * 72)

    # ensure both cache files exist (computed sequentially to save memory)
    ensure_caches()

    # load U10 cache and draw panel (b)
    print("Loading U10 cache for plotting ...")
    u10_era5, u10_seas5 = load_cache_u10(NPZ_U10)

    # load T2m cache
    print("Loading T2m cache for plotting ...")
    t2m_e_raw, t2m_e_det, t2m_s_raw, t2m_s_det = load_cache_t2m(NPZ_T2M)

    # Compose the figure: 3 rows × 2 columns, last row spans both columns
    fig = plt.figure(figsize=(15.5, 14.5))
    
    # Use GridSpec with 3 rows, 2 columns
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.9],
        hspace=0.32,
        wspace=0.24
    )

    # Panel (a) — monthly SSW frequency (row 0, col 0)
    ax_a = fig.add_subplot(gs[0, 0])
    draw_SSW_frequency(ax_a)

    # Panel (b) — post-SSW U10 distribution (row 0, col 1)
    ax_b = fig.add_subplot(gs[0, 1])
    draw_U10_distribution(ax_b, u10_era5, u10_seas5)
    del u10_era5, u10_seas5
    gc.collect()

    # Panel (c) — raw T2m (row 1, col 0)
    ax_c = fig.add_subplot(gs[1, 0])
    draw_t2m_panel(
        ax_c,
        t2m_s_raw, t2m_e_raw,
        f"Raw T2m (day +{DAY_START_T} to +{DAY_END_T})",
        "Post-SSW mean raw T2m (K)",
        panel_label="(c)",
        add_legend=True
    )
    ax_c.set_ylim(-10, 10)

    # Panel (d) — detrended T2m (row 1, col 1)
    ax_d = fig.add_subplot(gs[1, 1])
    draw_t2m_panel(
        ax_d,
        t2m_s_det, t2m_e_det,
        f"Detrended T2m (day +{DAY_START_T} to +{DAY_END_T})",
        "Post-SSW mean detrended T2m (K)",
        panel_label="(d)",
        add_legend=False
    )
    ax_d.set_ylim(-10, 10)
    
    del t2m_e_raw, t2m_e_det, t2m_s_raw, t2m_s_det
    gc.collect()

    # Panel (e) — sliding window frequency (row 2, spans both columns)
    ax_e = fig.add_subplot(gs[2, :])
    draw_sliding_window(ax_e)

    plt.savefig(PNG_OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved → {PNG_OUT}")
    print(f"U10 cache    → {NPZ_U10}")
    print(f"T2m cache    → {NPZ_T2M}")


if __name__ == "__main__":
    main()