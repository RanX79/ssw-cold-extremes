# -*- coding: utf-8 -*-
import gc
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import theilslopes

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1

# ================================================================
# SETTINGS
# ================================================================
SSW_CSV_PATH = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")

START_YEAR     = 1981
END_YEAR       = 2024
N_MEMBERS      = 25
BASELINE_START = 1981
BASELINE_END   = 2010

SLIDING_WINDOW = 10

TIME_WINDOWS = [
    (-20, -1,  "day -20 to -1"),
    (  0, 29,  "day 0 to +29"),
    ( 30, 59,  "day +30 to +59"),
]
LEGEND_LABEL_MAP = {
    "day -20 to -1":    "Precursor phase",
    "day 0 to +29":     "Onset phase",
    "day +30 to +59":   "Decay phase",
}
WIN_COLORS = {
    "day -20 to -1":  "#7A3E8E",
    "day 0 to +29":   "#2166AC",
    "day +30 to +59": "#D73027",
}

# ── output directory ──────────────────────────────────────────────
OUTPUT_ROOT    = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_ROOT / f"fig3_combined_SEAS5_SSW_T2m_U_10yrsliding_{BASELINE_END}.pdf"


# ================================================================
# T2M SETTINGS
# ================================================================
T2M_DAILY_DIR = Path(r"F:\data\IFS_t2m_daily")
T2M_VAR_CANDIDATES = ["2m_temperature"]

REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": -140, "lon_max": -60},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min": 0,    "lon_max": 40},
    "EastAsia":      {"lat_min": 45, "lat_max": 70, "lon_min": 60,   "lon_max": 120},
}
REGION_TITLES = {
    "NorthAmerica": "North America",
    "Europe":       "Europe",
    "EastAsia":     "East Asia",
}
REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]

NPZ_TS_T2M = OUTPUT_ROOT / f"fig3_SEAS5_SSW_T2m_timeseries_{SLIDING_WINDOW}Y_sliding_3regions_cache_daily_3win_m20_{BASELINE_END}.npz"
NPZ_TS_U   = OUTPUT_ROOT / f"fig3_SEAS5_SSW_U_10_100hPa_{SLIDING_WINDOW}Y_sliding_cache_daily_3win_m20_{BASELINE_END}.npz"

# ================================================================
# U SETTINGS
# ================================================================
U_DAILY_DIRS = {
    10:  Path(r"F:\data\IFS_U10_daily"),
    100: Path(r"F:\data\IFS_U100_daily"),
}
LEVELS_TO_PLOT = [10, 100]   # just 10 and 100 hPa

U_LAT = 60.0

# ================================================================
# HELPERS
# ================================================================
def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def read_ssw_events():
    df = pd.read_csv(SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"]).copy()
    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["member"] = df["member"].astype(int)
    df["month"]  = df["ssw_date"].dt.month
    df = df[df["month"].isin([11,12, 1, 2,3])].copy()
    df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
    print(f"Loaded {len(df)} SSW events")
    return df


def block_bootstrap_trend(xv, mv, n_boot=1000, block_size=3, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(mv)
    if n < 3: return np.nan, np.nan
    obs_slope   = theilslopes(mv, xv).slope
    boot_slopes = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            indices.extend(range(start, min(start + block_size, n)))
        boot_slopes.append(theilslopes(mv[np.array(indices[:n])], xv).slope)
    boot_slopes   = np.array(boot_slopes)
    boot_centered = boot_slopes - np.mean(boot_slopes)
    return obs_slope, float(np.mean(np.abs(boot_centered) >= np.abs(obs_slope)))


def p_to_sig(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def get_line_trend_label(x, y, fmt="{:+.2f}"):
    mask = np.isfinite(y)
    if mask.sum() <= 5: return None, None, None
    x1, y1   = x[mask], y[mask]
    ts_res   = theilslopes(y1, x1)
    _, p_val = block_bootstrap_trend(x1, y1, n_boot=1000, block_size=3)
    # R=Raw, D=Detrended
    slope_decade = ts_res.slope * 10.0   # to decade

    return x1[-1], y1[-1], fmt.format(slope_decade) + p_to_sig(p_val)


def adjust_label_positions(items, ymin, ymax, min_gap):
    if not items: return items
    items_s = sorted(items, key=lambda z: z["y"])
    items_s[0]["y_adj"] = items_s[0]["y"]
    for i in range(1, len(items_s)):
        items_s[i]["y_adj"] = max(items_s[i]["y"], items_s[i-1]["y_adj"] + min_gap)
    for _ in range(2):
        ov = items_s[-1]["y_adj"] - ymax
        if ov > 0:
            for it in items_s: it["y_adj"] -= ov
        uv = ymin - items_s[0]["y_adj"]
        if uv > 0:
            for it in items_s: it["y_adj"] += uv
        for i in range(len(items_s)-2, -1, -1):
            if items_s[i]["y_adj"] > items_s[i+1]["y_adj"] - min_gap:
                items_s[i]["y_adj"] = items_s[i+1]["y_adj"] - min_gap
    return items_s


def add_right_side_labels(ax, label_items, xpad, fs=16):
    if not label_items: return
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    adjusted = adjust_label_positions(
        label_items, ymin - 0.20*yr, ymax + 0.20*yr, min_gap=0.06*yr)

    for item in adjusted:
        ax.text(item["x"] + xpad, item["y_adj"], item["label"],
                color=item["color"], fontsize=fs,
                ha='left', va='center', fontweight='bold', clip_on=False)


def style_axis(ax):
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
    # ax.grid(True, axis='y', linestyle='--', alpha=0.18, lw=0.5)
    # ax.grid(False, axis='x')
    ax.grid(True, axis="both", linestyle=":", alpha=0.4, lw=0.7)


def get_var_name(ds, candidates):
    for v in candidates:
        if v in ds.data_vars: return v
    if len(ds.data_vars) == 1: return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify variable: {list(ds.data_vars)}")


def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c: raise ValueError(f"No lat dim: {da.dims}")
    if not lon_c: raise ValueError(f"No lon dim: {da.dims}")
    return lat_c[0], lon_c[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time"]:
        if name in da.dims: return name
    raise ValueError(f"No time dim: {da.dims}")


def safe_member_label(number_coord, idx):
    try:    return int(number_coord[idx])
    except: return idx + 1


# ================================================================
# read daily file
# ================================================================
def load_one_year_daily(year, daily_dir, file_pattern, var_candidates,
                        flip_lat=True, wrap_lon=True):
    fp = daily_dir / file_pattern.format(year=year)
    if not fp.exists():
        print(f"  Missing: {fp.name}"); return None
    try:
        ds  = xr.open_dataset(fp)
        var = get_var_name(ds, var_candidates)
        da  = ds[var].load(); ds.close(); del ds

        if "number" not in da.dims:
            raise ValueError(f"Missing number dim: {da.dims}")
        da = da.transpose("number", *[d for d in da.dims if d != "number"])
        da = da.isel(number=slice(0, N_MEMBERS))

        for dim in list(da.dims):
            if dim not in ("number",) and da.sizes[dim] == 1 and \
               "lat" not in dim.lower() and "lon" not in dim.lower() and \
               "time" not in dim.lower() and "valid" not in dim.lower():
                da = da.squeeze(dim, drop=True)

        lat_name, lon_name = get_lat_lon_dim_names(da)
        time_name          = get_time_dim_name(da)

        if flip_lat and da[lat_name].values[0] < da[lat_name].values[-1]:
            da = da.isel({lat_name: slice(None, None, -1)})

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
        del da; gc.collect()

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
        print(f"  Failed year={year}: {e}"); gc.collect(); return None


# ================================================================
# sliding window
# ================================================================
def _cosine_weighted_mean(field2d, lat_vals, lon_vals, rb):
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    lon_mask = (lon_vals >= rb["lon_min"]) & (lon_vals <= rb["lon_max"])
    sub      = field2d[np.ix_(lat_mask, lon_mask)]
    weights  = np.cos(np.deg2rad(lat_vals[lat_mask]))
    w2d      = weights[:, None] * np.ones(lon_mask.sum())
    valid    = np.isfinite(sub)
    if valid.sum() == 0: return np.nan
    return float(np.nansum(sub * w2d * valid) / np.nansum(w2d * valid))


def _zonal_mean_at_lat(field2d, lat_vals, lon_vals, target_lat):
    lat_idx = np.argmin(np.abs(lat_vals - target_lat))
    row     = field2d[lat_idx, :]
    valid   = np.isfinite(row)
    if valid.sum() == 0: return np.nan
    return float(np.nanmean(row[valid]))


def _build_baseline_clim(years, daily_dir, file_pattern, var_candidates):
    print(f"  Building rolling baseline climatology {years[0]}-{years[-1]} (±5 days)...")

    doy_to_fields = {}
    lat_vals = lon_vals = None

    # ----------------------------
    # collect raw climatology
    # ----------------------------
    for year in years:
        data = load_one_year_daily(year, daily_dir, file_pattern, var_candidates)
        if data is None:
            continue

        if lat_vals is None:
            lat_vals = data["lat_vals"]
            lon_vals = data["lon_vals"]

        ens_mean = data["arr"].mean(axis=0)
        times    = data["times"]

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(
                ens_mean[i].astype(np.float32)
            )

        del data, ens_mean

    gc.collect()

    # ----------------------------
    # build raw climatology
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
    # rolling smoothing
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
    # map back to month-day 
    # ----------------------------
    clim = {}

    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        clim[key] = clim_smooth[t.dayofyear - 1]

    del clim_smooth
    gc.collect()

    print(f"  Done: {len(clim)} keys (smoothed)")
    return clim, lat_vals, lon_vals

def _build_trend_params(clim, daily_dir, file_pattern, var_candidates):
    print(f"  Building trend params...")
    md_anom = {}
    for year in range(START_YEAR, END_YEAR + 1):
        data = load_one_year_daily(year, daily_dir, file_pattern, var_candidates)
        if data is None: continue
        ens_mean = data["arr"].mean(axis=0)
        for i, t in enumerate(data["times"]):
            key = (t.month, t.day)
            cf  = clim.get(month_day_key(t))
            if cf is None: continue
            md_anom.setdefault(key, []).append(
                (year, (ens_mean[i] - cf).astype(np.float32)))
        del data, ens_mean; gc.collect()
    trend_params = {}
    for key, entries in md_anom.items():
        if len(entries) < 2: continue
        yr_arr = np.array([e[0] for e in entries], dtype=np.float64)
        yr_c   = yr_arr - yr_arr.mean()
        denom  = (yr_c ** 2).sum()
        if denom < 1e-10: continue
        stack  = np.stack([e[1] for e in entries], 0).astype(np.float64)
        b      = (yr_c[:, None, None] * stack).sum(0) / denom
        a      = stack.mean(0)
        trend_params[key] = (a.astype(np.float32), b.astype(np.float32), float(yr_arr.mean()))
        del yr_arr, yr_c, stack, b, a
    del md_anom; gc.collect()
    print(f"  Trend params: {len(trend_params)} keys")
    return trend_params


def _extract_event_samples(ssw_df, clim, trend_params, lat_vals, lon_vals,
                            daily_dir, file_pattern, var_candidates, extract_fn):
    event_cache = {w_label: {} for _, _, w_label in TIME_WINDOWS}

    for year in range(START_YEAR, END_YEAR + 1):
        dfp = ssw_df[ssw_df["init_year"] == year]
        if dfp.empty: continue

        data = load_one_year_daily(year, daily_dir, file_pattern, var_candidates)
        if data is None: continue

        arr      = data["arr"]
        monthday = data["monthday"]
        times    = data["times"]
        m_to_idx = data["member_to_idx"]
        t_to_idx = data["time_to_idx"]
        tlen     = arr.shape[1]
        nT, nlat, nlon = arr.shape[1], arr.shape[2], arr.shape[3]

        clim_arr = np.full((nT, nlat, nlon), np.nan, dtype=np.float32)
        for i, key in enumerate(monthday):
            cf = clim.get(key)
            if cf is not None: clim_arr[i] = cf

        raw_anom = arr - clim_arr[None]
        det_anom = raw_anom.copy()
        for i, t in enumerate(times):
            key   = (t.month, t.day)
            coeff = trend_params.get(key)
            if coeff is None: continue
            a, b, yr_mean = coeff
            # det_anom[:, i] -= (a + b * (year - yr_mean))[None]
            det_anom[:, i] -= ( b * (year - yr_mean))[None]
        del clim_arr; gc.collect()

        for _, row in dfp.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()
            if member not in m_to_idx or center not in t_to_idx: continue
            mi    = m_to_idx[member]
            c_idx = t_to_idx[center]
            for d_start, d_end, w_label in TIME_WINDOWS:
                i0 = max(c_idx + d_start, 0)
                i1 = min(c_idx + d_end + 1, tlen)
                if (i1 - i0) != (d_end - d_start + 1): continue
                raw_field = np.nanmean(raw_anom[mi, i0:i1], axis=0)
                det_field = np.nanmean(det_anom[mi, i0:i1], axis=0)
                event_cache[w_label][(year, member, center)] = (
                    extract_fn(raw_field, lat_vals, lon_vals),
                    extract_fn(det_field, lat_vals, lon_vals)
                )
                del raw_field, det_field

        del data, arr, raw_anom, det_anom; gc.collect()

    return event_cache


def _aggregate_sliding(ssw_df, event_cache):
    yr_min  = int(ssw_df["init_year"].min())
    yr_max  = int(ssw_df["init_year"].max())
    windows = [(y, y + SLIDING_WINDOW - 1)
               for y in range(yr_min, yr_max - SLIDING_WINDOW + 2)]
    ts_data = {}
    for _, _, w_label in TIME_WINDOWS:
        cache = event_cache[w_label]
        xc, wl, rm, rs, dm, ds, rn = [], [], [], [], [], [], []
        for w_start, w_end in windows:
            dfp_win  = ssw_df[(ssw_df["init_year"] >= w_start) &
                              (ssw_df["init_year"] <= w_end)]
            rv_list, dv_list = [], []
            for _, row in dfp_win.iterrows():
                key = (int(row["init_year"]), int(row["member"]),
                       pd.Timestamp(row["ssw_date"]).normalize())
                if key in cache:
                    rv, dv = cache[key]
                    rv_list.append(rv); dv_list.append(dv)
            n = len(rv_list)
            rm.append(np.nanmean(rv_list) if n > 0 else np.nan)
            rs.append(np.nanstd(rv_list, ddof=1) / np.sqrt(n) if n > 1 else np.nan)
            dm.append(np.nanmean(dv_list) if n > 0 else np.nan)
            ds.append(np.nanstd(dv_list, ddof=1) / np.sqrt(n) if n > 1 else np.nan)
            rn.append(n)
            xc.append(w_start + (SLIDING_WINDOW - 1) / 2.0)
            wl.append(f"{w_start}\N{EN DASH}{w_end}")
        ts_data[w_label] = {
            "xc": np.array(xc), "wl": wl,
            "rm": np.array(rm), "rs": np.array(rs),
            "dm": np.array(dm), "ds": np.array(ds),
            "rn": np.array(rn),
        }
    return ts_data


def _tag(w_label):
    return w_label.replace(" ", "_").replace("-", "m").replace("+", "p")


# ================================================================
# T2M PIPELINE
# ================================================================
def compute_t2m_sliding(ssw_df):
    print("\n" + "="*60)
    print("T2m sliding window pipeline (streaming daily)")
    baseline_years = list(range(BASELINE_START, BASELINE_END + 1))
    file_pattern   = "SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"
    clim, lat_vals, lon_vals = _build_baseline_clim(
        baseline_years, T2M_DAILY_DIR, file_pattern, T2M_VAR_CANDIDATES)
    trend_params = _build_trend_params(
        clim, T2M_DAILY_DIR, file_pattern, T2M_VAR_CANDIDATES)
    all_ts = {}
    for rname, rb in REGION_BOXES.items():
        print(f"  Region: {rname}")
        def _fn(field, lv, lnv, _rb=rb):
            return _cosine_weighted_mean(field, lv, lnv, _rb)
        event_cache    = _extract_event_samples(
            ssw_df, clim, trend_params, lat_vals, lon_vals,
            T2M_DAILY_DIR, file_pattern, T2M_VAR_CANDIDATES, _fn)
        all_ts[rname]  = _aggregate_sliding(ssw_df, event_cache)
        del event_cache; gc.collect()
    del clim, trend_params, lat_vals, lon_vals; gc.collect()
    d = {}
    for rn, ts_data in all_ts.items():
        for w_label, v in ts_data.items():
            t = f"{rn}__{_tag(w_label)}"
            for k2, arr in [("xc", v["xc"]), ("wl", np.array(v["wl"])),
                             ("rm", v["rm"]), ("rs", v["rs"]),
                             ("dm", v["dm"]), ("ds", v["ds"]), ("rn", v["rn"])]:
                d[f"{t}__{k2}"] = arr
    np.savez(NPZ_TS_T2M, **d)
    print(f"Saved T2m cache: {NPZ_TS_T2M}")
    return all_ts


def load_t2m_sliding():
    data   = np.load(NPZ_TS_T2M, allow_pickle=True)
    all_ts = {}
    for rn in REGION_ORDER:
        ts_data = {}
        for _, _, w_label in TIME_WINDOWS:
            t = f"{rn}__{_tag(w_label)}"
            ts_data[w_label] = {
                "xc": data[f"{t}__xc"], "wl": list(data[f"{t}__wl"]),
                "rm": data[f"{t}__rm"], "rs": data[f"{t}__rs"],
                "dm": data[f"{t}__dm"], "ds": data[f"{t}__ds"],
                "rn": data[f"{t}__rn"].astype(int),
            }
        all_ts[rn] = ts_data
    print(f"Loaded T2m cache: {NPZ_TS_T2M}")
    return all_ts


# ================================================================
# U PIPELINE
# ================================================================
def compute_u_sliding(ssw_df):
    print("\n" + "="*60)
    print("U sliding window pipeline (streaming daily)")
    all_ts_u       = {}
    baseline_years = list(range(BASELINE_START, BASELINE_END + 1))
    for lev in LEVELS_TO_PLOT:
        print(f"\n  Level: {lev} hPa")
        daily_dir    = U_DAILY_DIRS[lev]
        file_pattern = f"SEAS5_u{lev}hPa_NH_{{year}}11_system51_m25_daily.nc"
        var_cands    = ["u", f"u{lev}", "uwnd", "var131"]
        clim, lat_vals, lon_vals = _build_baseline_clim(
            baseline_years, daily_dir, file_pattern, var_cands)
        trend_params = _build_trend_params(clim, daily_dir, file_pattern, var_cands)
        def _fn(field, lv, lnv, _lat=U_LAT):
            return _zonal_mean_at_lat(field, lv, lnv, _lat)
        event_cache      = _extract_event_samples(
            ssw_df, clim, trend_params, lat_vals, lon_vals,
            daily_dir, file_pattern, var_cands, _fn)
        all_ts_u[lev]    = _aggregate_sliding(ssw_df, event_cache)
        del clim, trend_params, event_cache, lat_vals, lon_vals; gc.collect()
    d = {}
    for lev, ts_data in all_ts_u.items():
        for w_label, v in ts_data.items():
            t = f"u{lev}__{_tag(w_label)}"
            for k2, arr in [("xc", v["xc"]), ("wl", np.array(v["wl"])),
                             ("rm", v["rm"]), ("rs", v["rs"]),
                             ("dm", v["dm"]), ("ds", v["ds"]), ("rn", v["rn"])]:
                d[f"{t}__{k2}"] = arr
    np.savez(NPZ_TS_U, **d)
    print(f"Saved U cache: {NPZ_TS_U}")
    return all_ts_u


def load_u_sliding():
    data     = np.load(NPZ_TS_U, allow_pickle=True)
    all_ts_u = {}
    for lev in LEVELS_TO_PLOT:
        ts_data = {}
        for _, _, w_label in TIME_WINDOWS:
            t = f"u{lev}__{_tag(w_label)}"
            ts_data[w_label] = {
                "xc": data[f"{t}__xc"], "wl": list(data[f"{t}__wl"]),
                "rm": data[f"{t}__rm"], "rs": data[f"{t}__rs"],
                "dm": data[f"{t}__dm"], "ds": data[f"{t}__ds"],
                "rn": data[f"{t}__rn"].astype(int),
            }
        all_ts_u[lev] = ts_data
    print(f"Loaded U cache: {NPZ_TS_U}")
    return all_ts_u


# ================================================================
#FIGURE: 3 items in the first row (T2m), 2 items in the second row (U 10/100 hPa), centered alignment
# ================================================================
def _plot_one_panel(ax, ts_data, ylim, xpad_frac=0.02, fs_trend=11):
    label_items_raw = []
    label_items_det = []

    for _, _, w_label in TIME_WINDOWS:
        d   = ts_data[w_label]
        xc  = d["xc"]
        rm  = d["rm"]; rs = d["rs"]
        dm  = d["dm"]; ds = d["ds"]
        col = WIN_COLORS[w_label]

        ax.plot(xc, rm, color=col, lw=2.5, ls='-',  zorder=4)
        ax.fill_between(xc, rm-rs, rm+rs, color=col, alpha=0.12, zorder=2)
        ax.plot(xc, dm, color=col, lw=2.5, ls='--', zorder=4)
        ax.fill_between(xc, dm-ds, dm+ds, color=col, alpha=0.07, zorder=2)

        xe, ye, lbl = get_line_trend_label(xc, rm)
        if lbl: label_items_raw.append(
            {"x": xe, "y": ye, "label": lbl+" (R)", "color": col})
        xe, ye, lbl = get_line_trend_label(xc, dm)
        if lbl: label_items_det.append(
            {"x": xe, "y": ye, "label": lbl+" (D)", "color": col})

    ax.axhline(0, color='k', lw=0.9, zorder=3)

    xc_ref = ts_data[TIME_WINDOWS[0][2]]["xc"]
    wl_ref = ts_data[TIME_WINDOWS[0][2]]["wl"]

    # ============================================================
    # X-axis label: Display one every 5 years, in the format "1981–1990"
    # ============================================================
    step = 5
    tick_idx = np.arange(0, len(xc_ref), step)
    
    
    xlabels = []
    for i in tick_idx:
        if i < len(wl_ref):
            label = wl_ref[i]
            if '–' in label or '-' in label:
                import re
                years = re.findall(r'\d{4}', label)
                if len(years) >= 2:
                    xlabels.append(f"{years[0]}–{years[1]}")
                else:
                    xlabels.append(label)
            else:
                xlabels.append(label)
        else:
            xlabels.append("")
    
    ax.set_xticks(xc_ref[tick_idx])
    ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=12)

    xrange = xc_ref[-1] - xc_ref[0]

    clipped_raw = []
    clipped_det = []
    for item in label_items_raw:
        clipped_raw.append({**item, "y": np.clip(item["y"], ylim[0], ylim[1])})
    for item in label_items_det:
        clipped_det.append({**item, "y": np.clip(item["y"], ylim[0], ylim[1])})

    ax.set_xlim(xc_ref[0] - 0.3, xc_ref[-1] + 0.3)
    ax.set_ylim(ylim)
    ax.autoscale(enable=False)
    ax.tick_params(axis='y', labelsize=11)

    add_right_side_labels(ax, clipped_raw + clipped_det,
                          xrange * xpad_frac, fs=fs_trend)

    ax.set_xlim(xc_ref[0] - 0.3, xc_ref[-1] + 0.3)
    ax.set_ylim(ylim)

    style_axis(ax)

def plot_combined(all_ts_t2m, all_ts_u, output_png):
    print(f"\nPlotting combined figure → {output_png}")

    fig = plt.figure(figsize=(21, 10))
    gs_top = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.06, right=0.88,
                               top=0.93, bottom=0.55,
                               wspace=0.32)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.20, right=0.75,
                               top=0.43, bottom=0.08,
                               wspace=0.35)

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    ax_row0 = []

    # ── Row 1: T2m ──────────────────────────────────────────────
    for ci, rname in enumerate(REGION_ORDER):
        ax = fig.add_subplot(gs_top[0, ci])
        ax_row0.append(ax)
        _plot_one_panel(ax, all_ts_t2m[rname], ylim=(-1.5, 2))
        ax.set_title(REGION_TITLES[rname], fontsize=16, fontweight='bold', pad=5)
        ax.text(0.02, 0.97, panel_labels[ci],
                transform=ax.transAxes, ha='left', va='top',
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        if ci == 0:
            ax.set_ylabel("T2m anomaly (K)", fontsize=16, fontweight='bold')

    # ── Row 2: U 10 和 100 hPa ───────────────────────────────────
    u_titles = ["10 hPa U (60°N)", "100 hPa U (60°N)"]
    u_ylims  = [(-20, 8), (-6, 4)]

    for ci, lev in enumerate(LEVELS_TO_PLOT):
        ax = fig.add_subplot(gs_bot[0, ci])
        _plot_one_panel(ax, all_ts_u[lev], ylim=u_ylims[ci])
        ax.set_title(u_titles[ci], fontsize=16, fontweight='bold', pad=5)
        ax.text(0.02, 0.97, panel_labels[3 + ci],
                transform=ax.transAxes, ha='left', va='top',
                fontsize=16, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        if ci == 0:
            ax.set_ylabel("U anomaly (m s$^{-1}$)", fontsize=16, fontweight='bold')

    # ── legend ─────────────────────────────────────────────────────
    solid_ln  = mlines.Line2D([], [], color='grey', ls='-',  lw=2,
                               label='Solid = Raw (R)')
    dashed_ln = mlines.Line2D([], [], color='grey', ls='--', lw=2,
                               label='Dashed = Detrended (D)')
    shade_pt  = mlines.Line2D([], [], color='grey', ls='-',  lw=6,
                               alpha=0.3, label='±1 SE')

    legend_label_map = {
        "day -20 to -1":  "Precursor phase",
        "day 0 to +29":   "Onset phase",
        "day +30 to +59": "Decay phase",
    }

    w_handles = [
        mlines.Line2D([], [], color=WIN_COLORS[wl], ls='-', lw=2.5,
                      label=legend_label_map[wl])
        for _, _, wl in TIME_WINDOWS
    ]

    ax_row0[0].legend(
        handles=[solid_ln, dashed_ln, shade_pt] + w_handles,
        loc='lower left',
        fontsize=12,
        framealpha=0.88,
        ncol=2,
        handlelength=2.0,
    )

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_png}")
# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(f"Output: {OUTPUT_PNG}")
    print("=" * 60)

    ssw_df = read_ssw_events()

    # After completing the T2m calculation and saving it, release it, and then proceed with the U calculation
    if NPZ_TS_T2M.exists():
        all_ts_t2m = load_t2m_sliding()
    else:
        all_ts_t2m = compute_t2m_sliding(ssw_df)
    gc.collect()

    if NPZ_TS_U.exists():
        all_ts_u = load_u_sliding()
    else:
        all_ts_u = compute_u_sliding(ssw_df)
    gc.collect()

    plot_combined(all_ts_t2m, all_ts_u, OUTPUT_PNG)

    del ssw_df, all_ts_t2m, all_ts_u
    gc.collect()
    print("\nAll done.")


if __name__ == "__main__":
    main()