# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from pathlib import Path
from scipy.stats import sem, theilslopes

plt.rcParams["font.family"] = "Arial"

# ================================================================
# SHARED SETTINGS
# ================================================================
SSW_SOURCE = "SEAS5"

SEAS5_SSW_CSV_PATH = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
ERA5_SSW_CSV_PATH  = Path(r"F:\data\ERA5_SSW_date\ERA5_SSW_dates_10hPa_NDJFM_events_only_1940_2024.csv")

T2M_DAILY_DIR = Path(r"F:\data\IFS_t2m_daily")
OUTPUT_DIR    = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR     = 1981
END_YEAR       = 2024
N_MEMBERS      = 25
BASELINE_START = 1981
BASELINE_END   = 2010

T2M_VAR_CANDIDATES = ["2m_temperature"]

REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": -140, "lon_max":  -60},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min":    0, "lon_max":   40},
    "EastAsia":      {"lat_min": 45, "lat_max": 70, "lon_min":   60, "lon_max":  120},
}
# REGION_TITLES = {
#     "NorthAmerica": "North America (45–70°N, 140–60°W)",
#     "Europe":       "Europe (45–70°N, 0–40°E)",
#     "EastAsia":      "East Asia (45–70°N, 60–120°E)",
# }

REGION_TITLES = {
    "NorthAmerica": "North America",
    "Europe":       "Europe",
    "EastAsia":      "East Asia",
}

REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]

SLIDING_WINDOW = 10

TIME_WINDOWS_CODE1 = [
    (  0, 15, "day 0 to +15"),
    ( 15, 30, "day +15 to +30"),
    ( 30, 45, "day +30 to +45"),
    ( 45, 60, "day +45 to +60"),
]

# lag composite：-20 到 +60
LAGS    = np.arange(-20, 61)

PERIOD1 = list(range(1981, 1991))
PERIOD2 = list(range(2015, 2025))


NPZ_LAG = OUTPUT_DIR / f"fig2_lag_combined_cache_{BASELINE_END}_m20to60.npz"
PNG_OUT_LAG = OUTPUT_DIR / f"fig2_SEAS5_SSW_T2m_lag_{BASELINE_END}.pdf"

# ================================================================
# HELPERS
# ================================================================
def get_var_name(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify t2m variable: {list(ds.data_vars)}")

def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c: raise ValueError(f"No lat dim: {da.dims}")
    if not lon_c: raise ValueError(f"No lon dim: {da.dims}")
    return lat_c[0], lon_c[0]

def get_time_dim_name(da):
    for name in ["valid_time", "time"]:
        if name in da.dims:
            return name
    raise ValueError(f"No time dim: {da.dims}")

def safe_member_label(number_coord, idx):
    try:    return int(number_coord[idx])
    except: return idx + 1

def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"

def cosine_weighted_region_mean(field2d, lat_vals, lon_vals, rb):
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    lon_mask = (lon_vals >= rb["lon_min"]) & (lon_vals <= rb["lon_max"])
    sub      = field2d[np.ix_(lat_mask, lon_mask)]
    weights  = np.cos(np.deg2rad(lat_vals[lat_mask]))
    w2d      = weights[:, None] * np.ones(lon_mask.sum())
    valid    = np.isfinite(sub)
    if valid.sum() == 0: return np.nan
    return float(np.nansum(sub * w2d * valid) / np.nansum(w2d * valid))

def _tag(w_label):
    return w_label.replace(" ", "_").replace("-", "m").replace("+", "p")


# ================================================================
# read daily file
# ================================================================
def load_one_year_daily(year):
    fp = T2M_DAILY_DIR / f"SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"
    if not fp.exists():
        print(f"  Missing: {fp.name}"); return None
    try:
        ds  = xr.open_dataset(fp)
        var = get_var_name(ds, T2M_VAR_CANDIDATES)
        da  = ds[var].load(); ds.close(); del ds

        if "number" not in da.dims:
            raise ValueError(f"Missing number dim: {da.dims}")
        da = da.transpose("number", *[d for d in da.dims if d != "number"])
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

        times         = pd.to_datetime(da[time_name].values).normalize()
        member_labels = [safe_member_label(da["number"].values, i)
                         for i in range(da.sizes["number"])]
        t2m_np        = da.values.astype(np.float32)
        lat_vals      = da[lat_name].values.copy()
        lon_vals      = da[lon_name].values.copy()
        del da

        print(f"  Loaded: {fp.name}  shape={t2m_np.shape}")
        gc.collect()
        return {
            "t2m":           t2m_np,
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
# read SSW events
# ================================================================
def read_ssw_events():
    if SSW_SOURCE.upper() == "SEAS5":
        df = pd.read_csv(SEAS5_SSW_CSV_PATH)
        df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["ssw_date"]).copy()
        df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
        df["member"] = df["member"].astype(int)
        df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
        print(f"Loaded {len(df)} SEAS5 SSW events.")
        return df
    elif SSW_SOURCE.upper() == "ERA5":
        df0 = pd.read_csv(ERA5_SSW_CSV_PATH)
        df0["ssw_date"] = pd.to_datetime(df0["ssw_date"], errors="coerce").dt.normalize()
        df0 = df0.dropna(subset=["ssw_date"]).copy()
        df0 = df0[(df0["init_year"] >= START_YEAR) & (df0["init_year"] <= END_YEAR)].copy()
        records = []
        for _, row in df0.iterrows():
            for member in range(N_MEMBERS):
                records.append({"init_year": int(row["init_year"]),
                                 "member": member,
                                 "ssw_date": pd.Timestamp(row["ssw_date"]).normalize()})
        df = pd.DataFrame(records)
        df = df.sort_values(["init_year", "member", "ssw_date"]).reset_index(drop=True)
        print(f"Loaded ERA5 SSW events expanded: {len(df)}")
        del df0, records; gc.collect(); return df
    else:
        raise ValueError("SSW_SOURCE must be 'SEAS5' or 'ERA5'")

def build_baseline_clim_streaming():
    print(f"\nBuilding smoothed baseline climatology (streaming) {BASELINE_START}-{BASELINE_END}...")

    doy_to_fields = {}
    lat_vals = lon_vals = None

    # ----------------------------
    # climatology
    # ----------------------------
    for year in range(BASELINE_START, BASELINE_END + 1):
        data = load_one_year_daily(year)
        if data is None:
            continue

        if lat_vals is None:
            lat_vals = data["lat_vals"]
            lon_vals = data["lon_vals"]

        ens_mean = data["t2m"].mean(axis=0)
        times    = data["times"]
        doys     = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(
                ens_mean[i].astype(np.float32)
            )

        del data, ens_mean, doys, times
        gc.collect()

    # ----------------------------
    # 366, lat, lon
    # ----------------------------
    print("  Building raw daily climatology...")

    nlat = len(lat_vals)
    nlon = len(lon_vals)

    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        stacked = np.stack(fields, axis=0)
        clim_raw[doy - 1] = np.nanmean(stacked, axis=0)
        del stacked

    del doy_to_fields
    gc.collect()

    # ----------------------------
    # rolling smoothing
    # ----------------------------
    print("  Applying ±5 day rolling smoothing...")

    window = 11
    pad = window // 2

    # circular padding
    clim_pad = np.concatenate([
        clim_raw[-pad:],
        clim_raw,
        clim_raw[:pad]
    ], axis=0)

    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i+window]

        # avoid empty slice warning
        if not np.isfinite(win).any():
            continue

        clim_smooth[i] = np.nanmean(win, axis=0)

    del clim_pad, clim_raw
    gc.collect()

    # ----------------------------
    # turn to monthday key 
    # ----------------------------
    print("  Mapping to monthday...")

    dates_ref = pd.date_range("2001-01-01", "2001-12-31")  
    baseline_clim = {}

    for d in dates_ref:
        key = f"{d.month:02d}-{d.day:02d}"
        baseline_clim[key] = clim_smooth[d.dayofyear - 1]

    del clim_smooth
    gc.collect()

    print(f"  Done: {len(baseline_clim)} keys (smoothed)")
    return baseline_clim, lat_vals, lon_vals
# ================================================================
# trend coefficient
# ================================================================
def build_trend_params_streaming(baseline_clim):
    print("\nBuilding cross-year trend params (streaming)...")
    md_ens_anom = {}
    for year in range(START_YEAR, END_YEAR + 1):
        data = load_one_year_daily(year)
        if data is None: continue
        ens_mean = data["t2m"].mean(axis=0)
        for i, t in enumerate(data["times"]):
            key = (t.month, t.day)
            cf  = baseline_clim.get(month_day_key(t))
            if cf is None: continue
            md_ens_anom.setdefault(key, []).append(
                (year, (ens_mean[i] - cf).astype(np.float32)))
        del data, ens_mean; gc.collect()

    trend_params = {}
    for key, entries in md_ens_anom.items():
        if len(entries) < 2: continue
        yr_arr = np.array([e[0] for e in entries], dtype=np.float64)
        yr_c   = yr_arr - yr_arr.mean()
        denom  = (yr_c ** 2).sum()
        if denom < 1e-10: continue
        stack = np.stack([e[1] for e in entries], 0).astype(np.float64)
        b     = (yr_c[:, None, None] * stack).sum(0) / denom
        a     = stack.mean(0)
        trend_params[key] = (a.astype(np.float32), b.astype(np.float32), float(yr_arr.mean()))
        del yr_arr, yr_c, stack, b, a
    del md_ens_anom; gc.collect()
    print(f"  Done. {len(trend_params)} keys.")
    return trend_params


# ================================================================
#  lag composite（lag = -10 to +60）
# ================================================================
def build_lag_composite_streaming(ssw_df, baseline_clim, trend_params, lat_vals, lon_vals):
    print("\nBuilding lag composites -10 to +60...")

    data_store = {
        tag: {rn: {lag: [] for lag in LAGS} for rn in REGION_ORDER}
        for tag in ["p1_raw", "p2_raw", "p1_det", "p2_det"]
    }

    for year in range(START_YEAR, END_YEAR + 1):
        dfp = ssw_df[ssw_df["init_year"] == year]
        if dfp.empty: continue

        data = load_one_year_daily(year)
        if data is None: continue

        t2m      = data["t2m"]
        times    = data["times"]
        m_to_idx = data["member_to_idx"]
        t_to_idx = data["time_to_idx"]
        tlen     = t2m.shape[1]
        nT, nlat, nlon = t2m.shape[1], t2m.shape[2], t2m.shape[3]

        clim_arr = np.full((nT, nlat, nlon), np.nan, dtype=np.float32)
        for i, t in enumerate(times):
            cf = baseline_clim.get(month_day_key(t))
            if cf is not None: clim_arr[i] = cf

        raw_anom = t2m - clim_arr[None]
        det_anom = raw_anom.copy()
        for i, t in enumerate(times):
            key   = (t.month, t.day)
            coeff = trend_params.get(key)
            if coeff is None: continue
            a, b, yr_mean = coeff
            det_anom[:, i] -= (a + b * (year - yr_mean))[None]
        del clim_arr; gc.collect()

        in_p1 = year in PERIOD1
        in_p2 = year in PERIOD2

        for _, row in dfp.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()
            if member not in m_to_idx or center not in t_to_idx: continue

            mi    = m_to_idx[member]
            c_idx = t_to_idx[center]

            for lag in LAGS:
                t_idx = c_idx + lag
                if t_idx < 0 or t_idx >= tlen: continue

                raw_field = raw_anom[mi, t_idx]
                det_field = det_anom[mi, t_idx]

                for rn, rb in REGION_BOXES.items():
                    rv = cosine_weighted_region_mean(raw_field, lat_vals, lon_vals, rb)
                    dv = cosine_weighted_region_mean(det_field, lat_vals, lon_vals, rb)
                    if in_p1:
                        if np.isfinite(rv): data_store["p1_raw"][rn][lag].append(rv)
                        if np.isfinite(dv): data_store["p1_det"][rn][lag].append(dv)
                    if in_p2:
                        if np.isfinite(rv): data_store["p2_raw"][rn][lag].append(rv)
                        if np.isfinite(dv): data_store["p2_det"][rn][lag].append(dv)

        del data, t2m, raw_anom, det_anom; gc.collect()

    def _agg(store):
        result = {}
        for rn in REGION_ORDER:
            means, sems_ = [], []
            for lag in LAGS:
                v = store[rn][lag]
                if len(v) >= 2:
                    means.append(np.mean(v)); sems_.append(sem(v))
                else:
                    means.append(np.nan); sems_.append(np.nan)
            result[rn] = (np.array(means), np.array(sems_))
        return result

    res1_raw = _agg(data_store["p1_raw"])
    res2_raw = _agg(data_store["p2_raw"])
    res1_det = _agg(data_store["p1_det"])
    res2_det = _agg(data_store["p2_det"])
    del data_store; gc.collect()
    return res1_raw, res2_raw, res1_det, res2_det


def save_lag_cache(npz_path, res1_raw, res2_raw, res1_det, res2_det):
    d = {}
    for rn in REGION_ORDER:
        for tag, res in [("r1raw", res1_raw), ("r2raw", res2_raw),
                         ("r1det", res1_det), ("r2det", res2_det)]:
            d[f"{rn}__{tag}__mean"] = res[rn][0]
            d[f"{rn}__{tag}__sem"]  = res[rn][1]
    np.savez(npz_path, **d); print(f"Saved lag cache: {npz_path}")


def load_lag_cache(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    def _load(tag):
        return {rn: (data[f"{rn}__{tag}__mean"], data[f"{rn}__{tag}__sem"])
                for rn in REGION_ORDER}
    return _load("r1raw"), _load("r2raw"), _load("r1det"), _load("r2det")


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

# ================================================================
# FIGURE 2: lag composite（-10 to +60）
# ================================================================
def plot_lag_only(res1_raw, res2_raw, res1_det, res2_det, output_png):
    print(f"Plotting lag-composite figure → {output_png}")

    ROW_COLORS   = ("#1C6AB1", "#ED4043")
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axes    = plt.subplots(2, 3, figsize=(20, 10.5))

    for (res1, res2, row_label, row_idx) in [
        (res1_raw, res2_raw, "Raw",       0),
        (res1_det, res2_det, "Detrended", 1),
    ]:
        for col_idx, rn in enumerate(REGION_ORDER):
            ax     = axes[row_idx, col_idx]
            m1, s1 = res1[rn]
            m2, s2 = res2[rn]

            ax.fill_between(LAGS, m1 - 1.96*s1, m1 + 1.96*s1, alpha=0.18, color=ROW_COLORS[0])
            ax.fill_between(LAGS, m2 - 1.96*s2, m2 + 1.96*s2, alpha=0.18, color=ROW_COLORS[1])

            ax.plot(LAGS, m1, lw=3.0, color=ROW_COLORS[0], label="1981–1990")
            ax.plot(LAGS, m2, lw=3.0, color=ROW_COLORS[1], label="2015–2024")

            ax.axvline(0,  ls="--", lw=1.0, color="black", alpha=0.7)
            ax.axhline(0,  ls="--", lw=1.0, color="black", alpha=0.7)

            ax.set_xlim(LAGS[0], LAGS[-1])
            ax.set_ylim(-2, 3)
   
            if row_idx == 0:
                ax.set_xticks([])  
            else:
                ax.set_xticks(np.arange(-20, 61, 10))  

            ax.grid(linestyle=":", alpha=0.4)
            ax.tick_params(axis='both', labelsize=14)
            if row_idx == 0:
                ax.set_title(REGION_TITLES[rn], fontsize=18, fontweight='bold', pad=6)

            panel_idx = row_idx * 3 + col_idx
            ax.text(0.02, 0.98, panel_labels[panel_idx], transform=ax.transAxes,
                    ha='left', va='top', fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5), zorder=10)

            if row_idx == 1:
                ax.set_xlabel("Lag (days relative to SSW onset)", fontsize=18, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nT2m anomaly (K)", fontsize=18, fontweight='bold')
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=16, loc='lower left')

    # fig.suptitle("SEAS5 SSW T2m lag composite",
    #              fontsize=18, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig); print(f"Saved: {output_png}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print(f"SSW source = {SSW_SOURCE}  |  Output = {OUTPUT_DIR}")
    print("=" * 60)

    ssw_df   = read_ssw_events()
    
    need_lag = not NPZ_LAG.exists()

   
    res1_raw = res2_raw = res1_det = res2_det = None

   
    if not need_lag:
        print("\nLoading lag composite cache...")
        res1_raw, res2_raw, res1_det, res2_det = load_lag_cache(NPZ_LAG)

    baseline_clim, lat_vals, lon_vals = build_baseline_clim_streaming()
    trend_params = build_trend_params_streaming(baseline_clim)

    if need_lag:
        res1_raw, res2_raw, res1_det, res2_det = build_lag_composite_streaming(
            ssw_df, baseline_clim, trend_params, lat_vals, lon_vals)
        save_lag_cache(NPZ_LAG, res1_raw, res2_raw, res1_det, res2_det)

        del trend_params, lat_vals, lon_vals; gc.collect()

   
    plot_lag_only(res1_raw, res2_raw, res1_det, res2_det, PNG_OUT_LAG)

    del baseline_clim,ssw_df, res1_raw, res2_raw, res1_det, res2_det
    gc.collect()
    print("\nAll done.")


if __name__ == "__main__":
    main()