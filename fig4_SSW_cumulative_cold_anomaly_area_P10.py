# -*- coding: utf-8 -*-
"""
SSW Post-event Cumulative Cold-Anomaly Area — Raw vs Detrended
10-year sliding window

Metric:
    For each day during day +15 to +59:
        calculate area fraction within each region where
        Tanom < baseline 10th percentil
    Then sum daily area fraction over day +15 to +59.
Unit: % area × day

Raw: raw_anom = T - baseline_climatology
Detrended: det = detrended anomaly from compute_detrended_year()
"""

import gc
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import theilslopes

plt.rcParams["font.family"] = "Arial"

# ================================================================
# settings
# ================================================================
SSW_CSV_PATH  = Path(r"F:\data\SSW_results\SEAS5_first25members_SSW_dates_NDJFM_events_only_1981_2024.csv")
T2M_DAILY_DIR = Path(r"F:\data\IFS_t2m_daily")
OUTPUT_DIR    = Path(r"F:\data\paper_SSW_impacts_under_global_warming\figure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR     = 1981
END_YEAR       = 2024
N_MEMBERS      = 25

DAY_START      = 15
DAY_END        = 59
N_DAYS         = DAY_END - DAY_START + 1
SLIDING_WINDOW = 10

BASELINE_START = 1981
BASELINE_END   = 2010

PERCENTILE_THRESHOLD = 10.0

T2M_VAR_CANDIDATES = ["t2m", "2t", "2m_temperature", "t2m_daily"]

# REGION_BOXES = {
#     "NorthAmerica": {"lat_min": 35, "lat_max": 50, "lon_min": -95, "lon_max": -65},
#     "Europe":       {"lat_min": 45, "lat_max": 60, "lon_min": 0,   "lon_max": 40},
#     "EastAsia":     {"lat_min": 35, "lat_max": 45, "lon_min": 110, "lon_max": 130},
# }
# REGION_LABELS = {
#     "NorthAmerica": "N. America (35-50°N, 95-65°W)",
#     "Europe":       "Europe (45-60°N, 0-40°E)",
#     "EastAsia":     "East Asia (35-45°N, 110-130°E)",
# }
REGION_BOXES = {
    "NorthAmerica": {"lat_min": 45, "lat_max": 70, "lon_min": -140, "lon_max": -60},
    "Europe":       {"lat_min": 45, "lat_max": 70, "lon_min": 0,    "lon_max": 40},
    "EastAsia":      {"lat_min": 45, "lat_max": 70, "lon_min": 60,   "lon_max": 120},
}
# REGION_LABELS = {
#     "NorthAmerica": "N. America (45-70°N, 140-60°W)",
#     "Europe":       "Europe (45-70°N, 0-40°E)",
#     "EastAsia":     "East Asia (45-70°N, 60-120°E)",
# }
REGION_LABELS = {
    "NorthAmerica": "North America",
    "Europe":       "Europe",
    "EastAsia":     "East Asia",
}

REGION_COLORS = {
    "NorthAmerica": "#874F8D",
    "Europe":       "#1C6AB1",
    "EastAsia":     "#ED4043",
}
REGION_ORDER = ["NorthAmerica", "Europe", "EastAsia"]


# It is necessary to use 'npz' when adding 'high'. 
# This is because the initial selected latitude was relatively low, 
# and 'high' corresponds to generating data for high-latitude regions.
NPZ_PATH = OUTPUT_DIR / (
    f"fig4_SSW_cumColdArea_TanomP{int(PERCENTILE_THRESHOLD)}"
    f"_day{DAY_START}to{DAY_END}"
    f"_mem{N_MEMBERS}_{START_YEAR}_{END_YEAR}"
    f"_sliding{SLIDING_WINDOW}"
    f"_NDJFM_daily_{BASELINE_END}_high.npz"
)

PNG_PATH = str(OUTPUT_DIR / (
    f"fig4_SEAS5_SSW_cumColdArea_TanomP{int(PERCENTILE_THRESHOLD)}"
    f"_10yrsliding_NDJFM_daily_{BASELINE_END}_high.pdf"
))

# ================================================================
# basic tools
# ================================================================
def get_var_name(ds):
    for v in T2M_VAR_CANDIDATES:
        if v in ds.data_vars:
            return v
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise ValueError(f"Cannot identify T2m variable: {list(ds.data_vars)}")


def get_lat_lon_dims(da):
    lat = next(d for d in da.dims if "lat" in d.lower())
    lon = next(d for d in da.dims if "lon" in d.lower())
    return lat, lon


def get_time_dim(da):
    for n in ["valid_time", "time"]:
        if n in da.dims:
            return n
    raise ValueError(f"No time dim: dims={da.dims}")


def month_day_key(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.month:02d}-{ts.day:02d}"


def get_daily_filepath(year):
    return T2M_DAILY_DIR / f"SEAS5_2mt_NH_{year}11_system51_m25_daily.nc"


def read_year_data(year):
    """
    Read one yearly daily file.
    Return:
        arr shape = (N_MEMBERS, n_days, nlat, nlon), float32
        times, members, lat_vals, lon_vals
    """
    fp = get_daily_filepath(year)
    if not fp.exists():
        raise FileNotFoundError(f"Missing: {fp}")

    ds  = xr.open_dataset(fp)
    var = get_var_name(ds)
    da  = ds[var]

    for dim in list(da.dims):
        if (
            dim not in ("number",)
            and da.sizes[dim] == 1
            and "lat"   not in dim.lower()
            and "lon"   not in dim.lower()
            and "time"  not in dim.lower()
            and "valid" not in dim.lower()
        ):
            da = da.squeeze(dim, drop=True)

    da = da.transpose("number", *[d for d in da.dims if d != "number"])
    da = da.isel(number=slice(0, N_MEMBERS))

    lat_name, lon_name = get_lat_lon_dims(da)
    time_name          = get_time_dim(da)

    if da[lat_name].values[0] < da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})

    lon0 = da[lon_name].values
    if np.nanmax(lon0) > 180:
        lon_new  = np.where(lon0 > 180, lon0 - 360, lon0)
        sort_idx = np.argsort(lon_new)
        da       = da.isel({lon_name: sort_idx})
        da       = da.assign_coords({lon_name: lon_new[sort_idx]})
        del lon_new, sort_idx

    lat_vals = da[lat_name].values.copy()
    lon_vals = da[lon_name].values.copy()
    times    = pd.to_datetime(da[time_name].values).normalize()
    members  = da["number"].values.astype(int).tolist()

    arr = da.values.astype(np.float32)

    ds.close()
    del da, ds
    gc.collect()

    return arr, times, members, lat_vals, lon_vals


def scan_all_years():
    """
    Return:
        meta_u[year] = {
            md,
            member_to_idx,
            time_to_idx,
            lat_vals,
            lon_vals,
            n_times
        }
    """
    meta_u = {}

    for year in range(START_YEAR, END_YEAR + 1):
        fp = get_daily_filepath(year)
        if not fp.exists():
            print(f"  [WARN] Missing file for {year}, skip.")
            continue

        try:
            arr, times, members, lat_vals, lon_vals = read_year_data(year)
            meta_u[year] = {
                "lat_vals":      lat_vals,
                "lon_vals":      lon_vals,
                "md":            [month_day_key(t) for t in times],
                "member_to_idx": {m: i for i, m in enumerate(members)},
                "time_to_idx":   {t: i for i, t in enumerate(times)},
                "n_times":       len(times),
            }
            print(f"  {year} meta scanned: shape {arr.shape}")
            del arr, times, members, lat_vals, lon_vals
            gc.collect()

        except Exception as e:
            print(f"  [WARN] Scan failed {year}: {e}")
            gc.collect()

    return meta_u


def cosine_weights_2d(lat_vals, lon_vals, rb):
    lat_mask = (lat_vals >= rb["lat_min"]) & (lat_vals <= rb["lat_max"])
    lon_mask = (lon_vals >= rb["lon_min"]) & (lon_vals <= rb["lon_max"])
    w2d = np.cos(np.deg2rad(lat_vals[lat_mask]))[:, None] * np.ones(lon_mask.sum())
    return lat_mask, lon_mask, w2d


def region_area_fraction_below_threshold(field2d, threshold2d,
                                         lat_mask, lon_mask, w2d):
    """
    Calculate area fraction within region where field2d < threshold2d.

    Return:
        fraction in [0, 1]
    """
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    sub = field2d[np.ix_(lat_idx, lon_idx)]
    thr = threshold2d[np.ix_(lat_idx, lon_idx)]

    valid = np.isfinite(sub) & np.isfinite(thr)
    cold  = (sub < thr) & valid

    total = (valid * w2d).sum()
    if total == 0:
        return np.nan

    return float((cold * w2d).sum() / total)


def read_ssw_events():
    df = pd.read_csv(SSW_CSV_PATH)
    df["ssw_date"] = pd.to_datetime(df["ssw_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ssw_date"])

    df = df[(df["init_year"] >= START_YEAR) & (df["init_year"] <= END_YEAR)].copy()
    df["member"] = df["member"].astype(int)
    df = df[df["member"] < N_MEMBERS]

    df["month"] = df["ssw_date"].dt.month
    df = df[df["month"].isin([11, 12, 1, 2, 3])].copy()

    print(f"Loaded {len(df)} SSW events (NDJFM).")
    return df


# ================================================================
# baseline climatology
# ================================================================
def build_baseline_climatology(meta_u):
    """
    Build smoothed baseline climatology using ensemble mean.
    Same logic as your original code:
        baseline years = BASELINE_START-BASELINE_END
        daily climatology smoothed by ±5 days
    """
    print(f"Building rolling climatology ({BASELINE_START}-{BASELINE_END}, ±5 days)...")

    doy_to_fields = {}

    for year in range(BASELINE_START, BASELINE_END + 1):
        if year not in meta_u:
            continue

        arr, times, _, _, _ = read_year_data(year)
        ens_mean = arr.mean(axis=0)   # (T, lat, lon)
        del arr
        gc.collect()

        doys = pd.to_datetime(times).dayofyear.values

        for i, doy in enumerate(doys):
            doy_to_fields.setdefault(doy, []).append(
                ens_mean[i].astype(np.float32)
            )

        del ens_mean, times
        gc.collect()

    sample = next(iter(doy_to_fields.values()))[0]
    nlat, nlon = sample.shape

    clim_raw = np.full((366, nlat, nlon), np.nan, dtype=np.float32)

    for doy, fields in doy_to_fields.items():
        clim_raw[doy - 1] = np.nanmean(np.stack(fields, axis=0), axis=0)

    del doy_to_fields
    gc.collect()

    window = 11
    pad = window // 2

    clim_pad = np.concatenate([clim_raw[-pad:], clim_raw, clim_raw[:pad]], axis=0)
    clim_smooth = np.full_like(clim_raw, np.nan)

    for i in range(366):
        win = clim_pad[i:i + window]
        if not np.isfinite(win).any():
            continue
        clim_smooth[i] = np.nanmean(win, axis=0)

    del clim_pad, clim_raw
    gc.collect()

    baseline_clim = {}
    for t in pd.date_range("2001-01-01", "2001-12-31"):
        key = month_day_key(t)
        baseline_clim[key] = clim_smooth[t.dayofyear - 1]

    del clim_smooth
    gc.collect()

    print(f"  Done: {len(baseline_clim)} keys (smoothed)")
    return baseline_clim


def build_baseline_p10_threshold(meta_u, baseline_clim):
    """
    Build baseline 10th percentile threshold for Tanom.

    Important:
        This threshold is based on raw anomaly:
            Tanom = T - baseline_clim

        For memory safety, this function loops over each target calendar day
        and reads baseline-year files only as needed.

    Threshold:
        for each calendar day, use ±5-day window from all baseline years
        and all ensemble members.

    Return:
        baseline_p10[month-day] = 2D threshold field, same shape as T2m.
    """
    print(
        f"Building baseline P{PERCENTILE_THRESHOLD:.0f} threshold for Tanom "
        f"({BASELINE_START}-{BASELINE_END}, ±5 days, all members)..."
    )

    baseline_years = [y for y in range(BASELINE_START, BASELINE_END + 1) if y in meta_u]

    # If memory is not enough, set USE_BASELINE_CACHE = False.
    USE_BASELINE_CACHE = True

    baseline_data = {}
    if USE_BASELINE_CACHE:
        for y in baseline_years:
            try:
                arr, _, _, _, _ = read_year_data(y)
                baseline_data[y] = arr
                print(f"  Cached baseline year {y}: {arr.shape}")
            except Exception as e:
                print(f"  [WARN] baseline cache failed {y}: {e}")
            gc.collect()

    baseline_p10 = {}

    target_dates = pd.date_range("2001-01-01", "2001-12-31")

    for target in target_dates:
        key = month_day_key(target)
        fields = []

        window_mds = set()
        for dd in range(-5, 6):
            t = target + pd.Timedelta(days=dd)
            window_mds.add((t.month, t.day))

        for y in baseline_years:
            if USE_BASELINE_CACHE:
                if y not in baseline_data:
                    continue
                arr = baseline_data[y]
            else:
                try:
                    arr, _, _, _, _ = read_year_data(y)
                except Exception as e:
                    print(f"  [WARN] read baseline {y}: {e}")
                    continue

            info = meta_u[y]

            matched = [
                idx for date, idx in info["time_to_idx"].items()
                if (date.month, date.day) in window_mds
            ]

            for tidx in matched:
                md_key = info["md"][tidx]
                c_field = baseline_clim.get(md_key)
                if c_field is None:
                    continue

                # all members, anomaly
                anom = arr[:, tidx] - c_field[None, :, :]
                fields.append(anom.astype(np.float32))

                del anom

            if not USE_BASELINE_CACHE:
                del arr
                gc.collect()

        if len(fields) == 0:
            baseline_p10[key] = None
            print(f"  [WARN] no fields for P10 {key}")
            continue

        vals = np.concatenate(fields, axis=0).astype(np.float32)
        baseline_p10[key] = np.nanpercentile(
            vals, PERCENTILE_THRESHOLD, axis=0
        ).astype(np.float32)

        del vals, fields
        gc.collect()

        if target.day == 1:
            print(f"  P{PERCENTILE_THRESHOLD:.0f} threshold done to {key}")

    if USE_BASELINE_CACHE:
        del baseline_data
        gc.collect()

    print(f"  Done: {len(baseline_p10)} P10 threshold fields")
    return baseline_p10


# ================================================================
# trend coefficients
# ================================================================
def build_slope_and_intercept(meta_u, baseline_clim):
    """
    Build linear trend coefficients for ensemble-mean anomaly.
    """
    print("Building trend coefficients...")
    md_entries = {}

    for year in sorted(meta_u.keys()):
        arr, _, _, _, _ = read_year_data(year)
        ens_mean = arr.mean(axis=0)
        del arr
        gc.collect()

        for i, key in enumerate(meta_u[year]["md"]):
            cf = baseline_clim.get(key)
            if cf is None:
                continue

            anom = ens_mean[i] - cf
            md_entries.setdefault(key, []).append((year, anom.astype(np.float32)))
            del anom

        del ens_mean
        gc.collect()
        print(f"  Trend accumulate {year}")

    coeffs = {}

    for key, entries in md_entries.items():
        if len(entries) < 2:
            continue

        yrs  = np.array([e[0] for e in entries], dtype=np.float64)
        vals = np.stack([e[1] for e in entries]).astype(np.float64)

        yc = yrs - yrs.mean()
        denom = (yc ** 2).sum()

        if denom == 0:
            del yrs, vals, yc
            continue

        b = (yc[:, None, None] * vals).sum(axis=0) / denom
        a = np.nanmean(vals, axis=0)

        coeffs[key] = (
            a.astype(np.float32),
            b.astype(np.float32),
            float(yrs.mean())
        )

        del yrs, vals, yc, b, a

    del md_entries
    gc.collect()

    print(f"  Done: {len(coeffs)} calendar days")
    return coeffs


def compute_detrended_year(year, meta_u, baseline_clim, coeffs):
    """
    Return:
        arr: original absolute T2m, shape (member, time, lat, lon)
        det: detrended Tanom, shape (member, time, lat, lon)
    """
    arr, _, _, _, _ = read_year_data(year)
    det = np.empty_like(arr)

    for tidx, key in enumerate(meta_u[year]["md"]):
        c_field = baseline_clim.get(key)
        if c_field is None:
            det[:, tidx] = np.nan
            continue

        anom = arr[:, tidx] - c_field[None, :, :]

        if key in coeffs:
            a, b, yr_mean = coeffs[key]

            # # Keep your uploaded-code detrending:
            det[:, tidx] = anom - (a[None, :, :] + b[None, :, :] * (year - yr_mean))

            # Alternative: remove only slope, retain mean anomaly:
            # det[:, tidx] = anom - b[None, :, :] * (year - yr_mean)

        else:
            det[:, tidx] = anom

        del anom

    return arr, det


# ================================================================
# event metric
# ================================================================
def build_event_cache(ssw_df, meta_u, baseline_clim, baseline_p10, coeffs):
    """
    For each SSW event and each region:

        raw_sum = sum over day15-59 of:
            100 * area_fraction(raw_anom < baseline_p10)

        det_sum = sum over day15-59 of:
            100 * area_fraction(det_anom < baseline_p10)

    Unit:
        % area × day
    """
    print("Building event cache...")
    region_weights  = {}
    event_cache_raw = {}
    event_cache_det = {}

    first_year = sorted(meta_u.keys())[0]

    for rn, rb in REGION_BOXES.items():
        region_weights[rn] = cosine_weights_2d(
            meta_u[first_year]["lat_vals"],
            meta_u[first_year]["lon_vals"],
            rb
        )

    for year in sorted(meta_u.keys()):
        dfp = ssw_df[ssw_df["init_year"] == year]
        if dfp.empty:
            continue

        arr, det = compute_detrended_year(year, meta_u, baseline_clim, coeffs)
        info = meta_u[year]

        for _, row in dfp.iterrows():
            member = int(row["member"])
            center = pd.Timestamp(row["ssw_date"]).normalize()

            if member not in info["member_to_idx"]:
                continue
            if center not in info["time_to_idx"]:
                continue

            mi = info["member_to_idx"][member]
            ci = info["time_to_idx"][center]

            i0 = ci + DAY_START
            i1 = ci + DAY_END + 1

            if i0 < 0 or i1 > arr.shape[1]:
                print(f"  [SKIP] {year} mem={member} {center.date()} out of range")
                continue

            for rn in REGION_ORDER:
                lat_mask, lon_mask, w2d = region_weights[rn]

                raw_sum = 0.0
                det_sum = 0.0
                raw_valid_days = 0
                det_valid_days = 0

                for tidx in range(i0, i1):
                    md_key = info["md"][tidx]

                    c_field = baseline_clim.get(md_key)
                    p10_field = baseline_p10.get(md_key)

                    if c_field is None or p10_field is None:
                        continue

                    raw_anom = arr[mi, tidx] - c_field
                    det_anom = det[mi, tidx]

                    frac_raw = region_area_fraction_below_threshold(
                        raw_anom, p10_field,
                        lat_mask, lon_mask, w2d
                    )

                    frac_det = region_area_fraction_below_threshold(
                        det_anom, p10_field,
                        lat_mask, lon_mask, w2d
                    )

                    if np.isfinite(frac_raw):
                        raw_sum += frac_raw * 100.0
                        raw_valid_days += 1

                    if np.isfinite(frac_det):
                        det_sum += frac_det * 100.0
                        det_valid_days += 1

                    del raw_anom, det_anom

                key_ev = (year, member, center, rn)
                event_cache_raw[key_ev] = raw_sum if raw_valid_days > 0 else np.nan
                event_cache_det[key_ev] = det_sum if det_valid_days > 0 else np.nan

        del arr, det
        gc.collect()

        print(f"  Event cache {year} done")

    return event_cache_raw, event_cache_det


# ================================================================
# sliding aggregation + no-SSW climatology
# ================================================================
def aggregate_sliding_with_global_clim(ssw_df, meta_u, baseline_clim,
                                       baseline_p10, coeffs,
                                       event_cache_raw, event_cache_det):
    """
    Sliding-window aggregation.

    SSW:
        average cumulative cold-anomaly area over all SSW events in each window.

    No-SSW climatology:
        for each SSW event in the window, take the same calendar-day window
        from no-SSW year/member pairs within the same 10-year window.
        Each no-SSW analog contributes one cumulative value.
    """
    yr_min = int(ssw_df["init_year"].min())
    yr_max = int(ssw_df["init_year"].max())

    windows = [
        (y, y + SLIDING_WINDOW - 1)
        for y in range(yr_min, yr_max - SLIDING_WINDOW + 2)
    ]

    first_year = sorted(meta_u.keys())[0]

    region_weights = {}
    for rn, rb in REGION_BOXES.items():
        region_weights[rn] = cosine_weights_2d(
            meta_u[first_year]["lat_vals"],
            meta_u[first_year]["lon_vals"],
            rb
        )

    ssw_pairs = set(
        zip(ssw_df["init_year"].astype(int), ssw_df["member"].astype(int))
    )

    ts = {
        rn: {
            tag: {"xc": [], "mean": [], "se": [], "n": []}
            for tag in ["raw", "det", "clim_raw", "clim_det"]
        }
        for rn in REGION_ORDER
    }

    def _append_stats(lst, d):
        vals = np.array(lst, dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)

        d["n"].append(n)

        if n == 0:
            d["mean"].append(np.nan)
            d["se"].append(np.nan)
        elif n == 1:
            d["mean"].append(vals[0])
            d["se"].append(np.nan)
        else:
            d["mean"].append(np.nanmean(vals))
            d["se"].append(np.nanstd(vals, ddof=1) / np.sqrt(n))

    for ws, we in windows:
        dfp = ssw_df[ssw_df["init_year"].between(ws, we)]
        xc = ws + (SLIDING_WINDOW - 1) / 2.0

        no_ssw_in_window = [
            (year, m)
            for year in range(ws, we + 1)
            if year in meta_u
            for m in range(N_MEMBERS)
            if (year, m) not in ssw_pairs
        ]

        # ── diagnostic output ──────────────────────────────────
        no_ssw_years  = sorted(set(y for y, m in no_ssw_in_window))
        no_ssw_counts = {
            y: sum(1 for yy, mm in no_ssw_in_window if yy == y)
            for y in no_ssw_years
        }
        total_possible = sum(
            N_MEMBERS for year in range(ws, we + 1) if year in meta_u
        )
        print(f"\nWindow {ws}-{we}:")
        print(f"  SSW events in window  : {len(dfp)}")
        print(f"  No-SSW (year,member)  : {len(no_ssw_in_window)} / {total_possible} total slots")
        print(f"  No-SSW years present  : {no_ssw_years}")
        print(f"  No-SSW members/year   : {no_ssw_counts}")
        # ─────────────────────────────────────────────────

        # Load all years in this sliding window
        window_data = {}

        # Load all years in this sliding window
        window_data = {}

        for year in range(ws, we + 1):
            if year not in meta_u:
                continue

            try:
                arr, det = compute_detrended_year(year, meta_u, baseline_clim, coeffs)
                window_data[year] = (arr, det)
            except Exception as e:
                print(f"  [WARN] window data {year}: {e}")

            gc.collect()

        for rn in REGION_ORDER:
            lat_mask, lon_mask, w2d = region_weights[rn]

            # ----------------------------
            #  Post-SSW event values
            # ----------------------------
            rv_list = []
            dv_list = []

            for _, row in dfp.iterrows():
                year = int(row["init_year"])
                member = int(row["member"])
                ssw_date = pd.Timestamp(row["ssw_date"]).normalize()

                key_ev = (year, member, ssw_date, rn)

                if key_ev in event_cache_raw:
                    rv_list.append(event_cache_raw[key_ev])
                    dv_list.append(event_cache_det[key_ev])

            ts[rn]["raw"]["xc"].append(xc)
            ts[rn]["det"]["xc"].append(xc)
            _append_stats(rv_list, ts[rn]["raw"])
            _append_stats(dv_list, ts[rn]["det"])

            # ----------------------------
            # No-SSW analog values
            # ----------------------------
            clim_raw_list = []
            clim_det_list = []

            for _, row in dfp.iterrows():
                center = pd.Timestamp(row["ssw_date"]).normalize()

                # same post-event calendar-day window for this event
                event_mds = set()
                for d in range(DAY_START, DAY_END + 1):
                    date = center + pd.Timedelta(days=d)
                    event_mds.add((date.month, date.day))

                for yr_c, m_c in no_ssw_in_window:
                    if yr_c not in window_data:
                        continue

                    arr_c, det_c = window_data[yr_c]
                    info_c = meta_u[yr_c]

                    matched = [
                        idx for date, idx in info_c["time_to_idx"].items()
                        if (date.month, date.day) in event_mds
                    ]

                    if len(matched) == 0:
                        continue

                    raw_sum = 0.0
                    det_sum = 0.0
                    raw_valid_days = 0
                    det_valid_days = 0

                    for tidx in matched:
                        md_key = info_c["md"][tidx]

                        c_field = baseline_clim.get(md_key)
                        p10_field = baseline_p10.get(md_key)

                        if c_field is None or p10_field is None:
                            continue

                        raw_anom = arr_c[m_c, tidx] - c_field
                        det_anom = det_c[m_c, tidx]

                        frac_raw = region_area_fraction_below_threshold(
                            raw_anom, p10_field,
                            lat_mask, lon_mask, w2d
                        )

                        frac_det = region_area_fraction_below_threshold(
                            det_anom, p10_field,
                            lat_mask, lon_mask, w2d
                        )

                        if np.isfinite(frac_raw):
                            raw_sum += frac_raw * 100.0
                            raw_valid_days += 1

                        if np.isfinite(frac_det):
                            det_sum += frac_det * 100.0
                            det_valid_days += 1

                        del raw_anom, det_anom

                    if raw_valid_days > 0:
                        clim_raw_list.append(raw_sum)

                    if det_valid_days > 0:
                        clim_det_list.append(det_sum)

            ts[rn]["clim_raw"]["xc"].append(xc)
            ts[rn]["clim_det"]["xc"].append(xc)
            _append_stats(clim_raw_list, ts[rn]["clim_raw"])
            _append_stats(clim_det_list, ts[rn]["clim_det"])

        for year in list(window_data.keys()):
            del window_data[year]

        del window_data
        gc.collect()

        print(f"  Window {ws}-{we} done  (no-SSW pairs: {len(no_ssw_in_window)})")

    for rn in REGION_ORDER:
        for tag in ["raw", "det", "clim_raw", "clim_det"]:
            for k in ["xc", "mean", "se", "n"]:
                ts[rn][tag][k] = np.array(ts[rn][tag][k])

    return ts


# ================================================================
# cache
# ================================================================
def save_cache(path, ts):
    d = {}

    for rn in REGION_ORDER:
        for tag in ["raw", "det", "clim_raw", "clim_det"]:
            pfx = f"{rn}__{tag}"
            d[f"{pfx}__xc"]   = ts[rn][tag]["xc"]
            d[f"{pfx}__mean"] = ts[rn][tag]["mean"]
            d[f"{pfx}__se"]   = ts[rn][tag]["se"]
            d[f"{pfx}__n"]    = ts[rn][tag]["n"]

    np.savez(path, **d)
    print(f"Saved: {path}")


def load_cache(path):
    data = np.load(path, allow_pickle=True)

    ts = {
        rn: {
            tag: {}
            for tag in ["raw", "det", "clim_raw", "clim_det"]
        }
        for rn in REGION_ORDER
    }

    for rn in REGION_ORDER:
        for tag in ["raw", "det", "clim_raw", "clim_det"]:
            pfx = f"{rn}__{tag}"
            ts[rn][tag]["xc"]   = data[f"{pfx}__xc"]
            ts[rn][tag]["mean"] = data[f"{pfx}__mean"]
            ts[rn][tag]["se"]   = data[f"{pfx}__se"]
            ts[rn][tag]["n"]    = data[f"{pfx}__n"].astype(int)

    print(f"Loaded: {path}")
    return ts


def block_bootstrap_trend(xv, mv, n_boot=2000, block_size=7, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mv)

    if n < 3:
        return np.nan, np.nan

    obs_slope = theilslopes(mv, xv).slope

    boot_slopes = []

    for _ in range(n_boot):
        indices = []

        while len(indices) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            indices.extend(range(start, min(start + block_size, n)))

        idx = np.array(indices[:n])
        boot_slopes.append(theilslopes(mv[idx], xv).slope)

    boot_slopes = np.array(boot_slopes)
    boot_centered = boot_slopes - np.mean(boot_slopes)

    p = float(np.mean(np.abs(boot_centered) >= np.abs(obs_slope)))

    return obs_slope, p


# ================================================================
# plot
# ================================================================
def plot_results(ts):
    import matplotlib.lines as mlines
    # =====================================================
    # Convert cumulative (% area × day) to mean area (%)
    # Only for plotting; original ts / npz remain unchanged
    # =====================================================
    ts_plot = {}

    for rn in REGION_ORDER:
        ts_plot[rn] = {}

        for tag in ["raw", "det", "clim_raw", "clim_det"]:
            ts_plot[rn][tag] = {
                "xc":   ts[rn][tag]["xc"].copy(),
                "mean": ts[rn][tag]["mean"].copy() / N_DAYS,
                "se":   ts[rn][tag]["se"].copy() / N_DAYS,
                "n":    ts[rn][tag]["n"].copy(),
            }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    panel_info = [
        ("raw", "clim_raw", "Raw Tanom"),
        ("det", "clim_det", "Detrended Tanom"),
    ]

    panel_labels = ["(a)", "(b)"]

    for ci, (tag, clim_tag, title) in enumerate(panel_info):
        ax = axes[ci]
        trend_lines = {rn: {} for rn in REGION_ORDER}

        # ----------------------------
        # no-SSW climatology, dashed
        # ----------------------------
        for rn in REGION_ORDER:
            col = REGION_COLORS[rn]

            xc = ts_plot[rn][clim_tag]["xc"]
            mean = ts_plot[rn][clim_tag]["mean"]
            se = ts_plot[rn][clim_tag]["se"]

            valid = np.isfinite(mean)
            xv, mv, sv = xc[valid], mean[valid], se[valid]

            if len(xv) == 0:
                continue

            ax.plot(xv, mv, color=col, lw=1.6, ls="--", alpha=0.65, zorder=1)

            good_se = np.isfinite(sv)
            if good_se.sum() > 0:
                ax.fill_between(
                    xv[good_se],
                    (mv - sv)[good_se],
                    (mv + sv)[good_se],
                    color=col,
                    alpha=0.08,
                    zorder=1,
                )

            if len(xv) >= 3:
                ts_res_c = theilslopes(mv, xv, method="joint")
                _, p_val_c = block_bootstrap_trend(
                    xv, mv, n_boot=2000, block_size=7
                )

                sig_c = "**" if p_val_c < 0.01 else ("*" if p_val_c < 0.05 else "")

                ax.plot(
                    xv,
                    ts_res_c.intercept + ts_res_c.slope * xv,
                    color=col,
                    lw=1.2,
                    ls=":",
                    alpha=0.75,
                    zorder=4,
                )

                trend_lines[rn]["clim"] = (ts_res_c.slope, sig_c)

        # ----------------------------
        # post-SSW, solid
        # ----------------------------
        for rn in REGION_ORDER:
            col = REGION_COLORS[rn]

            xc = ts_plot[rn][tag]["xc"]
            mean = ts_plot[rn][tag]["mean"]
            se = ts_plot[rn][tag]["se"]

            valid = np.isfinite(mean)
            xv, mv, sv = xc[valid], mean[valid], se[valid]

            if len(xv) == 0:
                continue

            good_se = np.isfinite(sv)
            if good_se.sum() > 0:
                ax.fill_between(
                    xv[good_se],
                    (mv - sv)[good_se],
                    (mv + sv)[good_se],
                    color=col,
                    alpha=0.22,
                    zorder=2,
                )

            ax.plot(
                xv,
                mv,
                color=col,
                lw=2.4,
                alpha=0.92,
                zorder=3,
                label=REGION_LABELS[rn],
            )

            if len(xv) >= 3:
                ts_res_s = theilslopes(mv, xv, method="joint")
                _, p_val_s = block_bootstrap_trend(
                    xv, mv, n_boot=2000, block_size=7
                )

                sig_s = "**" if p_val_s < 0.01 else ("*" if p_val_s < 0.05 else "")

                ax.plot(
                    xv,
                    ts_res_s.intercept + ts_res_s.slope * xv,
                    color=col,
                    lw=1.2,
                    ls=":",
                    alpha=0.75,
                    zorder=4,
                )

                trend_lines[rn]["ssw"] = (ts_res_s.slope, sig_s)

        # ----------------------------
        # trend text box
        # ----------------------------
        short_name = {
            "NorthAmerica": "N. America",
            "Europe": "Europe",
            "EastAsia": "East Asia",
        }

        lines_text = []
        lines_color = []
        lines_fw = []

        lines_text.append("Theil-Sen slope\n(% decade$^{-1}$)")
        lines_color.append("#888888")
        lines_fw.append("normal")

        lines_text.append(("Region", "SSW", "Non-SSW"))
        lines_color.append("#444444")
        lines_fw.append("bold")

        for rn in REGION_ORDER:
            col = REGION_COLORS[rn]
            sn = short_name[rn]

            ssw_s, ssw_sig = trend_lines[rn].get("ssw", (np.nan, ""))
            clim_s, clim_sig = trend_lines[rn].get("clim", (np.nan, ""))

            ssw_str = f"{ssw_s* 10:+.2f}{ssw_sig}" if np.isfinite(ssw_s) else "—"
            clim_str = f"{clim_s* 10:+.2f}{clim_sig}" if np.isfinite(clim_s) else "—"

            lines_text.append((sn, ssw_str, clim_str))
            lines_color.append(col)
            lines_fw.append("normal")

        fs_title = 9.0
        fs_data = 10.0
        x_left = 0.03
        y_bot = 0.03
        line_h = 0.068
        n_lines = len(lines_text)

        col_x = [
            x_left + 0.01,
            x_left + 0.24,
            x_left + 0.37,
        ]

        box_w = 0.40
        box_h = line_h * n_lines + 0.06

        ax.add_patch(
            plt.Rectangle(
                (x_left - 0.015, y_bot - 0.015),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="#aaaaaa",
                lw=1.0,
                alpha=0.93,
                zorder=8,
                clip_on=False,
            )
        )

        for i, (txt, col, fw) in enumerate(zip(lines_text, lines_color, lines_fw)):
            ypos = y_bot + (n_lines - 1 - i) * line_h + 0.02
            fs = fs_title if i == 0 else fs_data

            if i == 0:
                ax.text(
                    x_left + box_w / 2 - 0.015,
                    ypos,
                    txt,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=fs,
                    fontweight=fw,
                    color=col,
                    zorder=9,
                )
            else:
                region_txt, ssw_txt, clim_txt = txt

                ax.text(
                    col_x[0],
                    ypos,
                    region_txt,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=fs,
                    fontweight=fw,
                    color=col,
                    zorder=9,
                )

                ax.text(
                    col_x[1],
                    ypos,
                    ssw_txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=fs,
                    fontweight=fw,
                    color=col,
                    zorder=9,
                )

                ax.text(
                    col_x[2],
                    ypos,
                    clim_txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=fs,
                    fontweight=fw,
                    color=col,
                    zorder=9,
                )

        # ----------------------------
        # axes
        # ----------------------------
        ax.axhline(0, color="black", lw=0.8, alpha=0.40)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("10-year sliding window", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", lw=0.6, alpha=0.30)

        if ci == 0:
            ax.set_ylabel(
                f"Mean cold-anomaly area (%)"
            )
        else:
            ax.set_ylabel("")

        # dynamic y-limits
        all_vals = []
        for rn in REGION_ORDER:
            for ttag in [tag, clim_tag]:
                v = ts_plot[rn][ttag]["mean"]
                all_vals.extend(v[np.isfinite(v)].tolist())

        if len(all_vals) > 0:
            ymax = np.nanmax(all_vals)
            ax.set_ylim(0, ymax * 1.20 if ymax > 0 else 1)

        xc_ref = ts[REGION_ORDER[0]][tag]["xc"]
        mean_ref = ts[REGION_ORDER[0]][tag]["mean"]
        valid_ref = np.isfinite(mean_ref)

        if valid_ref.sum() > 0:
            xc_use = xc_ref[valid_ref]
            first_start = int(round(xc_use[0] - (SLIDING_WINDOW - 1) / 2.0))
            last_start = int(round(xc_use[-1] - (SLIDING_WINDOW - 1) / 2.0))

            window_starts = np.arange(first_start, last_start + 1, 5)
            xticks = window_starts + (SLIDING_WINDOW - 1) / 2.0
            xlabels = [f"{ys}–{ys + SLIDING_WINDOW - 1}" for ys in window_starts]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)

            ax.set_xlim(
                xc_ref[valid_ref][0] - 0.5,
                xc_ref[valid_ref][-1] + 0.5,
            )

        if ci == 0:
            style_handles = [
                mlines.Line2D([], [], color="#555", lw=2.4, ls="-", label="Post-SSW"),
                mlines.Line2D([], [], color="#555", lw=1.6, ls="--", label="Non-SSW"),
                mlines.Line2D([], [], color="#555", lw=1.2, ls=":", label="Theil-Sen trend"),
            ]

            region_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=REGION_COLORS[rn],
                    lw=2.4,
                    label=REGION_LABELS[rn],
                )
                for rn in REGION_ORDER
            ]

            ax.legend(
                handles=style_handles + region_handles,
                fontsize=8.5,
                loc="upper right",
                framealpha=0.92,
                edgecolor="#cccccc",
                frameon=True,
                ncol=1,
                handlelength=2.2,
                labelspacing=0.45,
            )

        ax.text(
            0.02,
            0.98,
            panel_labels[ci],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.5),
            zorder=10,
        )

    plt.suptitle(
        f"post-SSW mean cold-anomaly area ",
        fontsize=14,
        fontweight="bold",
        y=1,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {PNG_PATH}")


# ================================================================
# main
# ================================================================
def main():
    ssw_df = read_ssw_events()

    if NPZ_PATH.exists():
        print("Cache found, loading...")
        ts = load_cache(NPZ_PATH)
    else:
        print("Scanning year metadata...")
        meta_u = scan_all_years()

        baseline_clim = build_baseline_climatology(meta_u)

        baseline_p10 = build_baseline_p10_threshold(
            meta_u,
            baseline_clim
        )

        coeffs = build_slope_and_intercept(
            meta_u,
            baseline_clim
        )

        event_cache_raw, event_cache_det = build_event_cache(
            ssw_df,
            meta_u,
            baseline_clim,
            baseline_p10,
            coeffs
        )

        ts = aggregate_sliding_with_global_clim(
            ssw_df,
            meta_u,
            baseline_clim,
            baseline_p10,
            coeffs,
            event_cache_raw,
            event_cache_det
        )

        save_cache(NPZ_PATH, ts)
    
        del baseline_clim, baseline_p10, coeffs
        del event_cache_raw, event_cache_det, meta_u
        gc.collect()

    plot_results(ts)
    print("Done.")
    


if __name__ == "__main__":
    main()
