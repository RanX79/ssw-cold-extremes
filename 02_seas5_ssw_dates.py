"""
Identify stratospheric sudden warming (SSW) dates for the first
25 members of the SEAS5 hindcast.

The script identifies reversals of the 10-hPa, 60°N zonal-mean zonal
wind during November–March, excludes final warmings, and writes
member-wise SSW dates to CSV files.

Input:
    One daily SEAS5 zonal-wind NetCDF file per initialization year.
    Raw SEAS5 data are not distributed with this repository.

Output:
    CSV files containing all member-wise events and the retained
    NDJFM SSW events, written to OUTPUT_DIR.

Before running:
    Update U_DAILY_DIR to your directory containing the SEAS5 daily
    zonal-wind files.

Run:
    python 02_seas5_ssw_dates.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------
PRESSURE_LEVEL = 10
U_DAILY_DIR = Path("path/to/SEAS5/u10_daily")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1981
END_YEAR = 2024
LATITUDE = 60
N_MEMBERS = 25
MIN_WESTERLY_DAYS = 20
MIN_RECOVERY_DAYS = 10
DETECTION_MONTHS = (11, 12, 1, 2, 3)
TARGET_MONTHS = (11, 12, 1, 2, 3)

PRESSURE_LABEL = f"{PRESSURE_LEVEL}hPa"

# ================================================================
# helpers
# ================================================================
def get_lat_lon_dim_names(da):
    lat_c = [d for d in da.dims if "lat" in d.lower()]
    lon_c = [d for d in da.dims if "lon" in d.lower()]
    if not lat_c: raise ValueError(f"No lat dim: {da.dims}")
    if not lon_c: raise ValueError(f"No lon dim: {da.dims}")
    return lat_c[0], lon_c[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time", "forecast_period"]:
        if name in da.dims:
            return name
    raise ValueError(f"No time dim: {da.dims}")


def to_winter_year_label(init_year):
    return f"{init_year}-{init_year + 1}"


def safe_member_label(number_coord, idx):
    try:    return int(number_coord[idx])
    except: return idx + 1


def calc_max_westerly_streak(series):
    """计算序列中连续西风（U >= 0）的最大天数。"""
    max_streak = current_streak = 0
    for v in series:
        if pd.notna(v) and v >= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def is_final_warming(u_series, event_date, min_recovery_days=10):
    """
    判断一个东风反转事件是否为 Final Warming。
    规则：若在 event_date 之后、当年/次年 4 月 30 日之前，
          西风（U >= 0）连续天数不足 min_recovery_days，则为 Final Warming。
    返回 True = Final Warming（应排除），False = 真正的 SSW（保留）。
    """
    # 确定 4 月 30 日截止日期
    # 12月或11月发生的事件 → 截止到次年4月30日
    if event_date.month >= 10:
        deadline = pd.Timestamp(f"{event_date.year + 1}-04-30")
    else:
        deadline = pd.Timestamp(f"{event_date.year}-04-30")

    # 取 event_date 之后到 deadline 的 U 序列
    after = u_series.loc[
        (u_series.index > event_date) &
        (u_series.index <= deadline)
    ]

    if len(after) == 0:
        # 数据不足 deadline，保守认为是 Final Warming
        return True

    max_streak = calc_max_westerly_streak(after)
    return max_streak < min_recovery_days


# ================================================================
# 读取单年 daily 文件，提取 60N 纬向平均 U
# ================================================================
def preprocess_one_year(init_year, latitude=60, n_members=25):
    fp = U_DAILY_DIR / f"SEAS5_u{PRESSURE_LEVEL}hPa_NH_{init_year}11_system51_m25_daily.nc"

    if not fp.exists():
        raise FileNotFoundError(f"Missing: {fp.name}")

    ds  = xr.open_dataset(fp)
    var = list(ds.data_vars)[0]
    da  = ds[var].load()
    ds.close()

    if "number" not in da.dims:
        raise ValueError(f"Missing number dim: {da.dims}")
    da = da.transpose("number", *[d for d in da.dims if d != "number"])
    da = da.isel(number=slice(0, n_members))

    time_name          = get_time_dim_name(da)
    lat_name, lon_name = get_lat_lon_dim_names(da)

    da_60n = da.sel({lat_name: latitude}, method="nearest")
    da_zm  = da_60n.mean(dim=lon_name)

    # 只保留检测月份
    times      = pd.to_datetime(da_zm[time_name].values)
    month_mask = np.isin(times.month, DETECTION_MONTHS)
    da_zm      = da_zm.sel({time_name: times[month_mask]}).sortby(time_name)

    return da_zm, time_name, fp.name


# ================================================================
# SSW 检测（含 Final Warming 过滤）
# ================================================================
def detect_ssw_events_for_one_member(u_series, min_westerly_days=20,
                                     min_recovery_days=10):
    """
    检测 SSW 事件并排除 Final Warming。

    Final Warming 判据（Charlton & Polvani 2007）：
      若东风反转后，在 4 月 30 日前西风恢复连续天数 < min_recovery_days，
      则该事件为 Final Warming，不计入 SSW。
    """
    u_series = u_series.dropna().sort_index()
    if len(u_series) < 2:
        return []

    # 找所有从西风转为东风的日期（首次穿越零线）
    reversal_dates = [
        u_series.index[i]
        for i in range(1, len(u_series))
        if u_series.iloc[i] < 0 and u_series.iloc[i - 1] >= 0
        and u_series.index[i].month in DETECTION_MONTHS
    ]

    if not reversal_dates:
        return []

    raw_events = []
    for event_date in reversal_dates:
        # 计算东风持续到何时
        end_date  = event_date
        start_idx = u_series.index.get_loc(event_date)
        for i in range(start_idx + 1, len(u_series)):
            if u_series.iloc[i] >= 0:
                end_date = u_series.index[i - 1]
                break
            end_date = u_series.index[i]

        event_u = u_series.loc[event_date:end_date]

        # ── Final Warming 过滤 ──────────────────────────────────────
        if is_final_warming(u_series, event_date, min_recovery_days):
            continue   # 排除，不加入 raw_events
        # ────────────────────────────────────────────────────────────

        raw_events.append({
            "ssw_date":      event_date,
            "end_date":      end_date,
            "duration_days": (end_date - event_date).days + 1,
            "min_u":         float(event_u.min()),
            "min_u_date":    event_u.idxmin()
        })

    if not raw_events:
        return []

    # 20 天间隔规则：两次独立 SSW 之间须有 ≥ MIN_WESTERLY_DAYS 天西风
    filtered = [raw_events[0]]
    for event in raw_events[1:]:
        prev    = filtered[-1]
        between = u_series[
            (u_series.index > prev["end_date"]) &
            (u_series.index < event["ssw_date"])
        ]
        if len(between) > 0 and calc_max_westerly_streak(between) >= min_westerly_days:
            filtered.append(event)

    return filtered


# ================================================================
# 主流程
# ================================================================
memberwise_records = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n处理 {year} ...")
    try:
        da_zm, time_name, fname = preprocess_one_year(year, LATITUDE, N_MEMBERS)
        times       = pd.to_datetime(da_zm[time_name].values)
        winter_year = to_winter_year_label(year)

        print(f"  文件: {fname}")
        print(f"  Daily 时间范围: {times[0].date()} → {times[-1].date()}")

        n_with_ssw = 0
        n_total    = 0

        for m in range(da_zm.sizes["number"]):
            u_vals   = da_zm.isel(number=m).values
            u_series = pd.Series(u_vals, index=times)
            events   = detect_ssw_events_for_one_member(
                u_series,
                min_westerly_days=MIN_WESTERLY_DAYS,
                min_recovery_days=MIN_RECOVERY_DAYS
            )
            events    = [ev for ev in events if ev["ssw_date"].month in TARGET_MONTHS]
            member_id = safe_member_label(da_zm["number"].values, m)

            if not events:
                memberwise_records.append({
                    "init_year": year, "winter_year": winter_year,
                    "member": member_id, "event_index": np.nan,
                    "has_ssw_NDJFM": False, "ssw_date": pd.NaT,
                    "ssw_year": np.nan, "ssw_month": np.nan, "ssw_day": np.nan,
                    "end_date": pd.NaT, "duration_days": np.nan,
                    "min_u": np.nan, "min_u_date": pd.NaT, "u_at_ssw": np.nan
                })
            else:
                n_with_ssw += 1
                n_total    += len(events)
                for ie, ev in enumerate(events, start=1):
                    memberwise_records.append({
                        "init_year":     year,
                        "winter_year":   winter_year,
                        "member":        member_id,
                        "event_index":   ie,
                        "has_ssw_NDJFM": True,
                        "ssw_date":      ev["ssw_date"],
                        "ssw_year":      ev["ssw_date"].year,
                        "ssw_month":     ev["ssw_date"].month,
                        "ssw_day":       ev["ssw_date"].day,
                        "end_date":      ev["end_date"],
                        "duration_days": ev["duration_days"],
                        "min_u":         ev["min_u"],
                        "min_u_date":    ev["min_u_date"],
                        "u_at_ssw":      float(u_series.loc[ev["ssw_date"]])
                    })

        print(f"  {n_with_ssw}/{da_zm.sizes['number']} 个成员有 SSW，共 {n_total} 个事件")

    except Exception as e:
        print(f"  失败: year={year} | {e}")


# ================================================================
# 输出结果
# ================================================================
print("\n" + "=" * 80)
memberwise_df = pd.DataFrame(memberwise_records)
for c in ["ssw_date", "end_date", "min_u_date"]:
    memberwise_df[c] = pd.to_datetime(memberwise_df[c], errors="coerce")
memberwise_df = memberwise_df.sort_values(
    ["init_year", "member", "ssw_date"]
).reset_index(drop=True)

# 全部记录
out_all = OUTPUT_DIR / f"SEAS5_first25members_SSW_dates_NDJFM_{START_YEAR}_{END_YEAR}.csv"
memberwise_df.to_csv(out_all, index=False, encoding="utf-8-sig")
print(f"完整结果: {out_all}")

# 仅事件
events_df  = memberwise_df[memberwise_df["has_ssw_NDJFM"]].copy()
out_events = OUTPUT_DIR / f"SEAS5_first25members_SSW_dates_NDJFM_events_only_{START_YEAR}_{END_YEAR}.csv"
events_df.to_csv(out_events, index=False, encoding="utf-8-sig")
print(f"仅事件:   {out_events}")

# txt 摘要
txt_file = OUTPUT_DIR / f"SEAS5_first25members_SSW_dates_NDJFM_{START_YEAR}_{END_YEAR}.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SEAS5 first 25 members SSW dates (NDJFM)\n")
    f.write(f"Period: {START_YEAR}-{END_YEAR}  |  {PRESSURE_LABEL}  |  {LATITUDE}N\n")
    f.write(f"Min westerly separation: {MIN_WESTERLY_DAYS} days\n")
    f.write(f"Final Warming filter: recovery < {MIN_RECOVERY_DAYS} consec. westerly days"
            f" before Apr 30 → excluded\n")
    f.write("=" * 80 + "\n\n")
    for year in range(START_YEAR, END_YEAR + 1):
        sub = events_df[events_df["init_year"] == year].sort_values(
            ["ssw_date", "member"]
        )
        f.write(f"{to_winter_year_label(year)}\n" + "-" * 60 + "\n")
        if sub.empty:
            f.write("No NDJFM SSW detected.\n\n")
        else:
            for _, row in sub.iterrows():
                f.write(
                    f"member={int(row['member']):02d}, event={int(row['event_index'])}, "
                    f"ssw_date={row['ssw_date'].strftime('%Y-%m-%d')}, "
                    f"duration={int(row['duration_days'])} days, "
                    f"min_u={row['min_u']:.2f} m/s\n"
                )
            f.write("\n")
print(f"txt 摘要: {txt_file}")
print("\n完成！")
