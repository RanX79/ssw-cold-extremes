# -*- coding: utf-8 -*-
"""
Identify Northern Hemisphere stratospheric sudden warming (SSW) dates
from ERA5 daily zonal wind data.

The script applies the standard 10-hPa, 60°N zonal-mean zonal-wind
reversal criterion during November–March. Events that satisfy the
final-warming criterion are excluded.

Input:
    Daily ERA5 zonal wind data containing pressure-level, latitude,
    longitude, and time dimensions. Raw ERA5 data are not distributed
    with this repository.

Output:
    CSV files containing all detected events and the retained NDJFM SSW
    events, written to OUTPUT_DIR.

Before running:
    Update ERA5_U_FILE to the location of your downloaded ERA5 file.

Run:
    python 01_era5_ssw_dates.py
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
ERA5_U_FILE = Path("path/to/ERA5/u_daily_1940_2025_10hPa_no_feb29.nc")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1940
END_YEAR = 2024
LATITUDE = 60
TARGET_LEVEL = 10
MIN_WESTERLY_DAYS = 20
MIN_RECOVERY_DAYS = 10
DETECTION_MONTHS = (11, 12, 1, 2, 3)
TARGET_MONTHS = (11, 12, 1, 2, 3)

PRESSURE_LABEL = f"{TARGET_LEVEL}hPa"

# ===============================
# --- 工具函数 ---
# ===============================
def get_lat_lon_dim_names(da):
    lat_candidates = [d for d in da.dims if "lat" in d.lower()]
    lon_candidates = [d for d in da.dims if "lon" in d.lower()]
    if not lat_candidates:
        raise ValueError(f"未找到纬度维，当前 dims = {da.dims}")
    if not lon_candidates:
        raise ValueError(f"未找到经度维，当前 dims = {da.dims}")
    return lat_candidates[0], lon_candidates[0]


def get_time_dim_name(da):
    for name in ["valid_time", "time"]:
        if name in da.dims:
            return name
    raise ValueError(f"未找到时间维(valid_time/time)，当前 dims = {da.dims}")


def get_level_dim_name(da):
    for name in ["pressure_level", "level", "plev", "lev"]:
        if name in da.dims:
            return name
    raise ValueError(f"未找到气压层维，当前 dims = {da.dims}")


def to_winter_year_label(init_year):
    return f"{init_year}-{init_year + 1}"


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


def is_final_warming(u_series_full, event_date, min_recovery_days=10):
    """
    判断一个东风反转事件是否为 Final Warming。

    规则（Charlton & Polvani 2007）：
      若在 event_date 之后、当年/次年 4 月 30 日之前，
      西风（U >= 0）的最长连续天数 < min_recovery_days，
      则认定为 Final Warming，应排除。

    参数
    ----
    u_series_full : pd.Series
        完整时段的 U 序列（不限于当前 winter，需覆盖到 4 月底）。
    event_date : pd.Timestamp
        东风反转日期。
    min_recovery_days : int
        西风恢复所需最少连续天数，默认 10。

    返回
    ----
    True  → Final Warming（排除）
    False → 真正的 SSW（保留）
    """
    # 11–12 月发生的事件，截止到次年 4 月 30 日
    if event_date.month >= 10:
        deadline = pd.Timestamp(f"{event_date.year + 1}-04-30")
    else:
        deadline = pd.Timestamp(f"{event_date.year}-04-30")

    after = u_series_full.loc[
        (u_series_full.index > event_date) &
        (u_series_full.index <= deadline)
    ]

    if len(after) == 0:
        # 数据不足 deadline，保守视为 Final Warming
        return True

    return calc_max_westerly_streak(after) < min_recovery_days


def inspect_dataset_structure(ds):
    print("\n[DEBUG] 数据集变量结构：")
    for name, da in ds.data_vars.items():
        print(f"  data_var: {name}, dims={da.dims}, shape={da.shape}")
    for name, da in ds.coords.items():
        print(f"  coord   : {name}, dims={da.dims}, shape={da.shape}")


def choose_u_variable(ds):
    if "u" in ds.data_vars:
        return "u"
    candidate_names = []
    for name, da in ds.data_vars.items():
        dims_lower = [d.lower() for d in da.dims]
        has_time  = any(d in ["time", "valid_time"] for d in dims_lower)
        has_level = any(d in ["level", "lev", "pressure_level", "plev"] for d in dims_lower)
        has_lat   = any("lat" in d for d in dims_lower)
        has_lon   = any("lon" in d for d in dims_lower)
        if has_time and has_level and has_lat and has_lon:
            candidate_names.append(name)
    if len(candidate_names) == 1:
        return candidate_names[0]
    if len(candidate_names) > 1:
        print(f"[WARN] 找到多个疑似主变量: {candidate_names}，默认使用第一个。")
        return candidate_names[0]
    inspect_dataset_structure(ds)
    raise ValueError("未找到符合(time, level/lev, lat, lon)结构的 U 风主变量。")


def open_u_dataarray(file_path):
    ds       = xr.open_dataset(file_path)
    var_name = choose_u_variable(ds)
    da       = ds[var_name]
    print(f"[INFO] 使用变量: {var_name}")
    print(f"[INFO] 变量维度: {da.dims}")
    return ds, da


def select_pressure_level(da, target_level=10):
    level_name          = get_level_dim_name(da)
    level_values        = da[level_name].values
    level_values_numeric = pd.to_numeric(pd.Index(level_values), errors="coerce")
    if np.any(level_values_numeric == target_level):
        matched_value = level_values[np.where(level_values_numeric == target_level)[0][0]]
        da = da.sel({level_name: matched_value})
    else:
        raise ValueError(
            f"文件中未找到目标层 {target_level} hPa。"
            f" 可用层次示例: {list(level_values[:min(10, len(level_values))])}"
        )
    return da


def preprocess_era5_u(file_path, latitude=60, target_level=10):
    ds, da    = open_u_dataarray(file_path)
    da        = select_pressure_level(da, target_level=target_level)
    lat_name, lon_name = get_lat_lon_dim_names(da)
    time_name = get_time_dim_name(da)
    da_60n    = da.sel({lat_name: latitude}, method="nearest")
    da_zm     = da_60n.mean(dim=lon_name)
    times     = pd.to_datetime(da_zm[time_name].values)
    da_zm     = da_zm.sel({time_name: times}).sortby(time_name)
    ds.close()
    return da_zm, time_name


def extract_one_winter(u_series_all, init_year):
    """提取 init_year 年 11 月至 init_year+1 年 3 月的时间序列。"""
    start_date = pd.Timestamp(f"{init_year}-11-01")
    end_date   = pd.Timestamp(f"{init_year + 1}-03-31")
    sub = u_series_all[
        (u_series_all.index >= start_date) &
        (u_series_all.index <= end_date)
    ].copy()
    return sub.dropna().sort_index()


# ===============================
# --- SSW 检测（含 Final Warming 过滤）---
# ===============================
def detect_ssw_events_for_one_winter(u_series_winter, u_series_full,
                                     min_westerly_days=20,
                                     min_recovery_days=10):
    """
    检测单个 winter 内的所有独立 SSW 事件，并排除 Final Warming。

    参数
    ----
    u_series_winter : pd.Series
        当前 winter（11月–3月）的 U 序列。
    u_series_full : pd.Series
        完整时段的 U 序列，用于 Final Warming 判断时查找 4 月底的数据。
    min_westerly_days : int
        两次独立 SSW 事件间所需最少连续西风天数（默认 20）。
    min_recovery_days : int
        Final Warming 判据：4 月 30 日前西风恢复需 ≥ 此值（默认 10）。
    """
    u_series_winter = u_series_winter.dropna().sort_index()
    if len(u_series_winter) < 2:
        return []

    # 找所有西风→东风穿越日
    reversal_dates = [
        u_series_winter.index[i]
        for i in range(1, len(u_series_winter))
        if u_series_winter.iloc[i] < 0
        and u_series_winter.iloc[i - 1] >= 0
        and u_series_winter.index[i].month in DETECTION_MONTHS
    ]
    if not reversal_dates:
        return []

    raw_events = []
    for event_date in reversal_dates:

        # ── Final Warming 过滤 ─────────────────────────────────────
        if is_final_warming(u_series_full, event_date, min_recovery_days):
            continue   # 排除，不加入候选列表
        # ──────────────────────────────────────────────────────────

        # 计算东风持续结束日
        end_date  = event_date
        start_idx = u_series_winter.index.get_loc(event_date)
        for i in range(start_idx + 1, len(u_series_winter)):
            if u_series_winter.iloc[i] >= 0:
                end_date = u_series_winter.index[i - 1]
                break
            end_date = u_series_winter.index[i]

        event_u = u_series_winter.loc[event_date:end_date]
        raw_events.append({
            "ssw_date":      event_date,
            "end_date":      end_date,
            "duration_days": int((end_date - event_date).days + 1),
            "min_u":         float(event_u.min()),
            "min_u_date":    event_u.idxmin()
        })

    if not raw_events:
        return []

    # 20 天间隔规则
    filtered_events = [raw_events[0]]
    for event in raw_events[1:]:
        prev    = filtered_events[-1]
        between = u_series_winter[
            (u_series_winter.index > prev["end_date"]) &
            (u_series_winter.index < event["ssw_date"])
        ]
        if len(between) > 0 and calc_max_westerly_streak(between) >= min_westerly_days:
            filtered_events.append(event)

    return filtered_events


# ===============================
# --- 预处理全时段 ERA5 ---
# ===============================
print("\n" + "=" * 80)
print("读取 ERA5 数据...")
print("=" * 80)

da_zm, time_name = preprocess_era5_u(
    file_path=U_FILE,
    latitude=LATITUDE,
    target_level=TARGET_LEVEL
)

times_all = pd.to_datetime(da_zm[time_name].values)
u_all     = pd.Series(da_zm.values, index=times_all).sort_index()

print(f"文件: {U_FILE.name}")
print(f"时间范围: {times_all[0]} 到 {times_all[-1]}")
print(f"总天数: {len(times_all)}")


# ===============================
# --- 主流程 ---
# ===============================
winter_records = []

for year in range(START_YEAR, END_YEAR):
    print(f"\n处理 winter {year}-{year + 1} ...")

    winter_year    = to_winter_year_label(year)
    u_series_winter = extract_one_winter(u_all, year)

    if len(u_series_winter) == 0:
        print("  该冬季无数据，跳过。")
        continue

    print(f"  时间范围: {u_series_winter.index[0]} 到 {u_series_winter.index[-1]}")
    print(f"  样本长度: {len(u_series_winter)}")

    events = detect_ssw_events_for_one_winter(
        u_series_winter,
        u_series_full=u_all,            # ← 传入完整序列用于 Final Warming 判断
        min_westerly_days=MIN_WESTERLY_DAYS,
        min_recovery_days=MIN_RECOVERY_DAYS
    )

    events = [ev for ev in events if ev["ssw_date"].month in TARGET_MONTHS]

    if len(events) == 0:
        winter_records.append({
            "init_year": year, "winter_year": winter_year,
            "event_index": np.nan, "has_ssw_NDJFM": False,
            "ssw_date": pd.NaT, "ssw_year": np.nan,
            "ssw_month": np.nan, "ssw_day": np.nan,
            "end_date": pd.NaT, "duration_days": np.nan,
            "min_u": np.nan, "min_u_date": pd.NaT, "u_at_ssw": np.nan
        })
        print("  检测结果: 该冬季未发生 SSW")
    else:
        print(f"  检测结果: 该冬季发生 {len(events)} 个独立 SSW 事件")
        for ie, event in enumerate(events, start=1):
            winter_records.append({
                "init_year":     year,
                "winter_year":   winter_year,
                "event_index":   ie,
                "has_ssw_NDJFM": True,
                "ssw_date":      event["ssw_date"],
                "ssw_year":      event["ssw_date"].year,
                "ssw_month":     event["ssw_date"].month,
                "ssw_day":       event["ssw_date"].day,
                "end_date":      event["end_date"],
                "duration_days": event["duration_days"],
                "min_u":         event["min_u"],
                "min_u_date":    event["min_u_date"],
                "u_at_ssw":      float(u_series_winter.loc[event["ssw_date"]])
            })


# ===============================
# --- 输出结果 ---
# ===============================
print("\n" + "=" * 80)
print("输出结果...")
print("=" * 80)

winter_df = pd.DataFrame(winter_records)
for c in ["ssw_date", "end_date", "min_u_date"]:
    if c in winter_df.columns:
        winter_df[c] = pd.to_datetime(winter_df[c], errors="coerce")
winter_df = winter_df.sort_values(["init_year", "ssw_date"]).reset_index(drop=True)

all_csv = OUTPUT_DIR / f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_{START_YEAR}_{END_YEAR}.csv"
winter_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
print(f"完整结果已保存: {all_csv}")

events_only_df  = winter_df[winter_df["has_ssw_NDJFM"]].copy()
events_only_csv = OUTPUT_DIR / f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_events_only_{START_YEAR}_{END_YEAR}.csv"
events_only_df.to_csv(events_only_csv, index=False, encoding="utf-8-sig")
print(f"仅事件日期结果已保存: {events_only_csv}")

txt_file = OUTPUT_DIR / f"ERA5_SSW_dates_{PRESSURE_LABEL}_NDJFM_{START_YEAR}_{END_YEAR}.txt"
with open(txt_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"ERA5 SSW dates ({PRESSURE_LABEL}, NDJFM)\n")
    f.write(f"Analysis period: {START_YEAR}-{END_YEAR}\n")
    f.write(f"Criterion: {PRESSURE_LABEL}, {LATITUDE}N zonal-mean zonal wind reversal\n")
    f.write(f"Detection months: {DETECTION_MONTHS}\n")
    f.write(f"Retained months: {TARGET_MONTHS}\n")
    f.write(f"Minimum westerly separation: {MIN_WESTERLY_DAYS} days\n")
    f.write(f"Final Warming filter: recovery < {MIN_RECOVERY_DAYS} consec. "
            f"westerly days before Apr 30 → excluded\n")
    f.write("Input data: daily ERA5 multi-level file, Feb 29 removed in source file\n")
    f.write("=" * 80 + "\n\n")

    for year in range(START_YEAR, END_YEAR):
        sub         = events_only_df[events_only_df["init_year"] == year].copy()
        winter_year = to_winter_year_label(year)
        f.write(f"{winter_year}\n" + "-" * 60 + "\n")
        if len(sub) == 0:
            f.write("No NDJFM SSW detected.\n\n")
            continue
        sub = sub.sort_values(["ssw_date", "event_index"])
        for _, row in sub.iterrows():
            f.write(
                f"event={int(row['event_index'])}, "
                f"ssw_date={row['ssw_date'].strftime('%Y-%m-%d')}, "
                f"duration={int(row['duration_days'])} days, "
                f"min_u={row['min_u']:.2f} m/s\n"
            )
        f.write("\n")

print(f"文本结果已保存: {txt_file}")
print("\n" + "=" * 80)
print("完成")
print("=" * 80)
