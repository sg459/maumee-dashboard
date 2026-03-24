from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

RAW_ROOT = Path("/mnt/z/users/ganbat.3/Regrow")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE = "OH"
YEARS = list(range(2015, 2025))  # 2015-2024

GS_MONTHS = [4, 5, 6, 7, 8, 9, 10]
DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}
GDD_BASE_C = 10.0


def yy(year: int) -> str:
    return str(year)[-2:]


def annual_mean_from_monthly(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def mean_from_selected_months(df: pd.DataFrame, prefix: str, year: int, months: list[int]) -> pd.Series:
    cols = [f"{prefix}_{year}{m:02d}" for m in months if f"{prefix}_{year}{m:02d}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def sum_from_selected_months(df: pd.DataFrame, prefix: str, year: int, months: list[int]) -> pd.Series:
    cols = [f"{prefix}_{year}{m:02d}" for m in months if f"{prefix}_{year}{m:02d}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def compute_gdd_from_monthly_tmean(df: pd.DataFrame, year: int, months: list[int], base_c: float = 10.0) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    any_found = False

    for m in months:
        col = f"tmean_mean_{year}{m:02d}"
        if col in df.columns:
            tmean = pd.to_numeric(df[col], errors="coerce")
            monthly_gdd = (tmean - base_c).clip(lower=0) * DAYS_IN_MONTH[m]
            out = out.add(monthly_gdd.fillna(0), fill_value=0)
            any_found = True

    if not any_found:
        return pd.Series(np.nan, index=df.index)

    return out


main_path = RAW_ROOT / f"{STATE}_regrow_table.parquet"
supp4_path = RAW_ROOT / f"{STATE}_regrow_supplement_4_table.parquet"
supp6_path = RAW_ROOT / f"{STATE}_regrow_supplement_6_table.parquet"
supp7_path = RAW_ROOT / f"{STATE}_regrow_supplement_7_table.parquet"
supp8_path = RAW_ROOT / f"{STATE}_regrow_supplement_8_table.parquet"

print("Using files:")
print(" main    :", main_path)
print(" supp4   :", supp4_path)
print(" supp6   :", supp6_path)
print(" supp7   :", supp7_path)
print(" supp8   :", supp8_path)

main_schema = pq.ParquetFile(main_path).schema_arrow.names
supp6_schema = pq.ParquetFile(supp6_path).schema_arrow.names
supp7_schema = pq.ParquetFile(supp7_path).schema_arrow.names
supp4_schema = pq.ParquetFile(supp4_path).schema_arrow.names
supp8_schema = pq.ParquetFile(supp8_path).schema_arrow.names

main_cols = ["field_id"]
for year in YEARS:
    y2 = yy(year)
    for col in [f"cover_{y2}_1", f"crop_{y2}_1", f"PPtill_{y2}_1", f"PHtill_{y2}_1"]:
        if col in main_schema:
            main_cols.append(col)

supp4_cols = [c for c in ["field_id", "dist_to_road"] if c in supp4_schema]

supp8_cols = [c for c in [
    "field_id",
    "claytotal_r_30cm_weighted",
    "sandtotal_r_30cm_weighted",
    "ph1to1h2o_r_30cm_weighted",
    "drainagecl_dominant",
    "cropprodindex_dominant",
] if c in supp8_schema]

supp6_cols = ["field_id"]
for year in YEARS:
    for m in range(1, 13):
        for col in [f"ppt_mean_{year}{m:02d}", f"tmean_mean_{year}{m:02d}"]:
            if col in supp6_schema:
                supp6_cols.append(col)

supp7_cols = ["field_id"]
for year in YEARS:
    for m in range(1, 13):
        for col in [f"corn_price_county_{year}{m:02d}", f"soybeans_price_county_{year}{m:02d}"]:
            if col in supp7_schema:
                supp7_cols.append(col)

print("Loading main table...")
main = pd.read_parquet(main_path, columns=main_cols)
print("Main shape:", main.shape)

print("Loading supplement 4...")
supp4 = pd.read_parquet(supp4_path, columns=supp4_cols)
print("Supp4 shape:", supp4.shape)

print("Loading supplement 6...")
supp6 = pd.read_parquet(supp6_path, columns=supp6_cols)
print("Supp6 shape:", supp6.shape)

print("Replacing -9999 with NaN in weather data...")
weather_cols = [c for c in supp6.columns if c != "field_id"]
supp6[weather_cols] = supp6[weather_cols].replace(-9999, np.nan)

print("Loading supplement 7...")
supp7 = pd.read_parquet(supp7_path, columns=supp7_cols)
print("Supp7 shape:", supp7.shape)

print("Loading supplement 8...")
supp8 = pd.read_parquet(supp8_path, columns=supp8_cols)
print("Supp8 shape:", supp8.shape)

cover_label_map = {
    1: "No cover crop detected",
    2: "Potential cover crop",
    3: "Cover crop detected",
}

tillage_label_map = {
    1: "No tillage",
    2: "Reduced tillage",
    3: "Conventional tillage",
    4: "Double tillage (reduced+reduced)",
    5: "Double tillage (reduced+conventional)",
    6: "Double tillage (conventional+conventional)",
}

print("Building long field-year status table...")
main_records = []

for year in YEARS:
    y2 = yy(year)

    cover_col = f"cover_{y2}_1"
    crop_col = f"crop_{y2}_1"
    pp_col = f"PPtill_{y2}_1"
    ph_col = f"PHtill_{y2}_1"

    tmp = pd.DataFrame({
        "field_id": main["field_id"],
        "year": year,
        "cover_crop_status": pd.to_numeric(main[cover_col], errors="coerce") if cover_col in main.columns else np.nan,
        "crop_type_code": pd.to_numeric(main[crop_col], errors="coerce") if crop_col in main.columns else np.nan,
        "PPtill_status": pd.to_numeric(main[pp_col], errors="coerce") if pp_col in main.columns else np.nan,
        "PHtill_status": pd.to_numeric(main[ph_col], errors="coerce") if ph_col in main.columns else np.nan,
    })

    tmp["cover_crop_detected"] = tmp["cover_crop_status"].eq(3).fillna(False).astype(int)
    tmp["cover_crop_label"] = tmp["cover_crop_status"].map(cover_label_map)
    tmp["PPtill_label"] = tmp["PPtill_status"].map(tillage_label_map)
    tmp["PHtill_label"] = tmp["PHtill_status"].map(tillage_label_map)

    main_records.append(tmp)

main_long = pd.concat(main_records, ignore_index=True)
print("main_long shape:", main_long.shape)

print("Building long weather table...")
weather_records = []

for year in YEARS:
    tmp = pd.DataFrame({
        "field_id": supp6["field_id"],
        "year": year,
        "ppt_mean_annual": annual_mean_from_monthly(
            supp6, [f"ppt_mean_{year}{m:02d}" for m in range(1, 13)]
        ),
        "tmean_mean_annual": annual_mean_from_monthly(
            supp6, [f"tmean_mean_{year}{m:02d}" for m in range(1, 13)]
        ),
        "ppt_mean_gs": mean_from_selected_months(supp6, "ppt_mean", year, GS_MONTHS),
        "tmean_mean_gs": mean_from_selected_months(supp6, "tmean_mean", year, GS_MONTHS),
        "ppt_sum_gs": sum_from_selected_months(supp6, "ppt_mean", year, GS_MONTHS),
        "gdd_gs": compute_gdd_from_monthly_tmean(supp6, year, GS_MONTHS, base_c=GDD_BASE_C),
    })
    weather_records.append(tmp)

weather_long = pd.concat(weather_records, ignore_index=True)
print("weather_long shape:", weather_long.shape)

print("Building long price table...")
price_records = []

for year in YEARS:
    tmp = pd.DataFrame({
        "field_id": supp7["field_id"],
        "year": year,
        "corn_price_county_annual": annual_mean_from_monthly(
            supp7, [f"corn_price_county_{year}{m:02d}" for m in range(1, 13)]
        ),
        "soybeans_price_county_annual": annual_mean_from_monthly(
            supp7, [f"soybeans_price_county_{year}{m:02d}" for m in range(1, 13)]
        ),
    })
    price_records.append(tmp)

price_long = pd.concat(price_records, ignore_index=True)
print("price_long shape:", price_long.shape)

print("Joining tables...")
df = (
    main_long
    .merge(weather_long, on=["field_id", "year"], how="left")
    .merge(price_long, on=["field_id", "year"], how="left")
    .merge(supp4, on="field_id", how="left")
    .merge(supp8, on="field_id", how="left")
)

print("Merged shape:", df.shape)

print("Computing precipitation anomaly and year type...")
field_stats = (
    df.groupby("field_id", dropna=False)["ppt_sum_gs"]
      .agg(ppt_sum_gs_mean_field="mean", ppt_sum_gs_sd_field="std")
      .reset_index()
)

df = df.merge(field_stats, on="field_id", how="left")

df["ppt_sum_gs_z"] = (
    (df["ppt_sum_gs"] - df["ppt_sum_gs_mean_field"]) /
    df["ppt_sum_gs_sd_field"]
)

df.loc[~np.isfinite(df["ppt_sum_gs_z"]), "ppt_sum_gs_z"] = np.nan

df["year_type"] = np.select(
    [
        df["ppt_sum_gs_z"] <= -0.5,
        df["ppt_sum_gs_z"] >= 0.5,
    ],
    [
        "Dry",
        "Wet",
    ],
    default="Normal"
)

df.loc[df["ppt_sum_gs"].isna(), "year_type"] = np.nan

final_cols = [
    "field_id",
    "year",
    "cover_crop_status",
    "cover_crop_label",
    "cover_crop_detected",
    "crop_type_code",
    "PPtill_status",
    "PPtill_label",
    "PHtill_status",
    "PHtill_label",
    "ppt_mean_annual",
    "tmean_mean_annual",
    "ppt_mean_gs",
    "tmean_mean_gs",
    "ppt_sum_gs",
    "gdd_gs",
    "ppt_sum_gs_mean_field",
    "ppt_sum_gs_sd_field",
    "ppt_sum_gs_z",
    "year_type",
    "corn_price_county_annual",
    "soybeans_price_county_annual",
    "dist_to_road",
    "claytotal_r_30cm_weighted",
    "sandtotal_r_30cm_weighted",
    "ph1to1h2o_r_30cm_weighted",
    "drainagecl_dominant",
    "cropprodindex_dominant",
]

existing_final_cols = [c for c in final_cols if c in df.columns]
df = df[existing_final_cols]

print("Final shape:", df.shape)

out_path = OUT_DIR / "OH_dashboard_2015_2024_nogeo.parquet"
print("Saving to:", out_path)
df.to_parquet(out_path, index=False)
print("Saved successfully.")