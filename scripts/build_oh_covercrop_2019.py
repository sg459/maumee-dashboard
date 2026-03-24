from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


# =========================================================
# PATHS
# =========================================================
# Raw data on Windows shared drive, accessed from WSL/Ubuntu
RAW_ROOT = Path("/mnt/z/users/ganbat.3/Regrow")

# Project root inside Ubuntu
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output inside your project
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE = "OH"
YEAR = 2019
YY = "19"


# =========================================================
# HELPERS
# =========================================================
def annual_mean_from_monthly(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def first_existing_path(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exists: {candidates}")


# =========================================================
# FILE PATHS
# =========================================================
geom_path = first_existing_path(
    RAW_ROOT / f"{STATE}_regrow_fieldID_geometry.parquet",
    RAW_ROOT / f"{STATE}_regrow_fieldID_geometry.gpkg",
)

main_path = RAW_ROOT / f"{STATE}_regrow_table.parquet"
supp4_path = RAW_ROOT / f"{STATE}_regrow_supplement_4_table.parquet"
supp6_path = RAW_ROOT / f"{STATE}_regrow_supplement_6_table.parquet"
supp7_path = RAW_ROOT / f"{STATE}_regrow_supplement_7_table.parquet"
supp8_path = RAW_ROOT / f"{STATE}_regrow_supplement_8_table.parquet"

print("Using files:")
print(" geometry:", geom_path)
print(" main    :", main_path)
print(" supp4   :", supp4_path)
print(" supp6   :", supp6_path)
print(" supp7   :", supp7_path)
print(" supp8   :", supp8_path)


# =========================================================
# COLUMN LISTS
# =========================================================
main_cols = [
    "field_id",
    f"cover_{YY}_1",   # cycle 1 only
    f"crop_{YY}_1",    # cycle 1 only
]

supp4_cols = [
    "field_id",
    "dist_to_road",
]

supp6_cols = (
    ["field_id"]
    + [f"ppt_mean_{YEAR}{m:02d}" for m in range(1, 13)]
    + [f"tmean_mean_{YEAR}{m:02d}" for m in range(1, 13)]
)

supp7_cols = (
    ["field_id"]
    + [f"corn_price_county_{YEAR}{m:02d}" for m in range(1, 13)]
    + [f"soybeans_price_county_{YEAR}{m:02d}" for m in range(1, 13)]
)

supp8_cols = [
    "field_id",
    "claytotal_r_30cm_weighted",
    "sandtotal_r_30cm_weighted",
    "ph1to1h2o_r_30cm_weighted",
    "drainagecl_dominant",
    "cropprodindex_dominant",
]


# =========================================================
# LOAD DATA
# =========================================================
print("Skipping geometry for now...")
gdf_geom = None

# print("Geometry rows:", len(gdf_geom))

print("Loading main table...")
main = pd.read_parquet(main_path, columns=main_cols)
print("Main rows:", len(main))

print("Loading supplement 4...")
supp4 = pd.read_parquet(supp4_path, columns=supp4_cols)
print("Supp4 rows:", len(supp4))

print("Loading supplement 6...")
supp6 = pd.read_parquet(supp6_path, columns=supp6_cols)
print("Supp6 rows:", len(supp6))

print("Loading supplement 7...")
supp7 = pd.read_parquet(supp7_path, columns=supp7_cols)
print("Supp7 rows:", len(supp7))

print("Loading supplement 8...")
supp8 = pd.read_parquet(supp8_path, columns=supp8_cols)
print("Supp8 rows:", len(supp8))

# =========================================================
# BUILD YEAR TABLE
# =========================================================
print(f"Processing year {YEAR}...")

df = main.copy()

# Cover crop status
cover_col = f"cover_{YY}_1"
df["cover_crop_status"] = pd.to_numeric(df[cover_col], errors="coerce")
df["cover_crop_detected"] = df["cover_crop_status"].eq(3).fillna(False).astype(int)

# Crop type (kept as numeric code for now)
crop_col = f"crop_{YY}_1"
df["crop_type_code"] = pd.to_numeric(df[crop_col], errors="coerce")

# Annual weather means
ppt_cols = [f"ppt_mean_{YEAR}{m:02d}" for m in range(1, 13)]
tmean_cols = [f"tmean_mean_{YEAR}{m:02d}" for m in range(1, 13)]

supp6_year = pd.DataFrame({
    "field_id": supp6["field_id"],
    "ppt_mean_annual": annual_mean_from_monthly(supp6, ppt_cols),
    "tmean_mean_annual": annual_mean_from_monthly(supp6, tmean_cols),
})

# Annual price means
corn_cols = [f"corn_price_county_{YEAR}{m:02d}" for m in range(1, 13)]
soy_cols = [f"soybeans_price_county_{YEAR}{m:02d}" for m in range(1, 13)]

supp7_year = pd.DataFrame({
    "field_id": supp7["field_id"],
    "corn_price_county_annual": annual_mean_from_monthly(supp7, corn_cols),
    "soybeans_price_county_annual": annual_mean_from_monthly(supp7, soy_cols),
})

# Join everything
print("Joining tables...")
df = (
    df[["field_id", "cover_crop_status", "cover_crop_detected", "crop_type_code"]]
    .merge(supp4, on="field_id", how="left")
    .merge(supp6_year, on="field_id", how="left")
    .merge(supp7_year, on="field_id", how="left")
    .merge(supp8, on="field_id", how="left")
)

df["year"] = YEAR

# Convert to GeoDataFrame
gdf_out = df.copy()

# Clean column order
final_cols = [
    "field_id",
    "year",
    "cover_crop_status",
    "cover_crop_detected",
    "crop_type_code",
    "ppt_mean_annual",
    "tmean_mean_annual",
    "corn_price_county_annual",
    "soybeans_price_county_annual",
    "dist_to_road",
    "claytotal_r_30cm_weighted",
    "sandtotal_r_30cm_weighted",
    "ph1to1h2o_r_30cm_weighted",
    "drainagecl_dominant",
    "cropprodindex_dominant",
]
gdf_out = gdf_out[final_cols]

print("Final rows:", len(gdf_out))
print(gdf_out.head())


# =========================================================
# SAVE OUTPUT
# =========================================================
out_path = OUT_DIR / "OH_covercrop_dashboard_2019.parquet"

print("Saving output...")
gdf_out.to_parquet(out_path, index=False)

print(f"Saved: {out_path}")