from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import geopandas as gpd
from shapely import wkb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "OH_dashboard_2015_2024_nogeo.parquet"
LOCAL_GEOM_PATH = PROJECT_ROOT / "data" / "raw_local" / "OH_regrow_fieldID_geometry.parquet"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "OH_three_fields_story_2015_2024.parquet"

RANDOM_SEED = 42
MAX_SAMPLE = 1000


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


print("Loading processed Ohio panel...")
df = pd.read_parquet(INPUT_PATH)
print("Input shape:", df.shape)

print("Building field quality summary...")
field_summary = (
    df.groupby("field_id", dropna=False)
      .agg(
          years=("year", "nunique"),
          detected_years=("cover_crop_detected", "sum"),
          nonmissing_crop=("crop_type_code", lambda s: s.notna().sum()),
          nonmissing_ppt=("ppt_sum_gs", lambda s: s.notna().sum()),
          nonmissing_gdd=("gdd_gs", lambda s: s.notna().sum()),
          nonmissing_soil=("claytotal_r_30cm_weighted", lambda s: s.notna().sum()),
      )
      .reset_index()
)

candidates = field_summary[
    (field_summary["years"] >= 10) &
    (field_summary["nonmissing_crop"] >= 8) &
    (field_summary["nonmissing_ppt"] >= 8) &
    (field_summary["nonmissing_gdd"] >= 8) &
    (field_summary["nonmissing_soil"] >= 8)
].copy()

print("Candidate count:", len(candidates))

if len(candidates) < 3:
    raise ValueError("Not enough high-quality candidate fields found.")

if len(candidates) > MAX_SAMPLE:
    candidates = candidates.sample(MAX_SAMPLE, random_state=RANDOM_SEED).copy()
    print("Sampled candidate count:", len(candidates))

candidate_ids = candidates["field_id"].tolist()

print("Loading candidate geometries from local parquet...")
table = pq.read_table(
    LOCAL_GEOM_PATH,
    columns=["field_id", "geometry"],
    filters=[("field_id", "in", candidate_ids)],
)

geom_df = table.to_pandas()
geom_df["geometry"] = geom_df["geometry"].apply(lambda x: wkb.loads(x) if pd.notna(x) else None)

gdf_geom = gpd.GeoDataFrame(geom_df, geometry="geometry", crs="EPSG:5070")
gdf_geom["centroid"] = gdf_geom.geometry.centroid
gdf_geom["cx"] = gdf_geom["centroid"].x
gdf_geom["cy"] = gdf_geom["centroid"].y

geom_candidates = gdf_geom[gdf_geom["field_id"].isin(candidate_ids)].copy().reset_index(drop=True)
print("Geometry candidate count:", len(geom_candidates))

if len(geom_candidates) < 3:
    raise ValueError("Fewer than 3 geometry candidates found.")

coords = geom_candidates[["cx", "cy"]].to_numpy()
dist = pairwise_distances(coords)

best_triplet = None
best_score = np.inf
n = len(geom_candidates)

for i in range(n):
    nearest_idx = np.argsort(dist[i])[:3]
    if len(nearest_idx) == 3:
        score = dist[np.ix_(nearest_idx, nearest_idx)].sum()
        if score < best_score:
            best_score = score
            best_triplet = nearest_idx

selected_ids = geom_candidates.iloc[best_triplet]["field_id"].tolist()

print("Selected nearby field_ids:")
for fid in selected_ids:
    print(" -", fid)

story_df = df[df["field_id"].isin(selected_ids)].copy()
print("Story panel shape:", story_df.shape)

selected_geom = gdf_geom[gdf_geom["field_id"].isin(selected_ids)][["field_id", "geometry"]].copy()
story_gdf = story_df.merge(selected_geom, on="field_id", how="left")
story_gdf = gpd.GeoDataFrame(story_gdf, geometry="geometry", crs="EPSG:5070")

story_gdf.to_parquet(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Final shape:", story_gdf.shape)
print("Saved CRS:", story_gdf.crs)