from pathlib import Path
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium

st.set_page_config(page_title="Ohio 3-Field Story Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "OH_three_fields_story_2015_2024.parquet"
CROP_CODES_PATH = PROJECT_ROOT / "data" / "metadata" / "crop_codes.xlsx"


@st.cache_data
def load_data():
    gdf = gpd.read_parquet(DATA_PATH)
    gdf = gdf.to_crs(epsg=4326)

    crop_codes = pd.read_excel(CROP_CODES_PATH)
    crop_codes.columns = [c.strip() for c in crop_codes.columns]
    crop_lookup = dict(zip(crop_codes["Crop Type Code"], crop_codes["Display Name"]))
    gdf["crop_name"] = gdf["crop_type_code"].map(crop_lookup)

    for c in [
        "cover_crop_label",
        "year_type",
        "drainagecl_dominant",
        "crop_name",
        "PPtill_label",
        "PHtill_label",
    ]:
        if c in gdf.columns:
            gdf[c] = gdf[c].astype("string")

    return gdf


def truncate_label(val, max_len=16):
    if pd.isna(val):
        return "Unknown"
    s = str(val)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def build_matrix_panel(df_field: pd.DataFrame, title: str) -> go.Figure:
    attrs = [
        ("Crop Type", "crop_name"),
        ("Cover Crop", "cover_crop_label"),
        ("Pre-Plant Tillage", "PPtill_label"),
        ("Post-Harvest Tillage", "PHtill_label"),
        ("Climate Year", "year_type"),
    ]

    years = sorted(df_field["year"].dropna().astype(int).unique().tolist())
    x_labels = [a[0] for a in attrs]
    y_labels = [str(y) for y in years]

    # sophisticated attribute-specific color maps
    color_maps = {
        "Crop Type": {
            "Corn": "#2C7FB8",
            "Soybeans": "#8CC8F2",
            "Fallow/Idle Cropland": "#C65F5F",
            "Rice": "#E8B4B8",
            "Unknown": "#BDBDBD",
        },
        "Cover Crop": {
            "No cover crop detected": "#1F5AA6",
            "Potential cover crop": "#8EC5F4",
            "Cover crop detected": "#2E8B57",
            "Unknown": "#BDBDBD",
        },
        "Pre-Plant Tillage": {
            "No tillage": "#6BAF6B",
            "Reduced tillage": "#D8B24C",
            "Conventional tillage": "#A66A3F",
            "Double tillage (reduced+reduced)": "#9ACD66",
            "Double tillage (reduced+conventional)": "#C8883A",
            "Double tillage (conventional+conventional)": "#8B4A2F",
            "Unknown": "#BDBDBD",
        },
        "Post-Harvest Tillage": {
            "No tillage": "#5E9E5E",
            "Reduced tillage": "#C9A53D",
            "Conventional tillage": "#9C6035",
            "Double tillage (reduced+reduced)": "#8FBE58",
            "Double tillage (reduced+conventional)": "#B9782E",
            "Double tillage (conventional+conventional)": "#7A3E25",
            "Unknown": "#BDBDBD",
        },
        "Climate Year": {
            "Dry": "#B88A52",
            "Normal": "#7F8C8D",
            "Wet": "#4F81BD",
            "Unknown": "#BDBDBD",
        },
    }

    color_id = {}
    color_rgb = {}
    next_id = 1

    z = []
    text = []
    hover = []

    df_field = df_field.sort_values("year").copy()

    for year in years:
        row_vals = []
        row_text = []
        row_hover = []
        row = df_field[df_field["year"] == year]
        if len(row) == 0:
            row = None
        else:
            row = row.iloc[0]

        for attr_label, attr_col in attrs:
            raw_val = "Unknown" if row is None or pd.isna(row[attr_col]) else str(row[attr_col])
            display_val = truncate_label(raw_val, max_len=16)

            if raw_val not in color_maps[attr_label]:
                fill = "#D0D0D0"
            else:
                fill = color_maps[attr_label][raw_val]

            key = (attr_label, raw_val)
            if key not in color_id:
                color_id[key] = next_id
                color_rgb[key] = fill
                next_id += 1

            row_vals.append(color_id[key])
            row_text.append(display_val)
            row_hover.append(f"{attr_label}<br>Year: {year}<br>Value: {raw_val}")

        z.append(row_vals)
        text.append(row_text)
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertext=hover,
            hoverinfo="text",
            showscale=False,
            xgap=4,
            ygap=4,
            colorscale=[
                [0.0, "#FFFFFF"],
                [1.0, "#FFFFFF"],
            ],
        )
    )

    # apply per-cell colors by building a custom discrete colorscale
    # Plotly heatmap needs numeric z + continuous scale, so map ids to exact colors.
    n_ids = max(color_id.values()) if color_id else 1
    scale = []
    for (attr_label, raw_val), cid in sorted(color_id.items(), key=lambda x: x[1]):
        start = (cid - 1) / n_ids
        end = cid / n_ids
        c = color_rgb[(attr_label, raw_val)]
        scale.append([start, c])
        scale.append([end, c])

    fig.data[0].colorscale = scale
    fig.data[0].zmin = 1
    fig.data[0].zmax = n_ids

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="",
        yaxis_title="Year",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


gdf = load_data()

field_options = sorted(gdf["field_id"].dropna().unique().tolist())
field_label_map = {fid: f"Field {chr(65+i)}" for i, fid in enumerate(field_options)}
gdf["field_label"] = gdf["field_id"].map(field_label_map)

st.title("Ohio 3-Field Story Dashboard")
st.caption("2015–2024 prototype for nearby Ohio fields")

with st.sidebar:
    st.header("View Options")
    view_mode = st.radio("Mode", ["Single Field", "Compare All Three"])

    if view_mode == "Single Field":
        selected_field = st.selectbox(
            "Field",
            field_options,
            format_func=lambda x: f"{field_label_map[x]} ({x[:8]}...)"
        )
        selected_fields = [selected_field]
    else:
        selected_fields = field_options

plot_gdf = gdf[gdf["field_id"].isin(selected_fields)].copy()
plot_gdf["year"] = plot_gdf["year"].astype(int)

# -------------------------
# TOP SUMMARY TABLE
# -------------------------
st.subheader("Climate Summary & Latest Year Snapshot")

summary_rows = []
for fid in selected_fields:
    fg = plot_gdf[plot_gdf["field_id"] == fid].sort_values("year")
    last = fg[fg["year"] == fg["year"].max()].iloc[0]

    summary_rows.append({
        "Field": field_label_map[fid],
        "Dry Years": int((fg["year_type"] == "Dry").sum()),
        "Normal Years": int((fg["year_type"] == "Normal").sum()),
        "Wet Years": int((fg["year_type"] == "Wet").sum()),
        "Latest Crop": str(last["crop_name"]) if pd.notna(last["crop_name"]) else "NA",
        "Latest Cover Crop": str(last["cover_crop_label"]) if pd.notna(last["cover_crop_label"]) else "NA",
        "Latest Pre-Plant Tillage": str(last["PPtill_label"]) if pd.notna(last["PPtill_label"]) else "NA",
        "Latest Post-Harvest Tillage": str(last["PHtill_label"]) if pd.notna(last["PHtill_label"]) else "NA",
        "Latest Climate Year": str(last["year_type"]) if pd.notna(last["year_type"]) else "NA",
        "Avg GS Precipitation": round(float(fg["ppt_sum_gs"].mean()), 1) if fg["ppt_sum_gs"].notna().any() else None,
        "Avg GS GDD": round(float(fg["gdd_gs"].mean()), 1) if fg["gdd_gs"].notna().any() else None,
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width="stretch", hide_index=True)

st.divider()

# -------------------------
# MAP
# -------------------------
st.subheader("Field Map")

map_gdf = plot_gdf.drop_duplicates(subset=["field_id"]).copy()

for idx, row in map_gdf.iterrows():
    field_hist = plot_gdf[plot_gdf["field_id"] == row["field_id"]].sort_values("year")
    last = field_hist.iloc[-1]

    map_gdf.loc[idx, "field_label"] = field_label_map[row["field_id"]]
    map_gdf.loc[idx, "latest_crop"] = str(last["crop_name"]) if pd.notna(last["crop_name"]) else "NA"
    map_gdf.loc[idx, "latest_cover"] = str(last["cover_crop_label"]) if pd.notna(last["cover_crop_label"]) else "NA"
    map_gdf.loc[idx, "latest_pp_tillage"] = str(last["PPtill_label"]) if pd.notna(last["PPtill_label"]) else "NA"
    map_gdf.loc[idx, "latest_ph_tillage"] = str(last["PHtill_label"]) if pd.notna(last["PHtill_label"]) else "NA"
    map_gdf.loc[idx, "latest_year_type"] = str(last["year_type"]) if pd.notna(last["year_type"]) else "NA"
    map_gdf.loc[idx, "drainage"] = str(last["drainagecl_dominant"]) if pd.notna(last["drainagecl_dominant"]) else "NA"
    map_gdf.loc[idx, "soil_ph"] = round(float(last["ph1to1h2o_r_30cm_weighted"]), 2) if pd.notna(last["ph1to1h2o_r_30cm_weighted"]) else None
    map_gdf.loc[idx, "clay"] = round(float(last["claytotal_r_30cm_weighted"]), 2) if pd.notna(last["claytotal_r_30cm_weighted"]) else None
    map_gdf.loc[idx, "sand"] = round(float(last["sandtotal_r_30cm_weighted"]), 2) if pd.notna(last["sandtotal_r_30cm_weighted"]) else None
    map_gdf.loc[idx, "crop_prod"] = round(float(last["cropprodindex_dominant"]), 1) if pd.notna(last["cropprodindex_dominant"]) else None

if len(map_gdf) > 0 and map_gdf.geometry.notna().any():
    center = map_gdf.geometry.union_all().centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=13, tiles="CartoDB positron")

    color_cycle = ["#2e7d32", "#1565c0", "#ef6c00"]

    for i, (_, row) in enumerate(map_gdf.iterrows()):
        field_one = map_gdf[map_gdf["field_id"] == row["field_id"]]
        color = color_cycle[i % len(color_cycle)]

        folium.GeoJson(
            data=field_one.__geo_interface__,
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 3,
                "fillOpacity": 0.35,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "field_label", "latest_crop", "latest_cover",
                    "latest_pp_tillage", "latest_ph_tillage",
                    "latest_year_type", "drainage", "soil_ph", "clay",
                    "sand", "crop_prod"
                ],
                aliases=[
                    "Field", "Latest Crop", "Cover Crop",
                    "Pre-Plant Tillage", "Post-Harvest Tillage",
                    "Latest Climate Year", "Drainage", "Soil pH",
                    "Clay", "Sand", "Crop Productivity Index"
                ],
            ),
        ).add_to(m)

    bounds = map_gdf.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    st_folium(m, width=1100, height=500)
else:
    st.warning("No geometry available.")

st.divider()

# -------------------------
# MERGED LABELED MATRIX
# -------------------------
st.subheader("Annual Management and Climate Matrix")

if view_mode == "Single Field":
    fid = selected_fields[0]
    fg = plot_gdf[plot_gdf["field_id"] == fid].copy().sort_values("year")
    fig_matrix = build_matrix_panel(fg, f"{field_label_map[fid]} Annual Matrix")
    st.plotly_chart(fig_matrix, use_container_width=True)
else:
    for fid in selected_fields:
        fg = plot_gdf[plot_gdf["field_id"] == fid].copy().sort_values("year")
        fig_matrix = build_matrix_panel(fg, f"{field_label_map[fid]} Annual Matrix")
        st.plotly_chart(fig_matrix, use_container_width=True)

st.divider()

# -------------------------
# WEATHER CHARTS
# -------------------------
st.subheader("Weather Trends")

fig_ppt = px.line(
    plot_gdf,
    x="year",
    y="ppt_sum_gs",
    color="field_label",
    markers=True,
    title="Growing Season Precipitation by Year"
)
fig_ppt.update_layout(height=420, xaxis_title="Year", yaxis_title="GS Precipitation", legend_title="Field")
fig_ppt.update_xaxes(dtick=1, tickangle=-90)
st.plotly_chart(fig_ppt, use_container_width=True)

fig_gdd = px.line(
    plot_gdf,
    x="year",
    y="gdd_gs",
    color="field_label",
    markers=True,
    title="Growing Season GDD by Year"
)
fig_gdd.update_layout(height=420, xaxis_title="Year", yaxis_title="GS GDD", legend_title="Field")
fig_gdd.update_xaxes(dtick=1, tickangle=-90)
st.plotly_chart(fig_gdd, use_container_width=True)

st.divider()

# -------------------------
# TABLE
# -------------------------
st.subheader("Annual History Table")

history = plot_gdf[[
    "field_id",
    "year",
    "crop_name",
    "cover_crop_label",
    "PPtill_label",
    "PHtill_label",
    "cover_crop_detected",
    "ppt_sum_gs",
    "tmean_mean_gs",
    "gdd_gs",
    "year_type",
    "corn_price_county_annual",
    "soybeans_price_county_annual",
    "claytotal_r_30cm_weighted",
    "sandtotal_r_30cm_weighted",
    "ph1to1h2o_r_30cm_weighted",
    "drainagecl_dominant",
    "cropprodindex_dominant",
    "dist_to_road",
]].rename(columns={
    "field_id": "Field ID",
    "year": "Year",
    "crop_name": "Crop Type",
    "cover_crop_label": "Cover Crop",
    "PPtill_label": "Pre-Plant Tillage",
    "PHtill_label": "Post-Harvest Tillage",
    "cover_crop_detected": "Cover Crop Detected",
    "ppt_sum_gs": "GS Precipitation",
    "tmean_mean_gs": "GS Mean Temperature",
    "gdd_gs": "GS GDD",
    "year_type": "Climate Year",
    "corn_price_county_annual": "Annual Corn Price",
    "soybeans_price_county_annual": "Annual Soybean Price",
    "claytotal_r_30cm_weighted": "Clay (0-30 cm)",
    "sandtotal_r_30cm_weighted": "Sand (0-30 cm)",
    "ph1to1h2o_r_30cm_weighted": "Soil pH",
    "drainagecl_dominant": "Drainage Class",
    "cropprodindex_dominant": "Crop Productivity Index",
    "dist_to_road": "Distance to Road",
}).sort_values(["Field ID", "Year"])

history["Field"] = history["Field ID"].map(field_label_map)

history = history[[
    "Field",
    "Field ID",
    "Year",
    "Crop Type",
    "Cover Crop",
    "Pre-Plant Tillage",
    "Post-Harvest Tillage",
    "Climate Year",
    "Cover Crop Detected",
    "GS Precipitation",
    "GS Mean Temperature",
    "GS GDD",
    "Annual Corn Price",
    "Annual Soybean Price",
    "Clay (0-30 cm)",
    "Sand (0-30 cm)",
    "Soil pH",
    "Drainage Class",
    "Crop Productivity Index",
    "Distance to Road",
]]

st.dataframe(history, width="stretch")

# -------------------------
# DOWNLOAD
# -------------------------
st.subheader("Download CSV")

download_cols = st.multiselect(
    "Choose columns to include in the CSV",
    options=history.columns.tolist(),
    default=history.columns.tolist()
)

download_df = history[download_cols].copy() if download_cols else history.copy()
csv_data = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download selected columns as CSV",
    data=csv_data,
    file_name="oh_three_fields_history.csv",
    mime="text/csv",
)