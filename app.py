import streamlit as st
from data_access.sample_data import load_sample_data

st.set_page_config(page_title="Maumee Dashboard", layout="wide")

st.title("Maumee Dashboard")
st.caption("Internal dashboard prototype for field-level crop, weather, and historical data.")

# Load data
data = load_sample_data()

with st.sidebar:
    st.header("Filters")

    state = st.selectbox("State", sorted(data["state"].unique()))
    year = st.selectbox("Year", sorted(data["year"].unique()))
    crop_options = ["All"] + sorted(data["crop"].unique())
    crop = st.selectbox("Crop", crop_options)
    variable = st.selectbox("Variable", ["Precipitation"])

filtered = data[(data["state"] == state) & (data["year"] == year)]

if crop != "All":
    filtered = filtered[filtered["crop"] == crop]

fields_selected = len(filtered)
total_acres = filtered["acres"].sum()

if fields_selected > 0:
    most_common_crop = filtered["crop"].mode().iloc[0]
    avg_precip = round(filtered["precipitation"].mean(), 1)
else:
    most_common_crop = "N/A"
    avg_precip = "N/A"

col1, col2, col3, col4 = st.columns(4)

col1.metric("Fields Selected", fields_selected)
col2.metric("Total Acres", total_acres)
col3.metric("Most Common Crop", most_common_crop)
col4.metric("Avg Precipitation", avg_precip)

st.divider()

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Map")
    st.info("Map will go here.")

with right_col:
    st.subheader("Field Details")
    if fields_selected > 0:
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("No fields match the selected filters.")

st.divider()

st.subheader("Summary Charts")
if fields_selected > 0:
    st.bar_chart(filtered.set_index("field_id")["acres"])
else:
    st.info("No data available for chart.")

st.divider()

st.subheader("Download Data")
if fields_selected > 0:
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_data,
        file_name="filtered_fields.csv",
        mime="text/csv",
    )
else:
    st.info("No data available for download.")