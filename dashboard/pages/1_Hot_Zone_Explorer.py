"""Page 1 — Hot Zone Explorer: filters, map, temporal charts, heatmap, top zones."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Note: Plotly >= 5.24 migrated from Mapbox to MapLibre.
# Use px.choropleth_map / px.scatter_map / go.Scattermap (not *_mapbox variants).
from pathlib import Path
import numpy as np
import geopandas as gpd
import json

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "output"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_data
def load_2014() -> pd.DataFrame | None:
    """Load 2014 aggregated data (full dataset, zone-level)."""
    path = DATA_DIR / "dashboard_2014_aggregated.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["LocationID"] = df["LocationID"].astype(int)
    return df


@st.cache_data
def load_2015() -> pd.DataFrame | None:
    """Load 2015 aggregated data."""
    path = DATA_DIR / "dashboard_2015_aggregated.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_centers() -> pd.DataFrame | None:
    """Load KMeans cluster centers."""
    path = DATA_DIR / "kmeans_hotzone_centers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_geojson() -> dict | None:
    """Load NYC taxi zone polygons as GeoJSON for choropleth."""
    shp_path = Path(__file__).parent.parent.parent / "data" / "input" / "taxi_zones" / "taxi_zones.shp"
    if not shp_path.exists():
        return None
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)  # Reproject to WGS84 for web maps
    return json.loads(gdf.to_json())


st.title("Hot Zone Explorer")

st.sidebar.header("Filters")

year = st.sidebar.radio("Year", [2014, 2015], index=0)

month_labels = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

if year == 2014:
    months_available = list(range(4, 10))  # Apr-Sep
    default_months = [4, 5, 6]  # Apr-Jun (overlap period with 2015)
else:
    months_available = list(range(1, 7))  # Jan-Jun
    default_months = [4, 5, 6]  # Apr-Jun (overlap period with 2014)

selected_months = st.sidebar.multiselect(
    "Month",
    options=months_available,
    default=default_months,
    format_func=lambda x: month_labels[x],
)

selected_days = st.sidebar.multiselect(
    "Day of week",
    options=list(range(7)),
    default=list(range(7)),
    format_func=lambda x: DAY_NAMES[x],
)

hour_range = st.sidebar.slider("Hour range", min_value=0, max_value=23, value=(0, 23))

centers_df = load_centers()

if year == 2014:
    raw = load_2014()
    if raw is None:
        st.error("Data not found. Run notebooks first. (dashboard_2014_aggregated.csv)")
        st.stop()

    mask = (
        raw["month"].isin(selected_months)
        & raw["day_of_week"].isin(selected_days)
        & raw["hour"].between(hour_range[0], hour_range[1])
    )
    df = raw[mask].copy()

else:
    raw = load_2015()
    if raw is None:
        st.error("Data not found. Run notebooks first. (dashboard_2015_aggregated.csv)")
        st.stop()

    mask = (
        raw["month"].isin(selected_months)
        & raw["day_of_week"].isin(selected_days)
        & raw["hour"].between(hour_range[0], hour_range[1])
    )
    df = raw[mask].copy()

if df.empty:
    st.warning("No data for selected filters. Adjust the sidebar.")
    st.stop()

if year == 2014:
    total_pickups = int(df["count"].sum())
    n_active = df["LocationID"].nunique()
    peak_hour = int(df.groupby("hour")["count"].sum().idxmax())
    peak_zone = (
        df.groupby("zone")["count"].sum().idxmax() if "zone" in df.columns else "N/A"
    )
else:
    total_pickups = int(df["count"].sum())
    n_active = df["locationID"].nunique() if "locationID" in df.columns else 0
    peak_hour = int(df.groupby("hour")["count"].sum().idxmax())
    peak_zone = (
        df.groupby("zone")["count"].sum().idxmax() if "zone" in df.columns else "N/A"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pickups", f"{total_pickups:,}")
if year == 2014:
    n_clusters = len(centers_df) if centers_df is not None else 0
    c2.metric("Hot-Zone Clusters", str(n_clusters))
else:
    c2.metric("Active Taxi Zones", str(n_active))
c3.metric("Peak Hour", f"{peak_hour:02d}:00")
c4.metric("Peak Zone", str(peak_zone)[:30])

if year == 2014:
    st.caption(
        "Hot-Zone Clusters are KMeans-derived groupings of GPS pickups into high-demand macro-zones. "
        "See Methodology for details on how clusters are computed."
    )
    st.page_link(
        "pages/3_Methodology.py",
        label="See cluster visualization and methodology",
    )

st.caption(
    'Data note: zones labeled "Unknown" (TLC zones 264/265, representing 0.044% of 2015 trips)'
    " have been excluded from this analysis as they lack a real geographic location."
)

st.divider()

st.subheader("Interactive Map")

geojson = load_geojson()

if geojson is None:
    st.error("Taxi zone shapefile not found. Place it in data/input/taxi_zones/.")
    st.stop()

if year == 2014:
    zone_agg = (
        df.groupby(["LocationID", "zone", "borough"])["count"]
        .sum()
        .reset_index()
    )

    zone_agg["log_count"] = np.log10(zone_agg["count"].clip(lower=1))

    fig_map = px.choropleth_map(
        zone_agg,
        geojson=geojson,
        locations="LocationID",
        featureidkey="properties.LocationID",
        color="log_count",
        color_continuous_scale="YlOrRd",
        hover_name="zone",
        hover_data={"borough": True, "count": ":,", "LocationID": False, "log_count": False},
        zoom=10,
        center={"lat": 40.73, "lon": -73.98},
        map_style="open-street-map",
        title="2014 Pickups by Zone (choropleth — full dataset)",
        opacity=0.7,
        labels={"log_count": "Pickups (log\u2081\u2080)", "count": "Pickups"},
    )

    if centers_df is not None:
        fig_map.add_trace(go.Scattermap(
            lat=centers_df["lat"],
            lon=centers_df["lon"],
            mode="markers",
            marker=dict(size=14, color="#00BFFF", opacity=0.9),
            name="Cluster Centers",
            hovertemplate=(
                "<b>Cluster %{text}</b><br>"
                "Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
            ),
            text=centers_df["cluster"].astype(str),
        ))

else:
    zone_agg = (
        df.dropna(subset=["zone"])
        .rename(columns={"locationID": "LocationID"})
        .groupby(["LocationID", "zone", "borough"])["count"]
        .sum()
        .reset_index()
    )

    zone_agg["log_count"] = np.log10(zone_agg["count"].clip(lower=1))

    fig_map = px.choropleth_map(
        zone_agg,
        geojson=geojson,
        locations="LocationID",
        featureidkey="properties.LocationID",
        color="log_count",
        color_continuous_scale="YlOrRd",
        hover_name="zone",
        hover_data={"borough": True, "count": ":,", "LocationID": False, "log_count": False},
        zoom=10,
        center={"lat": 40.73, "lon": -73.98},
        map_style="open-street-map",
        title="2015 Pickups by Zone (choropleth — full dataset)",
        opacity=0.7,
        labels={"log_count": "Pickups (log\u2081\u2080)", "count": "Pickups"},
    )

fig_map.update_layout(height=550, margin={"l": 0, "r": 0, "t": 30, "b": 0})
st.plotly_chart(fig_map, width="stretch")

st.divider()

st.subheader("Temporal Patterns")

hourly = df.groupby("hour")["count"].sum().reset_index()
daily = df.groupby("day_of_week")["count"].sum().reset_index()

daily["day_name"] = daily["day_of_week"].map(lambda x: DAY_NAMES[x])

col_h, col_d = st.columns(2)

with col_h:
    fig_h = px.bar(
        hourly,
        x="hour",
        y="count",
        labels={"hour": "Hour of Day", "count": "Pickups"},
        title="Pickups by Hour",
        color_discrete_sequence=["#636EFA"],
    )
    fig_h.update_layout(height=320)
    st.plotly_chart(fig_h, width="stretch")

with col_d:
    fig_d = px.bar(
        daily,
        x="day_name",
        y="count",
        labels={"day_name": "Day", "count": "Pickups"},
        title="Pickups by Day of Week",
        color_discrete_sequence=["#EF553B"],
        category_orders={"day_name": DAY_NAMES},
    )
    fig_d.update_layout(height=320)
    st.plotly_chart(fig_d, width="stretch")

st.subheader("Heatmap: Hour × Day of Week")

heat_data = df.groupby(["hour", "day_of_week"])["count"].sum().reset_index()

pivot = (
    heat_data.pivot(index="hour", columns="day_of_week", values="count")
    .fillna(0)
    .reindex(columns=list(range(7)), fill_value=0)
)
pivot.columns = [DAY_NAMES[c] for c in pivot.columns]

fig_heat = px.imshow(
    pivot,
    labels={"x": "Day of Week", "y": "Hour of Day", "color": "Pickups"},
    color_continuous_scale="YlOrRd",
    aspect="auto",
    title="Pickup Intensity: Hour × Day",
)
fig_heat.update_layout(height=420)
st.plotly_chart(fig_heat, width="stretch")

st.divider()

st.subheader("Top Hot-Zones")

if year == 2014:
    if "zone" in df.columns:
        top_zones = (
            df.groupby(["zone", "borough"])["count"]
            .sum()
            .reset_index(name="Pickups")
            .sort_values("Pickups", ascending=False)
            .head(15)
            .rename(columns={"zone": "Zone", "borough": "Borough"})
        )
        top_zones["% of Total"] = (top_zones["Pickups"] / total_pickups * 100).round(2)
        st.dataframe(top_zones[["Zone", "Borough", "Pickups", "% of Total"]], hide_index=True, width="stretch")
    else:
        st.info("Zone information not available in 2014 data.")
else:
    top_zones = (
        df.groupby(["zone", "borough"])["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
        .head(15)
        .rename(columns={"zone": "Zone", "borough": "Borough", "count": "Pickups"})
    )
    top_zones["% of Total"] = (top_zones["Pickups"] / total_pickups * 100).round(2)
    st.dataframe(top_zones[["Zone", "Borough", "Pickups", "% of Total"]], hide_index=True, width="stretch")
