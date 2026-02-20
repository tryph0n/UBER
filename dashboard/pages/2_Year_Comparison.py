"""Page 2 — Year-over-Year Comparison: zones, boroughs, temporal overlay."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "output"


@st.cache_data
def load_zone_comparison() -> pd.DataFrame | None:
    """Load zone-level year-over-year comparison data."""
    path = DATA_DIR / "zone_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_2014() -> pd.DataFrame | None:
    """Load 2014 aggregated data for temporal overlay."""
    path = DATA_DIR / "dashboard_2014_aggregated.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_2015() -> pd.DataFrame | None:
    """Load 2015 aggregated for temporal overlay."""
    path = DATA_DIR / "dashboard_2015_aggregated.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


st.title("Year-over-Year Comparison")
st.caption("Comparing April–June 2014 vs April–June 2015 (overlap period). Zone comparison uses the full datasets filtered to the 3 overlapping months.")

zone_df = load_zone_comparison()
if zone_df is None:
    st.error("Data not found. Run notebooks first. (zone_comparison.csv)")
    st.stop()

raw_2014 = load_2014()
raw_2015 = load_2015()

st.subheader("Top 20 Zones: 2014 vs 2015 (April–June)")

top20 = zone_df.dropna(subset=["count_2014", "count_2015"]).copy()
top20["max_count"] = top20[["count_2014", "count_2015"]].max(axis=1)
top20 = top20.nlargest(20, "max_count").sort_values("max_count", ascending=True)

fig_top = go.Figure()
fig_top.add_trace(go.Bar(
    y=top20["zone"],
    x=top20["count_2014"],
    name="2014",
    orientation="h",
    marker_color="#636EFA",
))
fig_top.add_trace(go.Bar(
    y=top20["zone"],
    x=top20["count_2015"],
    name="2015",
    orientation="h",
    marker_color="#EF553B",
))
fig_top.update_layout(
    barmode="group",
    height=560,
    margin={"l": 0, "r": 20, "t": 10, "b": 0},
    xaxis_title="Pickup Count",
    yaxis_title="Zone",
    legend={"orientation": "h", "y": 1.02},
)
st.plotly_chart(fig_top, width="stretch")
st.caption(
    "Most top-10 zones are stable across both years, confirming consistent "
    "high-demand areas. Manhattan zones dominate the ranking in both periods."
)

st.divider()

st.subheader("Zone Growth / Decline Table")
st.warning(
    "These growth figures are approximate. The 2014 dataset covers April to September, "
    "while the 2015 dataset covers January to June — only April, May, and June appear "
    "in both. The table compares those three shared months, but the raw pickup counts "
    "were collected differently between years (GPS coordinates in 2014 vs. pre-grouped "
    "zone totals in 2015), so extreme values like -99% for Newark reflect a data "
    "collection difference, not an actual drop in demand."
)

col_sort, col_n = st.columns([2, 1])
with col_sort:
    sort_by = st.selectbox(
        "Sort by",
        ["growth_pct", "count_2015", "count_2014", "rank_change"],
        format_func=lambda x: {
            "growth_pct": "Growth %",
            "count_2015": "2015 Count",
            "count_2014": "2014 Count",
            "rank_change": "Rank Change",
        }[x],
    )
with col_n:
    ascending = st.checkbox("Ascending", value=False)

table = (
    zone_df[["zone", "borough", "count_2014", "count_2015", "growth_pct", "rank_2014", "rank_2015", "rank_change"]]
    .dropna(subset=["growth_pct"])
    .sort_values(sort_by, ascending=ascending)
    .rename(columns={
        "zone": "Zone",
        "borough": "Borough",
        "count_2014": "Count 2014",
        "count_2015": "Count 2015",
        "growth_pct": "Growth %",
        "rank_2014": "Rank 2014",
        "rank_2015": "Rank 2015",
        "rank_change": "Rank Change",
    })
)
table["Growth %"] = table["Growth %"].round(1)

st.dataframe(table, width="stretch", hide_index=True)

st.divider()

st.subheader("Borough-level Evolution")

borough_2014 = (
    zone_df.groupby("borough")["count_2014"].sum().reset_index()
    .rename(columns={"count_2014": "count"})
    .assign(year="2014")
)
borough_2015 = (
    zone_df.groupby("borough")["count_2015"].sum().reset_index()
    .rename(columns={"count_2015": "count"})
    .assign(year="2015")
)
borough_combined = pd.concat([borough_2014, borough_2015], ignore_index=True)

fig_boro = px.bar(
    borough_combined,
    x="borough",
    y="count",
    color="year",
    barmode="group",
    labels={"borough": "Borough", "count": "Pickup Count", "year": "Year"},
    title="Pickup Count by Borough: 2014 vs 2015",
    color_discrete_sequence=["#636EFA", "#EF553B"],
)
fig_boro.update_layout(height=380, legend={"orientation": "h", "y": 1.02})
st.plotly_chart(fig_boro, width="stretch")
st.caption(
    "Manhattan accounts for the vast majority of pickups in both years. "
    "Borough-level proportions remain stable, suggesting structural demand "
    "patterns rather than year-specific anomalies."
)

st.divider()

st.subheader("Hourly Distribution Overlay: 2014 vs 2015")

if raw_2014 is not None and raw_2015 is not None:
    # Restrict to Apr-Jun overlap for fair comparison
    overlap_months = [4, 5, 6]

    hourly_2014 = (
        raw_2014[raw_2014["month"].isin(overlap_months)]
        .groupby("hour")["count"]
        .sum()
        .reset_index()
    )
    hourly_2015 = (
        raw_2015[raw_2015["month"].isin(overlap_months)]
        .groupby("hour")["count"]
        .sum()
        .reset_index()
    )

    hourly_2014["pct"] = hourly_2014["count"] / hourly_2014["count"].sum() * 100
    hourly_2015["pct"] = hourly_2015["count"] / hourly_2015["count"].sum() * 100

    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Scatter(
        x=hourly_2014["hour"],
        y=hourly_2014["pct"],
        mode="lines+markers",
        name="2014",
        line={"color": "#636EFA", "width": 2},
        marker={"size": 6},
    ))
    fig_overlay.add_trace(go.Scatter(
        x=hourly_2015["hour"],
        y=hourly_2015["pct"],
        mode="lines+markers",
        name="2015",
        line={"color": "#EF553B", "width": 2},
        marker={"size": 6},
    ))
    fig_overlay.update_layout(
        height=380,
        xaxis_title="Hour of Day",
        yaxis_title="% of Daily Pickups",
        title="Hourly Pickup % Distribution (Apr–Jun overlap, proportions)",
        legend={"orientation": "h", "y": 1.02},
    )
    st.plotly_chart(fig_overlay, width="stretch")
    st.caption(
        "Percentages, not absolute counts — both years use zone-aggregated data. "
        "Absolute volumes are not directly comparable due to differing coverage periods."
    )
else:
    st.info("Hourly overlay unavailable: 2014 or 2015 data files not found.")
