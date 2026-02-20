"""Reusable Plotly chart factories for notebooks and dashboard."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import DAY_NAMES, RANDOM_STATE


def create_density_map(
    df: pd.DataFrame, sample_n: int = 20_000, title: str = "Pickup Density"
) -> go.Figure:
    """Geographic density heatmap of pickups. Samples data for performance."""
    sample = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_STATE)
    fig = px.density_mapbox(
        sample,
        lat="Lat",
        lon="Lon",
        radius=5,
        zoom=10,
        center={"lat": 40.73, "lon": -73.98},
        mapbox_style="open-street-map",
        title=title,
        height=600,
    )
    return fig


def create_pickup_map(
    df: pd.DataFrame,
    color_col: str = "cluster",
    centers: pd.DataFrame | None = None,
    sample_n: int = 10_000,
    title: str = "Pickup Clusters",
) -> go.Figure:
    """Scatter mapbox of pickups colored by a column, with optional cluster centers."""
    sample = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_STATE)

    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=sample["Lat"],
            lon=sample["Lon"],
            mode="markers",
            marker=dict(size=4, opacity=0.5, color=sample[color_col], colorscale="Viridis"),
            name="Pickups",
            hovertemplate="Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>" + color_col + ": %{marker.color}<extra></extra>",
        )
    )

    if centers is not None:
        fig.add_trace(
            go.Scattermapbox(
                lat=centers["lat"],
                lon=centers["lon"],
                mode="markers",
                marker=dict(
                    size=centers.get("size", pd.Series([10] * len(centers))).clip(5, 30),
                    color="red",
                    opacity=0.8,
                ),
                name="Cluster Centers",
                text=centers.index.astype(str),
                hovertemplate="Center %{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=40.73, lon=-73.98), zoom=10),
        title=title,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def create_hourly_bar(df: pd.DataFrame, title: str = "Pickups by Hour") -> go.Figure:
    """Bar chart of pickup counts by hour of day."""
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig = px.bar(hourly, x="hour", y="count", title=title, labels={"hour": "Hour", "count": "Pickups"}, height=350)
    fig.update_xaxes(dtick=1)
    return fig


def create_daily_bar(df: pd.DataFrame, title: str = "Pickups by Day of Week") -> go.Figure:
    """Bar chart of pickup counts by day of week."""
    daily = df.groupby("day_of_week").size().reset_index(name="count")
    daily["day_name"] = daily["day_of_week"].map(lambda x: DAY_NAMES[x])
    fig = px.bar(daily, x="day_name", y="count", title=title, labels={"day_name": "Day", "count": "Pickups"}, height=350)
    return fig


def create_heatmap_hour_day(df: pd.DataFrame, title: str = "Pickups: Hour x Day") -> go.Figure:
    """Heatmap of pickups by hour and day of week."""
    pivot = df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
    matrix = pivot.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    fig = px.imshow(
        matrix,
        labels=dict(x="Hour", y="Day of Week", color="Pickups"),
        x=[str(h) for h in range(24)],
        y=DAY_NAMES,
        color_continuous_scale="YlOrRd",
        title=title,
        height=400,
    )
    return fig


def create_cluster_metrics_plot(metrics_df: pd.DataFrame) -> go.Figure:
    """Four-panel plot: silhouette, Calinski-Harabasz, inertia, and average cluster size vs k."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Silhouette Score", "Calinski-Harabasz Index", "Inertia (Elbow)", "Avg Cluster Size"),
    )

    fig.add_trace(
        go.Scatter(x=metrics_df["k"], y=metrics_df["silhouette"], mode="lines+markers", name="Silhouette"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["k"], y=metrics_df["calinski_harabasz"], mode="lines+markers", name="CH Index"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["k"], y=metrics_df["inertia"], mode="lines+markers", name="Inertia"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=metrics_df["k"], y=metrics_df["avg_cluster_size"], mode="lines+markers", name="Avg Size"),
        row=2, col=2,
    )

    fig.update_layout(height=600, title_text="KMeans Hyperparameter Analysis", showlegend=False)
    return fig


def create_comparison_bars(
    df_2014: pd.DataFrame,
    df_2015: pd.DataFrame,
    group_col: str,
    title: str,
    use_pct: bool = True,
) -> go.Figure:
    """Grouped bar chart comparing 2014 vs 2015 by a grouping column."""
    g14 = df_2014.groupby(group_col).size().reset_index(name="count")
    g14["year"] = "2014"
    g15 = df_2015.groupby(group_col).size().reset_index(name="count")
    g15["year"] = "2015"

    if use_pct:
        g14["value"] = g14["count"] / g14["count"].sum() * 100
        g15["value"] = g15["count"] / g15["count"].sum() * 100
        y_label = "% of Total"
    else:
        g14["value"] = g14["count"]
        g15["value"] = g15["count"]
        y_label = "Pickups"

    combined = pd.concat([g14, g15])
    fig = px.bar(
        combined,
        x=group_col,
        y="value",
        color="year",
        barmode="group",
        title=title,
        labels={group_col: group_col.replace("_", " ").title(), "value": y_label},
        color_discrete_map={"2014": "#636EFA", "2015": "#EF553B"},
        height=400,
    )
    return fig
