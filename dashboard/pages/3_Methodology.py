"""Page 3 — Methodology: data sources, clustering rationale, limitations."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "output"


@st.cache_data
def load_sample():
    """Load 200k GPS sample with cluster assignments."""
    path = DATA_DIR / "dashboard_2014_sample.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_centers():
    """Load KMeans cluster centers."""
    path = DATA_DIR / "kmeans_hotzone_centers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_evaluation_metrics():
    """Load pre-computed KMeans evaluation metrics."""
    path = DATA_DIR / "kmeans_evaluation_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


st.title("Methodology")

st.markdown("""\
## Data Sources and Scope

This project uses two distinct Uber datasets for New York City:

- **2014 dataset**: GPS-level pickup records from April through September 2014.
  Each row contains a precise latitude/longitude coordinate and a timestamp.
  This granularity allows direct spatial clustering.

- **2015 dataset**: Aggregated pickup counts by TLC taxi zone, covering January
  through June 2015. Coordinates are not available at the individual pickup level;
  instead, each record maps to a predefined NYC taxi zone polygon.

The two datasets differ in structure, which means absolute pickup volumes cannot
be compared directly. Year-over-year analysis is performed on proportional
distributions and zone-level rankings.

---

## Clustering Approach: KMeans vs DBSCAN

Two candidate algorithms were evaluated for identifying hot-zones from the 2014
GPS coordinates: KMeans and DBSCAN.

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) was
considered first because it can discover clusters of arbitrary shape and does
not require a predefined number of clusters. However, in practice it presented
several drawbacks for this dataset:

- The two hyperparameters (`eps` and `min_samples`) are sensitive and difficult
  to tune at NYC's density gradient — a single `eps` value that works in dense
  Manhattan produces oversized clusters in outer boroughs.
- Runtime scales poorly with dataset size (200k+ points), requiring significant
  subsampling that biases density estimates.
- The resulting cluster boundaries did not align well with operational hot-zone
  definitions (interpretable geographic centers).
- A **k-distance plot** (sorted k-nearest-neighbor distances for multiple values
  of min_samples) confirms this: the curves show a smooth, gradual increase with
  no sharp elbow. NYC pickup density forms a continuous gradient — there is no
  natural density threshold to separate "cluster" from "noise", which is the
  fundamental assumption DBSCAN requires.

**KMeans** was ultimately chosen because:

- It produces compact, interpretable clusters with explicit centroids.
- Runtime is tractable on the full dataset.
- The assumption of approximately spherical clusters is reasonable at the
  resolution of NYC neighborhoods.
- Centroids can be directly used as "hot-zone anchors" for operational purposes.

---

## Optimal k Selection

The number of clusters k was chosen using a multi-criteria approach:

1. **Elbow method** (primary): Geometric elbow detection identifies the inflection
   point on the inertia (WCSS) curve — where adding more clusters yields
   diminishing returns. Mathematically, it finds the point farthest from the
   line connecting the first and last values on the normalized inertia curve
   (maximum-distance-to-diagonal method).

2. **Silhouette score** (confirmation): Average silhouette was computed across
   the full k range (3-50). Note that silhouette tends to favor low k
   (fewer clusters = easier separation). At k=7 (max silhouette), the top two
   clusters risk absorbing a disproportionate share of pickups — the same
   imbalance problem as DBSCAN.

3. **Business constraint**: Clusters must be actionable for driver positioning —
   granular enough to distinguish neighborhoods (airports, Midtown, Brooklyn),
   compact enough to memorize and act on.

The elbow method identified k=11, confirmed by reasonable silhouette values
and practical cluster sizes (from 41k to 1.13M pickups per cluster).
""")

metrics_df = load_evaluation_metrics()

if metrics_df is not None:
    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        fig_elbow = px.line(
            metrics_df,
            x="k",
            y="inertia",
            title="Elbow Method: Inertia vs Number of Clusters",
            labels={"k": "Number of clusters (k)", "inertia": "Inertia (WCSS)"},
        )
        fig_elbow.add_vline(
            x=11, line_dash="dash", line_color="red", annotation_text="k=11"
        )
        st.plotly_chart(fig_elbow, width="stretch")

    with col_sil:
        fig_sil = px.line(
            metrics_df,
            x="k",
            y="silhouette",
            title="Silhouette Score vs Number of Clusters",
            labels={"k": "Number of clusters (k)", "silhouette": "Average Silhouette Score"},
        )
        fig_sil.add_vline(
            x=11, line_dash="dash", line_color="red", annotation_text="k=11"
        )
        st.plotly_chart(fig_sil, width="stretch")

    st.caption(
        "Computed on a 200,000-pickup random sample. "
        "The elbow at k=11 (dashed line) marks the point of diminishing returns in inertia reduction."
    )
else:
    st.info("Evaluation metrics not found. Run notebook 03-Clustering_Analysis to generate kmeans_evaluation_metrics.csv.")

st.markdown("""\
---

## Why Spatial-Only Clustering?

KMeans is applied to **latitude/longitude only** — no temporal features are included.
This is a deliberate design choice, not an oversight.

**1. KMeans answers "where", not "when."** It identifies stable geographic
concentrations of demand. An airport or business district remains a hot-zone
regardless of the hour — only the intensity changes, not the spatial structure.

**2. Temporal analysis is performed post-clustering.** Once each pickup is assigned
to a spatial cluster, the analysis produces: heatmaps of cluster activity by hour of day
and day of week, top-5 most active zones per time slot (morning rush, evening rush,
late night), and the dashboard exposes interactive hour/day filters on the Overview page.

**3. Adding time as a KMeans feature would degrade results:**
- Running 24 separate KMeans (one per hour) produces 24 inconsistent cluster maps —
  unstable boundaries that are unusable for driver positioning.
- Including hour as a numeric feature is problematic: hour is cyclic (23:00 is close
  to 00:00, but numerically distant), and mixing a temporal dimension with spatial
  coordinates breaks geographic coherence — pickups at the same location but different
  hours would land in different clusters.

**4. This follows the standard approach** in transportation demand literature: define
stable geographic zones first, then model temporal demand patterns within each zone.
NYC's own TLC taxi zone system (263 fixed polygons) follows the same principle.
""")

st.markdown("## Cluster Visualization")
st.markdown("""\
The map below shows a random sample of **200,000 pickups** (out of 4.5 million total)
from the 2014 dataset. Each point is colored by its KMeans cluster assignment.
Large blue markers indicate the computed cluster centroids — the geographic
"hot-zone anchors" that drivers can use for positioning.

The sample was drawn uniformly at random (`random_state=42`) from the full dataset,
preserving the spatial distribution of pickups across all zones and time periods.
""")

sample_df = load_sample()
centers_df = load_centers()

if sample_df is not None and centers_df is not None:
    sample_df["cluster"] = sample_df["cluster"].astype(str)

    fig_clusters = px.scatter_map(
        sample_df,
        lat="Lat",
        lon="Lon",
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set3,
        opacity=0.3,
        zoom=10,
        center={"lat": 40.73, "lon": -73.98},
        map_style="open-street-map",
        title="KMeans Cluster Assignments — 200,000 pickups (random sample of 4.5M)",
        labels={"cluster": "Cluster"},
    )
    fig_clusters.update_traces(marker_size=2)

    fig_clusters.add_trace(go.Scattermap(
        lat=centers_df["lat"],
        lon=centers_df["lon"],
        mode="markers",
        marker=dict(size=16, color="#00BFFF", opacity=0.9),
        name="Centroids",
        hovertemplate="<b>Cluster %{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
        text=centers_df["cluster"].astype(str),
    ))

    fig_clusters.update_layout(height=600, margin={"l": 0, "r": 0, "t": 30, "b": 0})
    st.plotly_chart(fig_clusters, width="stretch")
else:
    st.info("Data file not found. Re-run the analysis pipeline to generate dashboard_2014_sample.csv.")

@st.cache_data
def load_profiles():
    """Load cluster profile summaries."""
    path = DATA_DIR / "cluster_profiles.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _fmt_pickups(n: int) -> str:
    """Format pickup count: >=1M → '1.13M', >=1k → '876k', else raw."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{round(n / 1_000)}k"
    return str(n)


profiles = load_profiles()
if profiles is not None:
    profiles = profiles.sort_values("total_pickups", ascending=False)
    k = len(profiles)

    rows = "| Cluster | Area | Borough | Pickups | Share |\n"
    rows += "|---------|------|---------|---------|-------|\n"
    for _, r in profiles.iterrows():
        rows += (
            f"| {int(r['cluster'])} "
            f"| {r['dominant_zone']} "
            f"| {r['dominant_borough']} "
            f"| {_fmt_pickups(int(r['total_pickups']))} "
            f"| {r['pct_of_total']:.1f}% |\n"
        )

    st.markdown(f"""\
### What Do These Clusters Represent?

Each cluster corresponds to a geographic area where Uber pickups concentrate.
KMeans partitions the 4.5 million GPS coordinates into k={k} groups, minimizing
the within-cluster distance to each centroid. The result:

{rows}
These clusters are not arbitrary — they emerge from the spatial distribution of
millions of real pickups. The centroids serve as actionable "hot-zone anchors"
for driver positioning strategies.
""")
else:
    st.info("Data file not found. Re-run the analysis pipeline to generate cluster_profiles.csv.")

st.markdown("""\
---

## Zone Mapping (2014 GPS to TLC Zones)

To enable year-over-year comparison, 2014 GPS coordinates were mapped to the
same TLC taxi zone system used in 2015. The mapping was performed via a
**spatial join**: each pickup point was matched to the TLC zone polygon it falls
within, using the official NYC TLC shapefile.

Points that fell outside all zone polygons (e.g., in water bodies or outside
city limits) were excluded. The resulting `LocationID` column makes 2014 and
2015 records directly comparable at zone level.

---

## Year-over-Year Methodology

Because the 2014 dataset is GPS-level and the 2015 dataset is already aggregated
by zone, a direct absolute count comparison would be misleading. The analysis
instead uses:

- **Proportional distributions**: hourly and daily patterns are expressed as
  percentages of total pickups within each year, removing the effect of dataset
  size differences.

- **Zone-level rankings**: zones are ranked by pickup count within each year.
  Rank change (rank_2014 - rank_2015) captures relative shifts in demand without
  depending on absolute volumes.

- **Growth percentage**: for zones present in both years, `growth_pct` is
  computed as `(count_2015 - count_2014) / count_2014 x 100`. These figures
  should be interpreted cautiously given the dataset asymmetry.

- **Overlap period**: when computing temporal overlays, only April-June months
  are used, since that period is covered by both datasets.

---

## Limitations

- **Dataset asymmetry**: 2014 is raw GPS, 2015 is pre-aggregated. Any comparison
  of absolute volumes conflates actual demand changes with data collection
  differences.

- **Seasonal mismatch**: 2014 covers April-September (warm months); 2015 covers
  January-June (includes winter). Seasonal patterns differ and should not be
  interpreted as demand trends without controlling for season.

- **KMeans assumptions**: clusters are assumed approximately spherical and of
  similar size. NYC demand is highly heterogeneous; Manhattan clusters are
  denser and smaller than outer-borough clusters by construction.

- **Spatial join accuracy**: zone boundaries are administrative polygons. A pickup
  at the exact edge of two zones may be assigned arbitrarily depending on
  polygon precision.

- **Sampling**: the 2014 sample used in the dashboard (200k rows) is a
  representative subsample of the full dataset. Cluster shapes are stable, but
  exact counts should not be treated as census-level.
""")

st.markdown("""\
---

## Industrial Context: Uber's H3 Grid

In production, Uber does not use KMeans or any traditional clustering algorithm
for demand zone detection. Instead, Uber developed **H3** — an open-source
hierarchical hexagonal spatial indexing system
([Uber Engineering Blog, June 2018](https://www.uber.com/blog/h3/)).

H3 partitions the globe into hexagons at 16 resolution levels. Each real-time
event (pickup request, driver position) is mapped to a hexagonal cell index.
Surge pricing is computed per hexagon by measuring local supply vs demand —
no clustering step required.

Hexagons are preferred over squares because neighboring cells are equidistant
(6 neighbors at identical distance vs 4+4 at two distances for squares),
reducing quantization error when users move across cell boundaries.

The KMeans approach used in this project is a valid academic proxy: it identifies
the same high-demand areas that H3 cells would capture, and produces interpretable
centroids that serve as "hot-zone anchors" for driver positioning. For a production
system at Uber's scale, H3's constant-time grid lookup is far more
efficient than real-time clustering.

*Source: [H3: Uber's Hexagonal Hierarchical Spatial Index](https://www.uber.com/blog/h3/) — Uber Engineering Blog, June 27, 2018.*
""")
