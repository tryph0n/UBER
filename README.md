# Uber NYC Hot-Zones Clustering Analysis

> Identify geographic hot-zones for Uber pickups in New York City using unsupervised ML clustering. Analyzes 4.5M+ GPS-based pickups (2014) and 14M+ zone-based pickups (2015) to recommend optimal driver positioning by time of day. Optimal cluster count (k=11) determined via geometric elbow detection, confirmed by silhouette score.

## Key Results

- **KMeans (k=11) outperforms DBSCAN** for NYC pickup clustering -- DBSCAN's density-based approach fails on Manhattan's continuous gradient, producing a single dominant cluster with 70%+ of all points. Optimal k selected by geometric elbow detection and confirmed by silhouette score.
- **Uber grew ~4x between 2014 and 2015** (April-June overlap comparison), with outer boroughs gaining pickup share.
- **Hot-zones shift predictably by time of day:** business districts dominate morning/evening rush hours, nightlife/entertainment districts (Union Sq, TriBeCa, Williamsburg, Park Slope) lead late night.
- **Temporal patterns are stable across years:** hourly and daily pickup distributions remain consistent despite the ~4x volume growth.

## Tech Stack

| Category           | Technology                           |
|--------------------|--------------------------------------|
| Analysis           | pandas, NumPy, scikit-learn, geopandas |
| Clustering         | KMeans, DBSCAN (scikit-learn)        |
| Visualization      | Plotly                               |
| Dashboard          | Streamlit                            |
| Data format        | PyArrow / Parquet                    |
| Package management | uv                                   |
| Python             | 3.10+                                |

## Installation

```bash
git clone <repository-url>
cd UBER

# Install all dependencies
uv sync

# Required for Plotly PNG export in notebooks
uv run plotly_get_chrome
```

## Usage

### 1. Run notebooks in order (each exports data for the next)

```bash
# Step 1: Data loading, preprocessing, EDA
uv run jupyter notebook 01-Uber_Hot_Zones_Analysis.ipynb

# Step 2: KMeans vs DBSCAN clustering comparison
uv run jupyter notebook 02-Clustering_Analysis.ipynb

# Step 3: Hot-zone patterns and year-over-year analysis
uv run jupyter notebook 03-Hot_Zone_Analysis.ipynb
```

### 2. Launch the dashboard

```bash
uv run streamlit run dashboard/0_Overview.py
```

The dashboard includes four pages: Hot Zone Explorer (interactive map + filters), Year Comparison (2014 vs 2015), Methodology (approach and limitations), and Glossary (key terms and definitions).

## Data

Raw data (~730MB total) is **not included in the repository**. You need to download it manually.

### Uber trip data

Download from the [FiveThirtyEight Uber TLC FOIL dataset](https://github.com/fivethirtyeight/uber-tlc-foil-response) and place files in `data/input/uber-trip-data/uber-trip-data/`:

- **2014 (6 files):** `uber-raw-data-{apr,may,jun,jul,aug,sep}14.csv` -- GPS coordinates per pickup
- **2015 (1 file):** `uber-raw-data-janjune-15.csv` -- zone IDs per pickup
- **Lookup:** `taxi-zone-lookup.csv`

### NYC taxi zones shapefile

```bash
mkdir -p data/input/taxi_zones
curl -L -o data/input/taxi_zones/taxi_zones.zip \
  https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
unzip data/input/taxi_zones/taxi_zones.zip -d data/input/taxi_zones/
rm data/input/taxi_zones/taxi_zones.zip
```
