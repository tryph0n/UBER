"""Spatial operations for taxi zone mapping."""
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


def spatial_join_2014(df_2014: pd.DataFrame, zones_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map 2014 GPS coordinates to taxi zones via spatial join.

    Converts pickup lat/lon to Point geometries and joins with taxi zone
    polygons. Adds columns: LocationID, zone, borough.
    """
    gdf = gpd.GeoDataFrame(
        df_2014,
        geometry=gpd.points_from_xy(df_2014["Lon"], df_2014["Lat"]),
        crs="EPSG:4326",
    )
    zones_4326 = zones_gdf.to_crs("EPSG:4326")
    joined = gpd.sjoin(
        gdf,
        zones_4326[["LocationID", "zone", "borough", "geometry"]],
        how="left",
        predicate="within",
    )
    cols_to_drop = ["geometry"] + [c for c in joined.columns if c.startswith("index_")]
    result = pd.DataFrame(joined.drop(columns=cols_to_drop))
    mapped = result["LocationID"].notna().sum()
    if len(result) > 0:
        pct = mapped / len(result) * 100
        logger.info(
            f"2014 spatial join: {mapped:,} / {len(result):,} records mapped "
            f"({pct:.1f}%)"
        )
    else:
        logger.warning("2014 spatial join: empty result, no records to map")
    return result


def compute_zone_centroids(zones_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute centroid lat/lon for each taxi zone.

    Returns DataFrame with: LocationID, zone, borough, centroid_lat, centroid_lon.
    Used for mapping 2015 zone-based data to geographic coordinates.
    """
    # Dissolve multi-polygon entries so each LocationID yields exactly one row.
    # Without this, LocationID 56 (2 polygons) and 103 (3 polygons) produce
    # duplicate centroids, inflating downstream CSV row counts from 260 to 263.
    zones_gdf = zones_gdf.dissolve(by="LocationID").reset_index()
    zones_projected = zones_gdf.to_crs("EPSG:2263")
    centroid_points = zones_projected.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame(geometry=centroid_points, crs="EPSG:2263").to_crs("EPSG:4326")
    result = zones_gdf[["LocationID", "zone", "borough"]].copy()
    result["centroid_lat"] = centroid_gdf.geometry.y
    result["centroid_lon"] = centroid_gdf.geometry.x
    return result
