"""Preprocessing functions for Uber pickup data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import NYC_BOUNDS, NYC_LAT_CENTER

import logging

logger = logging.getLogger(__name__)


def parse_timestamps_2014(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 2014 timestamps and extract temporal features.

    Adds columns: datetime, hour, day_of_week, month, is_weekday, year.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["Date/Time"], format="%m/%d/%Y %H:%M:%S")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekday"] = df["day_of_week"] < 5
    df["year"] = 2014
    return df


def parse_timestamps_2015(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 2015 timestamps and extract temporal features.

    Modifies the input DataFrame in place to avoid doubling memory on large datasets.
    Adds columns: datetime, hour, day_of_week, month, is_weekday, year.
    """
    df["datetime"] = pd.to_datetime(df["Pickup_date"], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekday"] = df["day_of_week"] < 5
    df["year"] = 2015
    return df


def filter_nyc_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records outside NYC geographic bounds. 2014 data only.

    Uses bounds defined in config.NYC_BOUNDS.
    """
    before = len(df)
    mask = (
        df["Lat"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        & df["Lon"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
    )
    result = df[mask].copy()
    removed = before - len(result)
    pct = (removed / before * 100) if before > 0 else 0.0
    logger.info(
        f"Geographic filter: {before:,} -> {len(result):,} "
        f"(removed {removed:,}, {pct:.2f}%)"
    )
    return result


def normalize_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Cosine-corrected min-max normalization of Lat/Lon for clustering.

    Applies cosine correction to longitude to compensate for latitude-dependent
    distortion: at NYC latitude (~40.7N), 1 deg lon = 84 km vs 1 deg lat = 111 km.
    Without correction, KMeans clusters are stretched ~24% along the N-S axis.

    Note: Modifies df in-place by adding lat_norm/lon_norm columns.
    """
    cos_factor = np.cos(np.radians(NYC_LAT_CENTER))
    coords = pd.DataFrame({
        "Lat": df["Lat"].values,
        "Lon_corr": df["Lon"].values * cos_factor,
    })
    scaler = MinMaxScaler()
    df[["lat_norm", "lon_norm"]] = scaler.fit_transform(coords)
    return df, scaler


def remap_duplicate_zones(df: pd.DataFrame, id_col: str = "locationID") -> pd.DataFrame:
    """Remap duplicate zone IDs to their canonical LocationID.

    The TLC lookup table assigns separate IDs to sub-polygons that share the
    same LocationID in the shapefile: zone 57 duplicates 56 (Corona, Queens),
    and zones 104-105 duplicate 103 (Governor's Island/Ellis/Liberty, Manhattan).
    Remapping prevents double/triple counting in downstream aggregations.
    """
    dupes = {57: 56, 104: 103, 105: 103}
    df = df.copy()
    df[id_col] = df[id_col].replace(dupes)
    logger.info(f"Remapped {len(dupes)} duplicate zone IDs in column '{id_col}'")
    return df


def map_2015_zones(df_2015: pd.DataFrame, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    """Map 2015 locationID to zone names and boroughs via lookup table."""
    df_2015 = remap_duplicate_zones(df_2015, id_col="locationID")

    # Zones 264/265 are unmapped TLC entries with no real geographic location
    unknown_mask = df_2015["locationID"].isin([264, 265])
    n_unknown = unknown_mask.sum()
    pct_unknown = (n_unknown / len(df_2015) * 100) if len(df_2015) > 0 else 0.0
    df_2015 = df_2015[~unknown_mask]
    logger.info(
        f"Dropped {n_unknown:,} unknown-zone rows (locationID 264/265, {pct_unknown:.3f}%)"
    )

    result = df_2015.merge(
        zone_lookup, left_on="locationID", right_on="LocationID", how="left"
    )
    result = result.rename(columns={"Zone": "zone", "Borough": "borough"})
    mapped = result["zone"].notna().sum()
    pct = (mapped / len(result) * 100) if len(result) > 0 else 0.0
    logger.info(
        f"2015 zone mapping: {mapped:,} / {len(result):,} records mapped "
        f"({pct:.1f}%)"
    )
    return result
