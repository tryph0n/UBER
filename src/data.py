"""Data loading functions for Uber pickup datasets."""
import pandas as pd
import geopandas as gpd
from .config import DATA_DIR, ZONES_DIR, CSV_2014_PATTERN, CSV_2015_FILENAME, ZONE_LOOKUP_FILENAME, ZONE_SHAPEFILE

import logging

logger = logging.getLogger(__name__)


def load_2014_data() -> pd.DataFrame:
    """Load and concatenate all 2014 CSV files (Apr-Sep).

    Returns DataFrame with columns: Date/Time, Lat, Lon, Base.
    ~4.5M rows total.
    """
    csv_files = sorted(DATA_DIR.glob(CSV_2014_PATTERN))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matching '{CSV_2014_PATTERN}' in {DATA_DIR}"
        )
    dtypes = {
        "Lat": "float32",
        "Lon": "float32",
        "Base": "category",
    }
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=dtypes)
        logger.debug(f"{f.name}: {len(df):,} rows")
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total 2014: {len(result):,} rows")
    return result


def load_2015_data() -> pd.DataFrame:
    """Load 2015 CSV (Jan-Jun). ~14M rows, 500 MB+.

    Uses dtype optimization to reduce memory footprint.
    Returns DataFrame with columns: Dispatching_base_num, Pickup_date,
    Affiliated_base_num, locationID.
    """
    dtypes = {
        "Dispatching_base_num": "category",
        "Affiliated_base_num": "category",
        "locationID": "int16",
    }
    filepath = DATA_DIR / CSV_2015_FILENAME
    df = pd.read_csv(filepath, dtype=dtypes)
    logger.info(f"Total 2015: {len(df):,} rows")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def load_zone_lookup() -> pd.DataFrame:
    """Load taxi zone lookup table (LocationID -> Borough, Zone).

    Note: this file ships with the trip data archive, hence read from DATA_DIR.
    """
    return pd.read_csv(DATA_DIR / ZONE_LOOKUP_FILENAME)


def load_zone_shapefile() -> gpd.GeoDataFrame:
    """Load NYC TLC taxi zone shapefile."""
    return gpd.read_file(ZONES_DIR / ZONE_SHAPEFILE)
