"""Project-wide configuration: paths, constants, and bounds."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input" / "uber-trip-data" / "uber-trip-data"
ZONES_DIR = PROJECT_ROOT / "data" / "input" / "taxi_zones"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

NYC_BOUNDS = {
    "lat_min": 40.5,
    "lat_max": 40.92,
    "lon_min": -74.3,
    "lon_max": -73.7,
}

RANDOM_STATE = 42
NYC_LAT_CENTER = 40.7128  # For cosine correction of longitude distortion

CSV_2014_PATTERN = "uber-raw-data-*14.csv"
CSV_2015_FILENAME = "uber-raw-data-janjune-15.csv"
ZONE_LOOKUP_FILENAME = "taxi-zone-lookup.csv"
ZONE_SHAPEFILE = "taxi_zones.shp"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
