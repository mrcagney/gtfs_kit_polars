import os
import sys
import pathlib as pb

import numpy as np
import polars as pl

# Make package importable from tests
sys.path.insert(0, os.path.abspath(".."))

import gtfs_kit_next
from gtfs_kit_next.feed import read_feed
from gtfs_kit_next.constants import DTYPES

# Load/create test feeds
DATA_DIR = pb.Path("data")
sample = read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
nyc_subway = read_feed(DATA_DIR / "nyc_subway_gtfs.zip", dist_units="mi")
cairns = read_feed(DATA_DIR / "cairns_gtfs.zip", dist_units="km")

# Shapeless copy
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None

# Remove shape_id on trips (set to nulls)
t = cairns_shapeless.trips
cairns_shapeless.trips = cairns_shapeless.trips.with_columns(pl.lit(None).cast(pl.Utf8).alias("shape_id"))

# Calendar helpers
week = cairns.get_first_week()
cairns_dates = [week[0], week[6]]

# Reference stats (Polars)
cairns_trip_stats = pl.read_csv(
    DATA_DIR / "cairns_trip_stats.csv",
    schema_overrides=(DTYPES["trips"] | DTYPES["routes"]),
)
