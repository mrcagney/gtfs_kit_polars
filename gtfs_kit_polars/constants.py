"""
Constants useful across modules.
"""

import polars as pl

#: GTFS data types (Polars dtypes)
DTYPES = {
    "agency": {
        "agency_id": pl.Utf8,
        "agency_name": pl.Utf8,
        "agency_url": pl.Utf8,
        "agency_timezone": pl.Utf8,
        "agency_lang": pl.Utf8,
        "agency_phone": pl.Utf8,
        "agency_fare_url": pl.Utf8,
        "agency_email": pl.Utf8,
    },
    "attributions": {
        "attribution_id": pl.Utf8,
        "agency_id": pl.Utf8,
        "route_id": pl.Utf8,
        "trip_id": pl.Utf8,
        "organization_name": pl.Utf8,
        "is_producer": pl.Int8,
        "is_operator": pl.Int8,
        "is_authority": pl.Int8,
        "attribution_url": pl.Utf8,
        "attribution_email": pl.Utf8,
        "attribution_phone": pl.Utf8,
    },
    "calendar": {
        "service_id": pl.Utf8,
        "monday": pl.Int8,
        "tuesday": pl.Int8,
        "wednesday": pl.Int8,
        "thursday": pl.Int8,
        "friday": pl.Int8,
        "saturday": pl.Int8,
        "sunday": pl.Int8,
        "start_date": pl.Utf8,  # YYYYMMDD
        "end_date": pl.Utf8,  # YYYYMMDD
    },
    "calendar_dates": {
        "service_id": pl.Utf8,
        "date": pl.Utf8,  # YYYYMMDD
        "exception_type": pl.Int8,
    },
    "fare_attributes": {
        "fare_id": pl.Utf8,
        "price": pl.Float64,
        "currency_type": pl.Utf8,
        "payment_method": pl.Int8,
        "transfers": pl.Int8,
        "transfer_duration": pl.Int16,
    },
    "fare_rules": {
        "fare_id": pl.Utf8,
        "route_id": pl.Utf8,
        "origin_id": pl.Utf8,
        "destination_id": pl.Utf8,
        "contains_id": pl.Utf8,
    },
    "feed_info": {
        "feed_publisher_name": pl.Utf8,
        "feed_publisher_url": pl.Utf8,
        "feed_lang": pl.Utf8,
        "feed_start_date": pl.Utf8,  # YYYYMMDD
        "feed_end_date": pl.Utf8,  # YYYYMMDD
        "feed_version": pl.Utf8,
    },
    "frequencies": {
        "trip_id": pl.Utf8,
        "start_time": pl.Utf8,  # HH:MM:SS (may exceed 24h)
        "end_time": pl.Utf8,  # HH:MM:SS (may exceed 24h)
        "headway_secs": pl.Int16,
        "exact_times": pl.Int8,
    },
    "routes": {
        "route_id": pl.Utf8,
        "agency_id": pl.Utf8,
        "route_short_name": pl.Utf8,
        "route_long_name": pl.Utf8,
        "route_desc": pl.Utf8,
        "route_type": pl.Int8,
        "route_url": pl.Utf8,
        "route_color": pl.Utf8,
        "route_text_color": pl.Utf8,
    },
    "shapes": {
        "shape_id": pl.Utf8,
        "shape_pt_lat": pl.Float64,
        "shape_pt_lon": pl.Float64,
        "shape_pt_sequence": pl.Int32,
        "shape_dist_traveled": pl.Float64,
    },
    "stop_times": {
        "trip_id": pl.Utf8,
        "arrival_time": pl.Utf8,  # HH:MM:SS (may exceed 24h)
        "departure_time": pl.Utf8,  # HH:MM:SS (may exceed 24h)
        "stop_id": pl.Utf8,
        "stop_sequence": pl.Int32,
        "stop_headsign": pl.Utf8,
        "pickup_type": pl.Int8,
        "drop_off_type": pl.Int8,
        "shape_dist_traveled": pl.Float64,
        "timepoint": pl.Int8,
    },
    "stops": {
        "stop_id": pl.Utf8,
        "stop_code": pl.Utf8,
        "stop_name": pl.Utf8,
        "stop_desc": pl.Utf8,
        "stop_lat": pl.Float64,
        "stop_lon": pl.Float64,
        "zone_id": pl.Utf8,
        "stop_url": pl.Utf8,
        "location_type": pl.Int8,
        "parent_station": pl.Utf8,
        "stop_timezone": pl.Utf8,
        "wheelchair_boarding": pl.Int8,
    },
    "transfers": {
        "from_stop_id": pl.Utf8,
        "to_stop_id": pl.Utf8,
        "transfer_type": pl.Int8,
        "min_transfer_time": pl.Int16,
    },
    "trips": {
        "route_id": pl.Utf8,
        "service_id": pl.Utf8,
        "trip_id": pl.Utf8,
        "trip_headsign": pl.Utf8,
        "trip_short_name": pl.Utf8,
        "direction_id": pl.Int8,
        "block_id": pl.Utf8,
        "shape_id": pl.Utf8,
        "wheelchair_accessible": pl.Int8,
        "bikes_allowed": pl.Int8,
    },
}

#: Valid distance units
DIST_UNITS = ["ft", "mi", "m", "km"]

#: Feed attributes
FEED_ATTRS = [
    # Metadata
    "dist_units",
    "unzip_dir",
    # GTFS tables
    "agency",
    "attributions",
    "calendar",
    "calendar_dates",
    "fare_attributes",
    "fare_rules",
    "feed_info",
    "frequencies",
    "routes",
    "shapes",
    "stops",
    "stop_times",
    "trips",
    "transfers",
]

#: WGS84 coordinate reference system (used by spatial ops)
WGS84 = 4326

#: Colorbrewer 8-class Set2 colors
COLORS_SET2 = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
