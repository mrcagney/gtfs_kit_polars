"""
Functions useful across modules.
"""

from __future__ import annotations

import copy
import datetime as dt
import functools as ft
import math
from bisect import bisect_left, bisect_right
from functools import cmp_to_key
from typing import Callable, Literal

import json2html as j2h
import numpy as np
import pandas as pd
import polars as pl
import polars_st as st
import shapely.geometry as sg
import utm

from . import constants as cs


def make_lazy(f: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return f if isinstance(f, pl.LazyFrame) else f.lazy()


def are_equal(f: pl.DataFrame | pl.LazyFrame, g: pl.DataFrame | pl.LazyFrame) -> bool:
    """
    Return True if and only if the tables are equal after sorting column names
    and sorting rows by all columns.
    Nulls are treated as equal.
    """
    if f is g:
        return True

    F = f.collect() if isinstance(f, pl.LazyFrame) else f
    G = g.collect() if isinstance(g, pl.LazyFrame) else g

    cols_f = sorted(F.columns)
    cols_g = sorted(G.columns)
    if cols_f != cols_g:
        return False

    cols = cols_f
    F = F.select(cols).sort(cols)
    G = G.select(cols).sort(cols)
    return F.equals(G, null_equal=True)


def is_empty(f: pl.DataFrame | pl.LazyDataFrame) -> bool:
    try:
        return f.is_empty()
    except AttributeError:
        return f.limit(1).collect().is_empty()


def get_srid(g: pl.DataFrame | pl.LazyFrame) -> int:
    """
    Table version of the Polars ST function ``srid``.
    """
    g = g.lazy() if isinstance(g, pl.DataFrame) else g
    return g.limit(1).select(srid=pl.col("geometry").st.srid()).collect().row(0)[0]


def get_utm_srid_0(lon, lat):
    """
    Given the WGS84 longitude and latitude of a point, return its UTM SRID.
    """
    _, _, znum, zlet = utm.from_latlon(lat, lon)
    return (32600 if zlet >= "N" else 32700) + int(znum)


def get_utm_srid(g: st.GeoDataFrame | st.GeoLazyFrame) -> int:
    """
    Return the UTM SRID for the given geotable.
    """
    g = g.lazy() if isinstance(g, pl.DataFrame) else g
    lon, lat = (
        g.limit(1)
        .select(geometry=pl.col("geometry").st.to_srid(cs.WGS84).st.coordinates())
        .collect()
        .row(0)[0][0]
    )
    return get_utm_srid_0(lon, lat)


def to_srid(g: pl.DataFrame | pl.LazyFrame, srid: int) -> pl.DataFrame | pl.LazyFrame:
    """
    Table version of the Polars ST function ``to_srid``.
    """
    return g.with_columns(geometry=pl.col("geometry").st.to_srid(srid))


def datestr_to_date(x: str | None, format_str: str = "%Y%m%d") -> dt.date | None:
    """
    Convert a date string to a datetime.date.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return dt.datetime.strptime(x, format_str).date()


def date_to_datestr(x: dt.date | None, format_str: str = "%Y%m%d") -> str | None:
    """
    Convert a datetime.date to a formatted string.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return x.strftime(format_str)


def timestr_to_seconds_0(x: str, *, mod24: bool = False) -> int | np.nan:
    """
    Given an HH:MM:SS time string ``x``, return the number of seconds
    past midnight that it represents.
    In keeping with GTFS standards, the hours entry may be greater than
    23.
    If ``mod24``, then return the number of seconds modulo ``24*3600``.
    Return ``np.nan`` in case of bad inputs.
    """
    try:
        hours, mins, seconds = x.split(":")
        result = int(hours) * 3600 + int(mins) * 60 + int(seconds)
        if mod24:
            result %= 24 * 3600
    except Exception:
        result = np.nan
    return result


def seconds_to_timestr_0(x: int, *, mod24: bool = False) -> str | np.nan:
    """
    The inverse of :func:`timestr_to_seconds`.
    If ``mod24``, then first take the number of seconds modulo ``24*3600``.
    Return NAN in case of bad inputs.
    """
    try:
        seconds = int(x)
        if mod24:
            seconds %= 24 * 3600
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        result = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        result = np.nan
    return result


def timestr_mod24_0(timestr: str) -> int | np.nan:
    """
    Given a GTFS HH:MM:SS time string, return a timestring in the same
    format but with the hours taken modulo 24.
    Return NAN in case of bad inputes
    """
    try:
        hours, mins, secs = [int(x) for x in timestr.split(":")]
        hours %= 24
        result = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        result = np.nan
    return result


def timestr_to_seconds(col: str, *, mod24: bool = False) -> pl.Expr:
    p = pl.col(col).str.split_exact(":", 3)
    h = p.struct.field("field_0").cast(pl.Int64)
    m = p.struct.field("field_1").cast(pl.Int64)
    s = p.struct.field("field_2").cast(pl.Int64)
    sec = h * 3600 + m * 60 + s
    return sec if not mod24 else (sec % (24 * 3600))


def seconds_to_timestr(col: str, *, mod24: bool = False) -> pl.Expr:
    x = pl.col(col).cast(pl.Int64)
    if mod24:
        x = x % (24 * 3600)
    hh = x // 3600
    mm = (x % 3600) // 60
    ss = x % 60
    return (
        hh.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + mm.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + ss.cast(pl.Utf8).str.zfill(2)
    )


def timestr_mod24(col: str) -> pl.Expr:
    p = pl.col(col).str.split_exact(":", 3)
    h = p.struct.field("field_0").cast(pl.Int64) % 24
    m = p.struct.field("field_1").cast(pl.Int64)
    s = p.struct.field("field_2").cast(pl.Int64)
    return (
        h.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + m.cast(pl.Utf8).str.zfill(2)
        + pl.lit(":")
        + s.cast(pl.Utf8).str.zfill(2)
    )


def timestr_to_min(col: str) -> pl.Expr:
    return timestr_to_seconds(col, mod24=True) // 60


def replace_date(f: pl.LazyFrame, date: str) -> pl.LazyFrame:
    """
    Given a table with a datetime object column called 'datetime' and given a
    YYYYMMDD date string, replace the datetime dates with the given date
    and return the resulting table.
    """
    d = datestr_to_date(date)
    return f.with_columns(
        datetime=pl.datetime(d.year, d.month, d.day)
        + pl.duration(
            hours=pl.col("datetime").dt.hour(),
            minutes=pl.col("datetime").dt.minute(),
            seconds=pl.col("datetime").dt.second(),
        )
    )


def get_segment_length(
    linestring: sg.LineString, p: sg.Point, q: sg.Point | None = None
) -> float:
    """
    Given a Shapely linestring and two Shapely points,
    project the points onto the linestring, and return the distance
    along the linestring between the two points.
    If ``q is None``, then return the distance from the start of the
    linestring to the projection of ``p``.
    The distance is measured in the native coordinates of the linestring.
    """
    # Get projected distances
    d_p = linestring.project(p)
    if q is not None:
        d_q = linestring.project(q)
        d = abs(d_p - d_q)
    else:
        d = d_p
    return d


def get_max_runs(x) -> np.array:
    """
    Given a list of numbers, return a NumPy array of pairs
    (start index, end index + 1) of the runs of max value.

    Example::

        >>> get_max_runs([7, 1, 2, 7, 7, 1, 2])
        array([[0, 1],
               [3, 5]])

    Assume x is not empty.
    Recipe comes from
    `Stack Overflow <http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array>`_.
    """
    # Get 0-1 array where 1 marks the max values of x
    x = np.array(x)
    m = np.max(x)
    y = (x == m) * 1
    # Bound y by zeros to detect runs properly
    bounded = np.hstack(([0], y, [0]))
    # Get 1 at run starts and -1 at run ends
    diffs = np.diff(bounded)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]
    return np.array([run_starts, run_ends]).T
    # # Get lengths of runs and find index of longest
    # idx = np.argmax(run_ends - run_starts)
    # return run_starts[idx], run_ends[idx]


def get_peak_indices(times: list, counts: list) -> np.array:
    """
    Given an increasing list of times as seconds past midnight and a
    list of trip counts at those respective times,
    return a pair of indices i, j such that times[i] to times[j] is
    the first longest time period such that for all i <= x < j,
    counts[x] is the max of counts.
    Assume times and counts have the same nonzero length.

    Examples::

        >>> times = [0, 10, 20, 30, 31, 32, 40]
        >>> counts = [7, 1, 2, 7, 7, 1, 2]
        >>> get_peak_indices(times, counts)
        array([0, 1])

        >>> counts = [0, 0, 0]
        >>> times = [18000, 21600, 28800]
        >>> get_peak_indices(times, counts)
        array([0, 3])

    """
    max_runs = get_max_runs(counts)

    def get_duration(a):
        return times[a[1]] - times[a[0]]

    if len(max_runs) == 1:
        result = max_runs[0]
    else:
        index = np.argmax(np.apply_along_axis(get_duration, 1, max_runs))
        result = max_runs[index]

    return result


def is_metric(dist_units: str) -> bool:
    """
    Return True if the given distance units equals 'm' or 'km';
    otherwise return False.
    """
    return dist_units in ["m", "km"]


def get_convert_dist(dist_units_in: str, dist_units_out: str):
    """
    Return a *Polars expression builder* for distance conversion:

      expr_or_col -> expr * factor

    Only supports units in :const:`constants.DIST_UNITS`.
    Usage:
        .with_columns(distance_km = get_convert_dist_pl("m","km")("distance_m"))
        .with_columns(distance_mi = get_convert_dist_pl("km","mi")(pl.col("dist")))
    """
    di, do = dist_units_in, dist_units_out
    DU = cs.DIST_UNITS
    if not (di in DU and do in DU):
        raise ValueError(f"Distance units must lie in {DU}")

    d = {
        "ft": {"ft": 1, "m": 0.3048, "mi": 1 / 5280, "km": 0.000_304_8},
        "m": {"ft": 1 / 0.3048, "m": 1, "mi": 1 / 1609.344, "km": 1 / 1000},
        "mi": {"ft": 5280, "m": 1609.344, "mi": 1, "km": 1.609_344},
        "km": {"ft": 1 / 0.000_304_8, "m": 1000, "mi": 1 / 1.609_344, "km": 1},
    }
    factor = d[di][do]

    def builder(expr_or_col):
        e = expr_or_col if isinstance(expr_or_col, pl.Expr) else pl.col(expr_or_col)
        return (e.cast(pl.Float64) * pl.lit(factor)).cast(pl.Float64)

    return builder


def is_not_null(f: pl.DataFrame | pl.LazyFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    f = f.lazy() if isinstance(f, pl.DataFrame) else f
    if (
        col_name in f.collect_schema().names()
        and f.select(bingo=pl.col("shape_dist_traveled").is_not_null().any())
        .collect()
        .row(0)[0]
    ):
        return True
    else:
        return False


def get_active_trips_df(trip_times: pd.DataFrame) -> pd.Series:
    """
    Count the number of trips in ``trip_times`` that are active
    at any given time.

    Assume ``trip_times`` contains the columns

    - start_time: start time of the trip in seconds past midnight
    - end_time: end time of the trip in seconds past midnight

    Return a Series whose index is times from midnight when trips
    start and end and whose values are the number of active trips for that time.
    """
    active_trips = (
        pd.concat(
            [
                pd.Series(1, trip_times.start_time),  # departed add 1
                pd.Series(-1, trip_times.end_time),  # arrived subtract 1
            ]
        )
        .groupby(level=0, sort=True)
        .sum()
        .cumsum()
        .ffill()
    )
    return active_trips


def combine_time_series(
    series_by_indicator: dict[str, pl.DataFrame | pl.LazyFrame],
    *,
    kind: Literal["route", "stop"],
    split_directions: bool = False,
) -> pl.LazyFrame:
    """
    Combine a dict of wide time series (one table per indicator, columns are entities)
    into a single long-form time series with columns

    - ``'datetime'``
    - ``'route_id'`` or ``'stop_id'``: depending on ``kind``
    - ``'direction_id'``: present if and only if ``split_directions``
    - one column per indicator provided in `series_by_indicator`
    - ``'service_speed'``: if both ``service_distance`` and ``service_duration`` present

    If ``split_directions``, then assume the original time series contains data
    separated by trip direction; otherwise, assume not.
    The separation is indicated by a suffix ``'-0'`` (direction 0) or ``'-1'``
    (direction 1) in the route ID or stop ID column values.
    """
    if not series_by_indicator:
        return pl.LazyFrame(schema={"datetime": pl.Datetime})

    indicators = list(series_by_indicator.keys())
    entity_col = "route_id" if kind == "route" else "stop_id"

    # Convert wide to long time series for each indicator
    long_frames = []
    for ind, f in series_by_indicator.items():
        f = f if isinstance(f, pl.LazyFrame) else f.lazy()
        value_cols = [c for c in f.collect_schema().names() if c != "datetime"]
        if not value_cols:
            continue
        g = f.unpivot(
            index=["datetime"],
            on=value_cols,
            variable_name=entity_col,
            value_name=ind,
        )
        long_frames.append(g)

    if not long_frames:
        return pl.LazyFrame(schema={"datetime": pl.Datetime, entity_col: pl.Utf8})

    # Full join all indicators on (``datetime``, ``entity``)
    f = ft.reduce(
        lambda left, right: left.join(
            right, on=["datetime", entity_col], how="full", coalesce=True
        ),
        long_frames,
    )

    # Optionally split direction from encoded IDs "<id>-<dir>"
    if split_directions:
        f = (
            f.with_columns(
                _base=pl.col(entity_col).str.extract(r"^(.*)-(0|1)$", group_index=1),
                _dir=pl.col(entity_col)
                .str.extract(r"^(.*)-(0|1)$", group_index=2)
                .cast(pl.Int32),
            )
            .with_columns(
                pl.when(pl.col("_base").is_not_null())
                .then(pl.col("_base"))
                .otherwise(pl.col(entity_col))
                .alias(entity_col),
                pl.col("_dir").alias("direction_id"),
            )
            .drop("_base", "_dir")
        )
    # Coerce numeric indicators and fill nulls with 0; speed handled later
    fcols = f.collect_schema().names()
    numeric_cols = [ind for ind in indicators if ind in fcols]
    if numeric_cols:
        f = f.with_columns(
            [pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in numeric_cols]
        )

    # Compute service_speed if possible
    if "service_distance" in fcols and "service_duration" in fcols:
        f = f.with_columns(
            service_speed=(
                pl.when(pl.col("service_duration") > 0)
                .then(pl.col("service_distance") / pl.col("service_duration"))
                .otherwise(0.0)
            ).cast(pl.Float64)
        )

    # Arrange columns
    cols0 = ["datetime", entity_col]
    if split_directions:
        cols0.append("direction_id")
    cols = cols0 + [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
        "service_speed",
    ]
    return f.select(cols).sort(cols0)


def get_bin_size(time_series: pl.LazyFrame) -> float:
    """
    Return the number of minutes per bin of the given time series with datetime column
    'datetime'.
    Assume the time series is regularly sampled and therefore has a single bin size.
    Return None if there's only one unique datetime present.
    """
    times = (
        time_series.select("datetime")
        .unique()
        .sort("datetime")
        .collect()["datetime"]
        .to_list()[:2]
    )
    if len(times) >= 2:
        return (times[1] - times[0]).seconds / 60


def downsample(
    time_series: pl.DataFrame | pl.LazyFrame, num_minutes: int
) -> pl.LazyFrame:
    """
    Downsample the given route, stop, or feed time series,
    (outputs of :func:`.routes.compute_route_time_series`,
    :func:`.stops.compute_stop_time_series`, or
    :func:`.miscellany.compute_feed_time_series`,
    respectively) to time bins of size ``num_minutes`` minutes.

    Return the given time series unchanged if it's empty or
    has only one time bin per date.
    Raise a value error if ``num_minutes`` does not evenly divide 1440
    (the number of minutes in a day) or if its not a multiple of the
    bin size of the given time series.
    """
    # Coerce to LazyFrame
    time_series = (
        time_series.lazy() if isinstance(time_series, pl.DataFrame) else time_series
    )

    # Handle defunct cases
    if is_empty(time_series):
        return time_series

    orig_num_minutes = get_bin_size(time_series)
    if orig_num_minutes is None:
        return time_series

    num_minutes = int(num_minutes)
    if num_minutes == orig_num_minutes:
        return time_series

    if 1440 % num_minutes != 0:
        raise ValueError("num_minutes must evenly divide 24*60")

    if num_minutes % orig_num_minutes != 0:
        raise ValueError(
            f"num_minutes must be a multiple of the original time series bin size "
            f"({orig_num_minutes} minutes)"
        )

    # Handle generic case
    cols = time_series.collect_schema().names()
    freq = f"{num_minutes}m"
    if "stop_id" in cols:
        # It's a stops time series
        metrics = ["num_trips"]
        dims = [c for c in cols if c not in (["datetime"] + metrics)]
        result = time_series.group_by_dynamic(
            "datetime",
            every=freq,
            closed="left",
            label="left",
            group_by=dims,
        ).agg(num_trips=pl.col("num_trips").sum())
    else:
        # It's a route or feed time series
        metrics = [
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        ]
        dims = [c for c in cols if c not in (["datetime"] + metrics)]

        # Sum across coarse timestamps and get last fine timestamp
        sums = (
            time_series.group_by_dynamic(
                "datetime",
                every=freq,
                closed="left",
                label="left",
                group_by=dims,
            )
            .agg(
                num_trip_starts=pl.col("num_trip_starts").sum(),
                num_trip_ends=pl.col("num_trip_ends").sum(),
                service_distance=pl.col("service_distance").sum(),
                service_duration=pl.col("service_duration").sum(),
                last_dt=pl.col("datetime").max(),
            )
            .rename({"datetime": "big"})  # coarse timestamp label
        )
        # Get last fine timestamp values per coarse timestamp values of
        # num_trips and num_trip_ends to use in aggregation:
        # num_trips = num_trips_last + sum(num_trip_ends in all but last fine timestamp)
        #           = num_trips_last + (num_trip_ends_sum - num_trip_ends_last)
        last_vals = (
            time_series.select(dims + ["datetime", "num_trips", "num_trip_ends"])
            .join(sums.select(dims + ["big", "last_dt"]), on=dims, how="inner")
            .filter(pl.col("datetime") == pl.col("last_dt"))
            .group_by(dims + ["big"])
            .agg(
                num_trips_last=pl.col("num_trips").max(),
                num_trip_ends_last=pl.col("num_trip_ends").max(),
            )
        )

        # Merge and compute final metrics
        result = sums.join(last_vals, on=dims + ["big"], how="left").with_columns(
            num_trips=pl.col("num_trips_last")
            + pl.col("num_trip_ends")
            - pl.col("num_trip_ends_last"),
            service_speed=(
                pl.col("service_distance") / pl.col("service_duration")
            ).fill_null(0),
            datetime=pl.col("big"),
        )

    return result.select(["datetime"] + dims + metrics).sort(["datetime"] + dims)


def make_html(d: dict) -> str:
    """
    Convert the given dictionary into an HTML table (string) with
    two columns: keys of dictionary, values of dictionary.
    """
    return j2h.json2html.convert(
        json=d, table_attributes="class='table table-condensed table-hover'"
    )


def drop_feature_ids(collection: dict) -> dict:
    """
    Given a GeoJSON FeatureCollection, remove the ``'id'`` attribute of each
    Feature, if it exists.
    """
    new_features = []
    for f in collection["features"]:
        new_f = copy.deepcopy(f)
        if "id" in new_f:
            del new_f["id"]
        new_features.append(new_f)

    collection["features"] = new_features
    return collection


def longest_subsequence(
    seq, mode="strictly", order="increasing", key=None, *, index=False
):
    """
    Return the longest increasing subsequence of `seq`.

    Parameters
    ----------
    seq : sequence object
      Can be any sequence, like `str`, `list`, `numpy.array`.
    mode : {'strict', 'strictly', 'weak', 'weakly'}, optional
      If set to 'strict', the subsequence will contain unique elements.
      Using 'weak' an element can be repeated many times.
      Modes ending in -ly serve as a convenience to use with `order` parameter,
      because `longest_sequence(seq, 'weakly', 'increasing')` reads better.
      The default is 'strict'.
    order : {'increasing', 'decreasing'}, optional
      By default return the longest increasing subsequence, but it is possible
      to return the longest decreasing sequence as well.
    key : function, optional
      Specifies a function of one argument that is used to extract a comparison
      key from each list element (e.g., `str.lower`, `lambda x: x[0]`).
      The default value is `None` (compare the elements directly).
    index : bool, optional
      If set to `True`, return the indices of the subsequence, otherwise return
      the elements. Default is `False`.

    Returns
    -------
    elements : list, optional
      A `list` of elements of the longest subsequence.
      Returned by default and when `index` is set to `False`.
    indices : list, optional
      A `list` of indices pointing to elements in the longest subsequence.
      Returned when `index` is set to `True`.

    Taken from `this Stack Overflow answer <https://stackoverflow.com/a/38337443>`_.
    """
    bisect = bisect_left if mode.startswith("strict") else bisect_right

    # compute keys for comparison just once
    rank = seq if key is None else map(key, seq)
    if order == "decreasing":
        rank = map(cmp_to_key(lambda x, y: 1 if x < y else 0 if x == y else -1), rank)
    rank = list(rank)

    if not rank:
        return []

    lastoflength = [0]  # end position of subsequence with given length
    predecessor = [None]  # penultimate element of l.i.s. ending at given position

    for i in range(1, len(seq)):
        # seq[i] can extend a subsequence that ends with a lesser (or equal) element
        j = bisect([rank[k] for k in lastoflength], rank[i])
        # update existing subsequence of length j or extend the longest
        try:
            lastoflength[j] = i
        except Exception:
            lastoflength.append(i)
        # remember element before seq[i] in the subsequence
        predecessor.append(lastoflength[j - 1] if j > 0 else None)

    # trace indices [p^n(i), ..., p(p(i)), p(i), i], where n=len(lastoflength)-1
    def trace(i):
        if i is not None:
            yield from trace(predecessor[i])
            yield i

    indices = trace(lastoflength[-1])

    return list(indices) if index else [seq[i] for i in indices]


def make_ids(n: int, prefix: str = "id_"):
    """
    Return a length ``n`` list of unique sequentially labelled strings for use as IDs.

    Example::

        >>> make_ids(11, prefix="s")
        ['s00', s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10']

    """
    if n < 1:
        result = []
    elif n == 1:
        result = [f"{prefix}0"]
    else:
        k = int(math.log10(n - 1)) + 1  # Number of digits for IDs
        result = [f"{prefix}{i:0{k}d}" for i in range(n)]

    return result
