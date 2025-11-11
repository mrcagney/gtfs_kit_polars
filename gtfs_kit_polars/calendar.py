"""
Functions about calendar and calendar_dates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import datetime as dt
import polars as pl

from . import helpers as hp

# Help type checkers but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_dates(feed: "Feed", *, as_date_obj: bool = False) -> list[str] | list[dt.date]:
    """
    Return the inclusive date range covered by `feed.calendar` and `feed.calendar_dates`
    as consecutive days.
    If neither table yields dates, return the empty list.

    If ``as_date_obj``, then return datetime.date objects instead.

    Note that this is a range and not the set of actual service days.
    """

    def min_max(lf: pl.LazyFrame | None, col: str):
        if lf is None or col not in lf.collect_schema().names():
            return None, None
        e = pl.col(col).cast(pl.Utf8).str.pad_start(8, "0")
        out = lf.select(e.min().alias("a"), e.max().alias("b")).collect()
        return out["a"][0], out["b"][0]

    a, _ = min_max(feed.calendar, "start_date")
    _, b = min_max(feed.calendar, "end_date")
    c, d = min_max(feed.calendar_dates, "date")

    bounds = [x for x in (a, b, c, d) if x is not None]
    if not bounds:
        return []

    s = hp.datestr_to_date(min(bounds))
    e = hp.datestr_to_date(max(bounds))
    days = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    return days if as_date_obj else [hp.date_to_datestr(d) for d in days]


def get_week(
    feed: "Feed", k: int, *, as_date_obj: bool = False
) -> list[str] | list[dt.date]:
    """
    Given a Feed and a positive integer ``k``,
    return a list of YYYYMMDD date strings corresponding to the kth Monday--Sunday week
    (or initial segment thereof) for which the Feed is valid.
    For example, k=1 returns the first Monday--Sunday week (or initial segment thereof).
    If the Feed does not have k Mondays, then return the empty list.

    If ``as_date_obj``, then return datetime.date objects instead.
    """
    if k < 1:
        return []

    dates = feed.get_dates(as_date_obj=True)
    n = len(dates)

    monday_index = next((i for i, d in enumerate(dates) if d.weekday() == 0), None)
    if monday_index is None:
        return []

    start = monday_index + 7 * (k - 1)
    if start >= n:
        return []

    end = start + 7
    out = dates[start:end]
    return out if as_date_obj else [hp.date_to_datestr(x) for x in out]


def get_first_week(
    feed: "Feed", *, as_date_obj: bool = False
) -> list[str] | list[dt.date]:
    """
    Return a list of YYYYMMDD date strings for the first Monday--Sunday
    week (or initial segment thereof) for which the given Feed is valid.
    If the feed has no Mondays, then return the empty list.

    If ``as_date_obj``, then return datetime.date objects instead.
    """
    return get_week(feed, 1, as_date_obj=as_date_obj)


def subset_dates(feed: "Feed", dates: list[str]) -> list[str]:
    """
    Given a Feed and a list of YYYYMMDD date strings,
    return the sorted sublist of dates that lie in the Feed's dates
    (the output :func:`feed.get_dates`).
    Could be an empty list.
    """
    return sorted(set(dates) & set(feed.get_dates()))
