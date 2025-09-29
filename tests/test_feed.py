import pytest
from pathlib import Path
import shutil
import tempfile

import polars as pl

from .context import DATA_DIR
from gtfs_kit_polars import feed as gkf
from gtfs_kit_polars import constants as cs


def test_feed():
    feed = gkf.Feed(agency=pl.DataFrame(), dist_units="km")
    for key in cs.FEED_ATTRS:
        val = getattr(feed, key)
        if key == "dist_units":
            assert val == "km"
        elif key == "agency":
            assert isinstance(val, pl.LazyFrame)
        else:
            assert val is None


def test_str():
    feed = gkf.Feed(agency=pl.DataFrame(), dist_units="km")
    assert isinstance(str(feed), str)


def test_eq():
    assert gkf.Feed(dist_units="m") == gkf.Feed(dist_units="m")

    feed1 = gkf.Feed(
        dist_units="m",
        stops=pl.DataFrame([[1, 2], [3, 4]], schema=["a", "b"]),
    )
    assert feed1 == feed1

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pl.DataFrame([[4, 3], [2, 1]], schema=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pl.DataFrame([[3, 4], [2, 1]], schema=["b", "a"]),
    )
    assert feed1 != feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pl.DataFrame([[4, 3], [2, 1]], schema=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(dist_units="mi", stops=feed1.stops)
    assert feed1 != feed2


def test_copy():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    feed2 = feed1.copy()

    for key in cs.FEED_ATTRS:
        v = getattr(feed2, key)
        w = getattr(feed1, key)
        if isinstance(v, pl.LazyFrame):
            assert v.collect().equals(w.collect())
        else:
            assert v == w


# --------------------------------------------
# Test functions about inputs and outputs
# --------------------------------------------
def test_list_feed():
    with pytest.raises(ValueError):
        gkf.list_feed("bad_path!")

    for path in [DATA_DIR / "sample_gtfs.zip", DATA_DIR / "sample_gtfs"]:
        f = gkf.list_feed(path)
        assert isinstance(f, pl.DataFrame)
        assert set(f.collect_schema().names()) == {"file_name", "file_size"}
        assert f.shape[0] in [12, 13]


def test_read_feed():
    with pytest.raises(ValueError):
        gkf.read_feed("bad_path!", dist_units="km")

    with pytest.raises(ValueError):
        gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="bingo")

    with pytest.raises(TypeError):
        gkf.read_feed(path=DATA_DIR / "sample_gtfs.zip")  # missing dist_units

    feed = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="m")
    assert isinstance(feed, gkf.Feed)

    feed = gkf.read_feed(DATA_DIR / "cairns_gtfs.zip", dist_units="m")
    assert isinstance(feed, gkf.Feed)

    # Feed should have None feed_info table
    assert feed.feed_info is None


def test_to_file():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")

    # Export feed1, import it as feed2, and then test equality
    for out_path in [DATA_DIR / "bingo.zip", DATA_DIR / "bingo"]:
        feed1.to_file(out_path)
        feed2 = gkf.read_feed(out_path, "km")
        assert feed1 == feed2
        try:
            out_path.unlink()
        except Exception:
            shutil.rmtree(str(out_path))

    # Test that integer columns with nulls get output properly.
    feed3 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")

    # Collect trips, set direction_id on the first three rows: [None, 1, 0]
    t = (
        feed3.trips.collect()
        .with_row_index("i")
        .with_columns(
            pl.when(pl.col("i") == 0)
            .then(pl.lit(None))
            .when(pl.col("i") == 1)
            .then(pl.lit(1))
            .when(pl.col("i") == 2)
            .then(pl.lit(0))
            .otherwise(pl.col("direction_id"))
            .cast(pl.Int8)
            .alias("direction_id")
        )
        .drop("i")
    )
    feed3.trips = t.lazy()

    q = DATA_DIR / "bingo.zip"
    feed3.to_file(q)

    tmp_dir = tempfile.TemporaryDirectory()
    shutil.unpack_archive(str(q), tmp_dir.name, "zip")
    qq = Path(tmp_dir.name) / "trips.txt"
    u = pl.read_csv(qq, schema_overrides={"direction_id": pl.Int8})

    bad = u.filter(~pl.col("direction_id").is_in([None, 0, 1]))
    assert bad.is_empty()

    tmp_dir.cleanup()
    q.unlink()
