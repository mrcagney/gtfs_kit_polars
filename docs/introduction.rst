Introduction
=============
GTFS Kit Polars is a Python 3.12+ library for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data.
It uses Polars and Polars ST LazyFrames to do the heavy lifting.

The functions/methods of GTFS Kit Polars assume a valid GTFS feed but offer no inbuilt validation, because GTFS validation is complex and already solved by dedicated libraries.
So unless you know what you're doing, use the `Canonical GTFS Validator <https://gtfs-validator.mobilitydata.org/>`_ before you analyze a feed with GTFS Kit Polars.

GTFS Kit Polars is an experimental port of the `GTFS Kit library <https://github.com/mrcagney/gtfs_kit>`_ from Pandas to Polars.
It can process large feeds much faster than the Pandas version, and if it proves useful enough, then i'll incorporate it into GTFS Kit as a new release.

The one thing i don't like about this Polars version is its dependence on `Polars ST <https://github.com/oreilles/polars-st>`_, a promising new geospatial library but one that is not yet as user-friendly as GeoPandas.


Authors
=========
- Alex Raichev, 2025-11


Installation
=============
Install it from PyPI with UV, say, via ``uv add gtfs_kit_polars``.


Examples
========
In `this Jupyter notebook <https://github.com/mrcagney/gtfs_kit_polars/notebooks/examples.ipynb>`_.


Conventions
============
- In conformance with GTFS, dates are encoded as YYYYMMDD date strings, and times are encoded as HH:MM:SS time strings with the possibility that HH > 24. **Watch out** for that possibility, because it has counterintuitive consequences; see e.g. :func:`.trips.is_active_trip`, which is used in :func:`.routes.compute_route_stats`,  :func:`.stops.compute_stop_stats`, and :func:`.miscellany.compute_network_stats`.
- 'DataFrame' and 'Series' refer to Pandas DataFrame and Series objects,
  respectively
