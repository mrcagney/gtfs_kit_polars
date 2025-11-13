GTFS Kit Polars
***************
.. image:: https://github.com/mrcagney/gtfs_kit_polars/actions/workflows/test.yml/badge.svg

GTFS Kit Polars is a Python 3.12+ library for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data.
It uses Polars and Polars ST LazyFrames to do the heavy lifting.

The functions/methods of GTFS Kit Polars assume a valid GTFS feed but offer no inbuilt validation, because GTFS validation is complex and already solved by dedicated libraries.
So unless you know what you're doing, use the `Canonical GTFS Validator <https://gtfs-validator.mobilitydata.org/>`_ before you analyze a feed with GTFS Kit Polars.

GTFS Kit Polars is an experimental port of the `GTFS Kit library <https://github.com/mrcagney/gtfs_kit>`_ from Pandas to Polars.
It can process large feeds much faster than the Pandas version, and if it proves useful enough, then i'll incorporate it into GTFS Kit as a new release.

The one thing i don't like about this Polars version is its dependence on `Polars ST <https://github.com/oreilles/polars-st>`_, a promising new geospatial library but one that is not yet as user-friendly as GeoPandas.

Installation
=============
Install it from PyPI with UV, say, via ``uv add gtfs_kit_polars``.

Examples
========
In the Jupyter notebook ``notebooks/examples.ipynb``, which is a Github-displayable export of the Marimo notebook ``notebooks/examples.py``.

Authors
=========
- Alex Raichev (2025-11), maintainer

Notes
=====
- This project's development status is Alpha.
  I use GTFS Kit Polars at my job and change it breakingly to suit my needs.
- This project uses semantic versioning.
- I aim for GTFS Kit Polars to handle `the current GTFS <https://developers.google.com/transit/gtfs/reference>`_.
  In particular, i avoid handling `GTFS extensions <https://developers.google.com/transit/gtfs/reference/gtfs-extensions>`_.
  That is the most reasonable scope boundary i can draw at present, given this project's tiny budget.
  If you would like to fund this project to expand its scope, please email me.
- Thanks to `MRCagney <http://www.mrcagney.com/>`_ for periodically donating to this project.
- Constructive feedback and contributions are welcome.
  Please issue pull requests from a feature branch into the ``develop`` branch and include tests.
- GTFS time is measured relative to noon minus 12 hours, which can mess things up when crossing into daylight savings time.
  I don't think this issue causes any bugs in GTFS Kit, but you and i have been warned.
  Thanks to user Github user ``derhuerst`` for bringing this to my attention in `closed Issue 8 <https://github.com/mrcagney/gtfs_kit_polars/issues/8#issue-1063633457>`_.

Maintainer Notes
================
- Before pushing to master, export the example Marimo notebook to Jupyter via ``uv run marimo export ipynb notebooks/examples.py -o notebooks/examples.ipynb --include-outputs -f``, because the docs refer to that Github-displayable version.
- After pushing to master, update the published docs via ``uv run make -C docs publish-docs-github``