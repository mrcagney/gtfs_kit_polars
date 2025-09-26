GTFS Kit Next
**************
.. image:: https://github.com/mrcagney/gtfs_kit_next/actions/workflows/test.yml/badge.svg

GTFS Kit Next is a Python 3.12+ library for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It uses Polars and Polars ST to do the heavy lifting.


Installation
=============
Install it from PyPI with UV, say, via ``uv add gtfs_kit_next_next``.


Examples
========
In the Marimo notebook ``notebooks/examples.py``.


Authors
=========
- Alex Raichev (2019-09), maintainer


Progress
=========
Progress porting the Pandas original to Polars.

[x] calendar.py
[] cleaners.py
[x] constants.py
[x] feed.py
[] helpers.py: in progress
[] miscellany.py
[] routes.py
[] shapes.py: in progress
[] stop_times.py
[] stops.py
[] trips.py

Notes
=====
- This project's development status is Alpha.
  I use GTFS Kit at my job and change it breakingly to suit my needs.
- This project uses semantic versioning.
- I aim for GTFS Kit to handle `the current GTFS <https://developers.google.com/transit/gtfs/reference>`_.
  In particular, i avoid handling `GTFS extensions <https://developers.google.com/transit/gtfs/reference/gtfs-extensions>`_.
  That is the most reasonable scope boundary i can draw at present, given this project's tiny budget.
  If you would like to fund this project to expand its scope, please email me.
- Thanks to `MRCagney <http://www.mrcagney.com/>`_ for periodically donating to this project.
- Constructive feedback and contributions are welcome.
  Please issue pull requests from a feature branch into the ``develop`` branch and include tests.
- GTFS time is measured relative to noon minus 12 hours, which can mess things up when crossing into daylight savings time.
  I don't think this issue causes any bugs in GTFS Kit, but you and i have been warned.
  Thanks to user Github user ``derhuerst`` for bringing this to my attention in `closed Issue 8 <https://github.com/mrcagney/gtfs_kit_next/issues/8#issue-1063633457>`_.
- With release 10.0.0, i removed the validation module ``validators.py`` to avoid duplicating the work of what is now `the canonical feed validator <https://github.com/MobilityData/gtfs-validator>`_ (written in Java).