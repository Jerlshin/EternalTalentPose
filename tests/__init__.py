"""REDSTACK test suite root — makes ``tests`` an importable package.

Required so absolute imports across test subpackages (e.g.
``from tests.fixtures.fake_ports import ...`` from a module under
``tests/integration/``) resolve under pytest's package-rootpath insertion,
which walks up until it finds a directory with no ``__init__.py``.
"""
