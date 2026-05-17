import pytest


def test_engine_importable():
    import pigsmc._engine  # noqa: F401


def test_engine_has_docstring():
    import pigsmc._engine
    assert pigsmc._engine.__doc__ is not None
    assert len(pigsmc._engine.__doc__) > 0
