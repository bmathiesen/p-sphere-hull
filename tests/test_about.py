import PSphereHull


def test_no_version_attribute():
    assert not hasattr(PSphereHull, "__version__")


def test_can_get_version():
    # If this fails, then it probably means that the library is not
    # installed via `pip install -e .` or equivalent.
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata

    version = metadata.version("PSphereHull")
    assert version is not None
