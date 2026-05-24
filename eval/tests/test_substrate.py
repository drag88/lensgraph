"""Substrate sanity. Verifies pytest config + marker registration."""


def test_pytest_slow_marker_registered(pytestconfig):
    """The 'slow' marker MUST be registered so @pytest.mark.slow doesn't
    emit PytestUnknownMarkWarning in subsequent test files."""
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("slow:") for m in markers), (
        f"slow marker not registered; got: {markers}"
    )
