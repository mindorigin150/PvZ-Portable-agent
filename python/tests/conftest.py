"""Native environment contract tests require an explicit binary and asset directory."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--binary")
    parser.addoption("--resource-dir")


@pytest.fixture
def native_kwargs(request):
    binary = request.config.getoption("--binary")
    resources = request.config.getoption("--resource-dir")
    if binary is None or resources is None:
        pytest.skip("Pass --binary and --resource-dir for native integration tests")
    return {"binary": binary, "resource_dir": resources}
