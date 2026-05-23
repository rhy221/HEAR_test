import pytest


def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default=None, help="Path to submission CSV to validate")


@pytest.fixture
def csv_path(request):
    return request.config.getoption("--csv")
