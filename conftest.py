import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", type=str, default="cpu")
    parser.addoption("--batch_size", action="store", type=int, default=64)
    parser.addoption("--dataset_name", action="store", type=str, default="imagenet")


@pytest.fixture(scope="session")
def device(pytestconfig):
    return pytestconfig.getoption("device")


@pytest.fixture(scope="session")
def batch_size(pytestconfig):
    return pytestconfig.getoption("batch_size")


@pytest.fixture(scope="session")
def dataset_name(pytestconfig):
    return pytestconfig.getoption("dataset_name")
