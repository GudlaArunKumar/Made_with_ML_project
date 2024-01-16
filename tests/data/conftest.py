import great_expectations as ge
import pandas as pd
import pytest


# it's a inbuilt func from pytest library to store cli arguments
def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings
    """
    parser.addoption("--dataset-loc", action="store", default=None, help="Dataset location for testing")


# df used as fixture in test script
@pytest.fixture(scope="module")
def df(request):
    dataset_loc = request.config.getoption("--dataset-loc")
    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc))
    return df
