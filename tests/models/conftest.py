import pytest

from ml_scripts import predict
from ml_scripts.predict import TorchPredictor


# To retreive and store run-id argument from CLI command
def pytest_addoption(parser):
    parser.addoption("--run-id", action="store", default=None, help="Run ID of an model to use")


# two fixtures to store run_id and predictor in a variable
@pytest.fixture(scope="module")
def run_id(request):
    return request.config.getoption("--run-id")


@pytest.fixture(scope="module")
def predictor(run_id):
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor
