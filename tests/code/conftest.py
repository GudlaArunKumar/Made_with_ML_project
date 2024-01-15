import pytest

from ml_scripts.data import CustomPreprocessor

"""
This config test script to store fixture variables to be used
in other test scripts for pytest package (it is how it works)
"""


@pytest.fixture
def dataset_loc():
    return "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"


@pytest.fixture
def preprocessor():
    return CustomPreprocessor()
