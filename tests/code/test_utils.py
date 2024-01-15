import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from ray.train.torch import get_device

from ml_scripts import utils


def test_set_seed():
    """testing the set_seed function present in utils module"""
    utils.set_seeds()
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    utils.set_seeds()
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    assert np.array_equal(a, x)
    assert np.array_equal(b, y)


def test_save_and_load_dict():
    """testing the save_dict and load_dict function present in utils module"""
    with tempfile.TemporaryDirectory() as dp:
        d = {"hello": "world"}
        fp = Path(dp, "d.json")  # saving a json dict
        utils.save_dict(d=d, path=fp)
        d = utils.load_dict(path=fp)  # laoding a dict
        assert d["hello"] == "world"


def test_pad_array():
    """testing the pad_array function present in utils module"""
    arr = np.array([[1, 2], [1, 2, 3]], dtype="object")
    padded_arr = np.array([[1, 2, 0], [1, 2, 3]])
    assert np.array_equal(utils.pad_array(arr), padded_arr)


def test_collate_fn():
    """testing the collate_fn function present in utils module by
    passing array of two records
    """
    batch = {
        "ids": np.array([[1, 2], [1, 2, 3]], dtype="object"),
        "masks": np.array([[1, 1], [1, 1, 1]], dtype="object"),
        "targets": np.array([3, 1]),
    }
    processed_batch = utils.collate_fn(batch)
    expected_batch = {
        "ids": torch.as_tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.int32, device=get_device()),
        "masks": torch.as_tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.int32, device=get_device()),
        "targets": torch.as_tensor([3, 1], dtype=torch.int64, device=get_device()),
    }
    for ind in batch:
        assert torch.allclose(processed_batch[ind], expected_batch[ind])


@pytest.mark.parametrize(
    "dic, keys, list_of_dic",
    [
        ({"a": [1, 2], "b": [1, 2]}, ["a", "b"], [{"a": 1, "b": 1}, {"a": 2, "b": 2}]),
        ({"a": [1, 2], "b": [1, 2]}, ["a"], [{"a": 1}, {"a": 2}]),
    ],
)
def test_dict_to_list(dic, keys, list_of_dic):
    """testing the dict_to_list function present in utils module"""
    assert utils.dict_to_list(dic, keys=keys) == list_of_dic
