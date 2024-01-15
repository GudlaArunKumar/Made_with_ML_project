import pytest
import utils

from ml_scripts import train


@pytest.mark.training  # use training explicitly to run this test
def test_train_model(dataset_loc):
    experiment_name = utils.generate_experiment_name(prefix="test_train")
    # train_loop_config = {"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}
    result = train.train_model(
        experiment_name=experiment_name,
        dataset_loc=dataset_loc,
        num_workers=5,
        cpu_per_worker=1,
        gpu_per_worker=1,
        num_epochs=2,
        num_samples=512,
        batch_size=256,
        results_fp=None,
    )
    utils.delete_experiment(experiment_name=experiment_name)
    train_loss_list = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_list[0] > train_loss_list[1]  # loss decreased for subsequent epochs
