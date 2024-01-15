import pandas as pd
import pytest
import ray

from ml_scripts import data


# two defined fixtures acts as a constant variable in this module
@pytest.fixture(scope="module")
def df():
    data = [{"title": "nlp is booming", "description": "LLM and GenAI are the reason", "tag": "nlp"}]
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def class_to_index():
    class_to_index = {"c0": 0, "c1": 1}
    return class_to_index


def test_load_data(dataset_loc):
    """testing the load_data function present in the data module"""
    num_samples = 15
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=num_samples)
    assert ds.count() == num_samples


def test_stratify_split():
    """testing the stratify_split function present in the data module
    by splitting a random dataset into 50:50 ratio
    """
    n_per_class = 10
    targets = n_per_class * ["c0"] + n_per_class * ["c1"]
    ds = ray.data.from_items([dict(target=t) for t in targets])
    train_ds, val_ds = data.stratify_split(ds, stratify="target", test_size=0.5)
    train_target_counts = train_ds.to_pandas().target.value_counts().to_dict()
    val_target_counts = val_ds.to_pandas().target.value_counts().to_dict()
    assert train_target_counts == val_target_counts


@pytest.mark.parametrize(
    "text, stp_wrd, cleaned_text",
    [
        ("hello", "[]", "hello"),
        ("hi, how are you?", ["are"], "hi how you"),
        ("hi yous", ["you"], "hi yous"),
    ],
)
def test_clean_text(text, stp_wrd, cleaned_text):
    """testing the clean_text function present in the data module
    by passing multiple test inputs
    """
    assert data.clean_text(text=text, stopwords=stp_wrd) == cleaned_text


def test_preprocess(df, class_to_index):
    """testing the preprocess function present in the data module"""
    assert "text" not in df.columns
    outputs = data.preprocess(df=df, class_to_index=class_to_index)
    assert set(outputs) == {"ids", "masks", "targets"}


def test_fit_transform(dataset_loc, preprocessor):
    """testing the fit and transform method in CustomPreprocessor class
    present in the data module
    """
    ds = data.load_data(dataset_loc=dataset_loc)
    preprocessor = preprocessor.fit(ds)
    preprocessed_ds = preprocessor.transform(ds)
    assert len(preprocessor.class_to_index) == 4
    assert ds.count() == preprocessed_ds.count()
