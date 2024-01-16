import ray

from ml_scripts import predict


def get_label(text, predictor):
    """Function to retrieve label from the model given data and the predictor"""
    sample_ds = ray.data.from_items([{"title": text, "description": "", "tag": "other"}])
    results = predict.predict_proba(df=sample_ds, predictor=predictor)
    return results[0]["prediction"]
