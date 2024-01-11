import argparse
import os
from http import HTTPStatus
from typing import Dict 

import ray 
from fastapi import FastAPI 
from ray import serve 
from starlette.requests import Request 
import mlflow

from ml_scripts import evaluate, predict
from ml_scripts.config import MLFLOW_TRACKING_URI


# define and initiating fastapi application 
app = FastAPI(
    title= "ML Content classification Application",
    description= "Built this application by fine tuning LLM model and implemented with Mlops \
          framework which classify incoming ML content into four different categories",
    version="0.1"
)

# for scaling the inference pipeline
@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 7, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: float = 0.9):
        """Initialize the model and fastapi api"""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @app.get("/")
    def _index(self) -> Dict:
        """
        Health check of the api
        """
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {}
        }

        return response
    

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the Run Id """
        return {"run_id": self.run_id}
    
    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        """Function to make a batch prediction on a datatset """
        data = await request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}
    
    @app.post("/predict/")
    async def _predict(self, request: Request):
        """ Real Time Inference API to make quick prediction on individual data point"""
        data = await request.json()
        sample_ds = ray.data.from_items([{"title": data.get("title", ""), "description": data.get("description", ""), "tag": ""}])
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return {"results": results}
    

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="Run ID to select best checkpoint")
    parser.add_argument("--threshold", type=float, default=0.9, help="thresold for other class label")
    args = parser.parse_args()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold), route_prefix="/")
