"""
DeepAR — global probabilistic forecaster with two backends:

  backend="local"      → gluonts.torch DeepAREstimator on this machine.
                         Free, slow on CPU (~30-60min for ~1k SKUs / 50 epochs),
                         fast on Mac MPS / NVIDIA. No AWS needed.
  backend="sagemaker"  → AWS SageMaker built-in DeepAR algorithm. Uses the $100
                         credit. Parallel training on GPU instances; pay per
                         second. Skeleton below — wire to your AWS_SAGEMAKER_ROLE.

Both backends produce the same `.forecast(sku, horizon)` API so the registry
and `select_best_model` don't care which one ran.

Loss: NegativeBinomialOutput (gluonts default for counts; non-negative,
overdispersion-friendly) — aligns with WMAPE on intermittent SKU demand.

Pseudocode for SageMaker (when you decide to spend AWS credit):

    1) write `weekly_sku.pivot(...).fillna(0)` to S3 as JSON Lines (one entry
       per SKU: {"start": ..., "target": [...], "cat": [...]}).
    2) Estimator(image_uri=..., role=AWS_SAGEMAKER_ROLE, instance_type="ml.c5.xlarge",
                 hyperparameters={"prediction_length": 12, "context_length": 52,
                                  "freq": "W", "epochs": 50, "num_layers": 2}).fit(s3_path)
    3) predictor = estimator.deploy(...) OR batch_transform(...) to get forecasts back.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

PRED_LEN = 12
CTX_LEN = 52
FREQ = "W-MON"


class DeepARWrapper:
    def __init__(self, context_length: int = CTX_LEN, prediction_length: int = PRED_LEN,
                 backend: str = "local"):
        if backend not in ("local", "sagemaker"):
            raise ValueError(f"backend must be 'local' or 'sagemaker', got {backend!r}")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.backend = backend
        self.predictor = None       # populated after fit()
        self._sku_index: dict = {}  # sku -> position in the dataset (for local)
        self._panel: pd.DataFrame | None = None

    # ─────────────────────────── fit ────────────────────────────

    def fit(self, weekly_sku: pd.DataFrame, skus: list[str], max_epochs: int = 50):
        if self.backend == "local":
            self._fit_local(weekly_sku, skus, max_epochs)
        else:
            self._fit_sagemaker(weekly_sku, skus, max_epochs)
        return self

    def _fit_local(self, weekly_sku: pd.DataFrame, skus: list[str], max_epochs: int):
        from gluonts.torch.model.deepar import DeepAREstimator
        from gluonts.torch.distributions import NegativeBinomialOutput
        from gluonts.dataset.pandas import PandasDataset

        panel = (
            weekly_sku[weekly_sku["StockCode"].isin(skus)]
            .pivot(index="Week", columns="StockCode", values="Quantity")
            .reindex(columns=skus)
        )
        full = pd.date_range(panel.index.min(), panel.index.max(), freq=FREQ)
        panel = panel.reindex(full).fillna(0).astype("float32")
        self._panel = panel
        self._sku_index = {s: i for i, s in enumerate(skus)}

        ds = PandasDataset(dict(panel.items()), target=None, freq=FREQ)
        estimator = DeepAREstimator(
            freq=FREQ,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=NegativeBinomialOutput(),
            trainer_kwargs={"max_epochs": max_epochs, "accelerator": "auto"},
        )
        self.predictor = estimator.train(ds)

    def _fit_sagemaker(self, weekly_sku: pd.DataFrame, skus: list[str], max_epochs: int):
        # See module docstring for full pseudocode. Sketch:
        # import boto3, sagemaker; from sagemaker.amazon.amazon_estimator import get_image_uri
        # role = os.environ["AWS_SAGEMAKER_ROLE"]; bucket = os.environ["AWS_S3_BUCKET"]
        # ... write JSON Lines to s3://bucket/train/ ...
        # estimator = sagemaker.estimator.Estimator(image_uri=..., role=role, ...)
        # estimator.fit({"train": "s3://.../train/", "test": "s3://.../test/"})
        # self.predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m5.large")
        raise NotImplementedError(
            "SageMaker backend: see module docstring for the 3-step pseudocode. "
            "Requires AWS_SAGEMAKER_ROLE + AWS_S3_BUCKET in .env."
        )

    # ─────────────────────────── predict ────────────────────────

    def forecast(self, sku: str, horizon: int | None = None) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("Call .fit(...) first.")
        if self.backend == "local":
            return self._forecast_local(sku, horizon or self.prediction_length)
        return self._forecast_sagemaker(sku, horizon or self.prediction_length)

    def _forecast_local(self, sku: str, horizon: int) -> np.ndarray:
        from gluonts.dataset.pandas import PandasDataset

        if sku not in self._sku_index:
            raise KeyError(f"SKU {sku!r} not in fitted panel.")
        # Predict on the full panel, return median for the requested SKU.
        ds = PandasDataset(dict(self._panel.items()), target=None, freq=FREQ)
        forecasts = list(self.predictor.predict(ds))
        idx = self._sku_index[sku]
        median = forecasts[idx].quantile(0.5)[:horizon]
        return np.maximum(0.0, np.asarray(median, dtype=float))

    def _forecast_sagemaker(self, sku: str, horizon: int) -> np.ndarray:
        # See module docstring. Use self.predictor.predict({"instances": [...]})
        # against the deployed endpoint.
        raise NotImplementedError("Wire to deployed SageMaker endpoint.")


def deepar_factory(wrapper: DeepARWrapper):
    """Adapt the global DeepAR wrapper to the per-SKU `(train, horizon)` registry contract.
    Used by playground/selection.py without extra orchestration."""
    def model(train: pd.Series, horizon: int) -> np.ndarray:
        sku = str(train.name) if train.name is not None else "<unknown>"
        return wrapper.forecast(sku, horizon)
    return model
