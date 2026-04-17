import datetime
import json
import os
import tempfile
import typing

import numpy as np
import pandas as pd
import ray

import ray.tune
import ray.train
from mlflow.pyfunc import PythonModel

from util import extract_timestamp_and_value
from operator_lib.util.helpers import TrainMlflowLogger


import torch
import torch.nn as nn


# Centralized training configuration for quick, safe tuning.
HORIZON_SECONDS = 60 * 60
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 2
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32
NUM_WORKERS = 2
NORMALIZATION_EPS = 1e-6
SAMPLE_ROWS_TO_LOG = 100
LOG_DATASET_SAMPLE_ROWS = False
MODEL_LAYERS_DESC = f"Linear({INPUT_DIM},{HIDDEN_DIM})-ReLU-Linear({HIDDEN_DIM},{OUTPUT_DIM})"
TUNE_NUM_SAMPLES = 8
TUNE_LR_MIN = 1e-4
TUNE_LR_MAX = 5e-2
TUNE_EPOCH_OPTIONS = (2, 4, 8)
TUNE_BATCH_SIZE_OPTIONS = (8, 16, 32)


def build_model(nn_module: typing.Any) -> nn.Module:
    return nn_module.Sequential(
        nn_module.Linear(INPUT_DIM, HIDDEN_DIM),
        nn_module.ReLU(),
        nn_module.Linear(HIDDEN_DIM, OUTPUT_DIM),
    )


def resolve_batch_size(num_examples: int) -> int:
    return min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, num_examples))


def resolve_tune_batch_size_options(num_examples: int) -> typing.List[int]:
    max_allowed = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, num_examples))
    options = [int(size)
               for size in TUNE_BATCH_SIZE_OPTIONS if size <= max_allowed]
    if len(options) == 0:
        return [resolve_batch_size(num_examples)]
    return sorted(set(options))


class _StreamingOneHourPairBuilder:
    """Build one-hour-ahead training pairs while carrying unresolved rows across batches."""

    def __init__(self) -> None:
        self._pending_ts = np.array([], dtype=np.float64)
        self._pending_values = np.array([], dtype=np.float64)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        if batch.empty:
            return pd.DataFrame(columns=["ts", "value", "target"])

        ordered = batch.sort_values("ts").reset_index(drop=True)
        ts = ordered["ts"].astype(float).to_numpy()
        values = ordered["value"].astype(float).to_numpy()

        if self._pending_ts.size > 0:
            ts = np.concatenate((self._pending_ts, ts))
            values = np.concatenate((self._pending_values, values))

        target_idxs = np.searchsorted(ts, ts + HORIZON_SECONDS, side="left")
        resolved = target_idxs < len(ts)

        self._pending_ts = ts[~resolved]
        self._pending_values = values[~resolved]

        if not np.any(resolved):
            return pd.DataFrame(columns=["ts", "value", "target"])

        return pd.DataFrame(
            {
                "ts": ts[resolved],
                "value": values[resolved],
                "target": values[target_idxs[resolved]],
            }
        )


class TorchOneHourAheadModel(PythonModel):
    def __init__(
        self,
        model: nn.Module,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        self._model = model
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._y_std = y_std

    @staticmethod
    def _to_epoch_seconds(value: typing.Any) -> float:
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.timezone.utc)
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parsed = datetime.datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            return float(parsed.timestamp())
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    def _predict_single(self, ts: typing.Any, value: typing.Any) -> float:
        x = torch.tensor(
            [[self._to_epoch_seconds(ts), float(value)]], dtype=torch.float32)
        x = (x - self._x_mean) / self._x_std
        with torch.no_grad():
            y_hat_norm = self._model(x)
        y_hat = (y_hat_norm * self._y_std) + self._y_mean
        return float(y_hat.item())

    def predict(self, context: typing.Any, model_input: list[typing.Any]) -> typing.Union[float, typing.List[float]]:
        if isinstance(model_input, dict):
            ts = model_input.get("timestamp")
            value = model_input.get("value")
            if ts is None or value is None:
                raise ValueError(
                    "model_input must include 'timestamp' and 'value'.")
            return self._predict_single(ts, value)
        if isinstance(model_input, list):
            return [self.predict(context, item) for item in model_input]
        if hasattr(model_input, "to_dict"):
            rows = model_input.to_dict(orient="records")
            return [self.predict(context, row) for row in rows]
        raise TypeError(f"Unsupported model_input type: {type(model_input)}")

@ray.remote
def train_one_hour_ahead_model(ds: typing.List[ray.ObjectRef[ray.data.Dataset]], mlflow_logger: TrainMlflowLogger) -> TorchOneHourAheadModel:
    with mlflow_logger.trace("operator_training_pipeline"):
        parsed_datasets: typing.List[ray.data.Dataset] = []

        with mlflow_logger.trace("parse_input_datasets"):
            for ds_ref in ds:
                dataset = ray.get(ds_ref) if isinstance(
                    ds_ref, ray.ObjectRef) else ds_ref

                parsed_dataset = dataset.map(
                    lambda row: (lambda p: {"ts": p[0], "value": p[1]} if p is not None else None)(
                        extract_timestamp_and_value(row)
                    )
                ).filter(lambda row: row is not None)
                parsed_datasets.append(parsed_dataset)

        if len(parsed_datasets) == 0:
            raise RuntimeError("Need at least one dataset to train the model.")

        with mlflow_logger.trace("merge_datasets"):
            merged_dataset = parsed_datasets[0]
            for additional_dataset in parsed_datasets[1:]:
                merged_dataset = merged_dataset.union(additional_dataset)

        with mlflow_logger.trace("sort_materialize"):
            sorted_dataset = merged_dataset
            merged_dataset.sort("ts").materialize()

        with mlflow_logger.trace("count_points"):
            num_points = sorted_dataset.count()
        if num_points < 2:
            raise RuntimeError(
                "Need at least two timestamp/value points to train the model.")

        with mlflow_logger.trace("build_pairs_materialize"):
            training_pairs_dataset = sorted_dataset.map_batches(
                _StreamingOneHourPairBuilder,
                batch_format="pandas",
                batch_size=8192,
            ).materialize()

        with mlflow_logger.trace("count_training_pairs"):
            num_training_pairs = training_pairs_dataset.count()
        if num_training_pairs == 0:
            raise RuntimeError(
                "No valid one-hour training pairs found in historic data.")

        with mlflow_logger.trace("aggregate_pair_stats"):
            pair_stats = training_pairs_dataset.aggregate(
                ray.data.aggregate.Mean("ts"),
                ray.data.aggregate.Std("ts", ddof=0),
                ray.data.aggregate.Mean("value"),
                ray.data.aggregate.Std("value", ddof=0),
                ray.data.aggregate.Mean("target"),
                ray.data.aggregate.Std("target", ddof=0),
            )

        x_mean_np = np.array(
            [pair_stats["mean(ts)"], pair_stats["mean(value)"]], dtype=np.float64)
        x_std_np = np.array(
            [pair_stats["std(ts)"], pair_stats["std(value)"]], dtype=np.float64)
        y_mean_np = np.array([pair_stats["mean(target)"]], dtype=np.float64)
        y_std_np = np.array([pair_stats["std(target)"]], dtype=np.float64)

        x_mean = torch.tensor([x_mean_np.tolist()], dtype=torch.float32)
        x_std = torch.tensor([x_std_np.tolist()],
                             dtype=torch.float32).clamp_min(NORMALIZATION_EPS)
        y_mean = torch.tensor([y_mean_np.tolist()], dtype=torch.float32)
        y_std = torch.tensor([y_std_np.tolist()],
                             dtype=torch.float32).clamp_min(NORMALIZATION_EPS)
        batch_size = resolve_batch_size(num_training_pairs)

        # Log static run metadata once (params, tags, and dataset summary artifact).
        mlflow_logger.set_tags({
            "training.framework": "ray-train-torch",
            "training.task": "one-hour-ahead-regression",
            "training.trace": "enabled",
        })
        mlflow_logger.log_params({
            "horizon_seconds": HORIZON_SECONDS,
            "num_points": num_points,
            "num_training_pairs": num_training_pairs,
            "num_input_datasets": len(parsed_datasets),
            "model.layers": MODEL_LAYERS_DESC,
        })
        mlflow_logger.log_dict({
            "counts": {
                "num_points": num_points,
                "num_training_pairs": num_training_pairs,
                "num_input_datasets": len(parsed_datasets),
            },
            "normalization": {
                "x_mean": x_mean.squeeze(0).tolist(),
                "x_std": x_std.squeeze(0).tolist(),
                "y_mean": y_mean.squeeze(0).tolist(),
                "y_std": y_std.squeeze(0).tolist(),
            },
        }, "training_dataset_summary.json")

        if LOG_DATASET_SAMPLE_ROWS:
            with mlflow_logger.trace("sample_training_pairs"):
                sample_rows = [
                    {
                        "ts": float(row["ts"]),
                        "value": float(row["value"]),
                        "target_value_plus_1h": float(row["target"]),
                    }
                    for row in training_pairs_dataset.take(SAMPLE_ROWS_TO_LOG)
                ]
            mlflow_logger.log_text(json.dumps(
                sample_rows, indent=2), "training_pairs_sample.json")

        model = build_model(nn)
        with mlflow_logger.trace("repartition_train_dataset"):
            train_dataset = training_pairs_dataset.repartition(NUM_WORKERS)

        def train_loop_per_worker(config: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
            local_model = build_model(torch.nn)

            optimizer = torch.optim.Adam(
                local_model.parameters(), lr=float(config["lr"]))
            loss_fn = torch.nn.MSELoss()

            x_mean_tensor = torch.tensor(
                config["x_mean"], dtype=torch.float32)
            x_std_tensor = torch.tensor(
                config["x_std"], dtype=torch.float32).clamp_min(NORMALIZATION_EPS)
            y_mean_tensor = torch.tensor(
                config["y_mean"], dtype=torch.float32)
            y_std_tensor = torch.tensor(
                config["y_std"], dtype=torch.float32).clamp_min(NORMALIZATION_EPS)

            dataset_shard = config["datasets"]["train"]

            final_loss = 0.0
            for epoch in range(int(config["num_epochs"])):
                batch_iter = dataset_shard.iter_torch_batches(
                    batch_size=int(config["batch_size"]),
                    dtypes={"ts": torch.float32,
                            "value": torch.float32, "target": torch.float32},
                )

                running_loss = 0.0
                running_mae_norm = 0.0
                running_rmse_norm = 0.0
                running_mae = 0.0
                running_rmse = 0.0
                running_batch_size = 0
                steps = 0
                for batch in batch_iter:
                    batch_x = torch.stack(
                        (batch["ts"], batch["value"]), dim=1)
                    batch_y = batch["target"].unsqueeze(1)

                    batch_x = (batch_x - x_mean_tensor) / x_std_tensor
                    batch_y = (batch_y - y_mean_tensor) / y_std_tensor

                    optimizer.zero_grad()
                    pred = local_model(batch_x)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    optimizer.step()

                    # Track both normalized metrics (optimization space) and
                    # denormalized metrics (original value space).
                    mae_norm = torch.mean(
                        torch.abs(pred - batch_y))
                    rmse_norm = torch.sqrt(
                        torch.mean((pred - batch_y) ** 2))
                    pred_denorm = (pred * y_std_tensor) + y_mean_tensor
                    target_denorm = (batch_y * y_std_tensor) + y_mean_tensor
                    mae = torch.mean(
                        torch.abs(pred_denorm - target_denorm))
                    rmse = torch.sqrt(torch.mean(
                        (pred_denorm - target_denorm) ** 2))

                    running_loss += float(loss.item())
                    running_mae_norm += float(mae_norm.item())
                    running_rmse_norm += float(rmse_norm.item())
                    running_mae += float(mae.item())
                    running_rmse += float(rmse.item())
                    running_batch_size += int(batch_x.shape[0])
                    steps += 1

                final_loss = running_loss / max(1, steps)
                final_mae_norm = running_mae_norm / max(1, steps)
                final_rmse_norm = running_rmse_norm / max(1, steps)
                final_mae = running_mae / max(1, steps)
                final_rmse = running_rmse / max(1, steps)

                # Emit metrics and a checkpoint each epoch so the final trained state is recoverable.
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    state_dict = local_model.module.state_dict() if hasattr(
                        local_model, "module") else local_model.state_dict()
                    torch.save(state_dict, os.path.join(
                        temp_checkpoint_dir, "model.pt"))
                    checkpoint = ray.tune.Checkpoint.from_directory(temp_checkpoint_dir)
                    metrics = {
                        "loss": final_loss,
                        "mae_norm": final_mae_norm,
                        "rmse_norm": final_rmse_norm,
                        "mae": final_mae,
                        "rmse": final_rmse,
                        "learning_rate": float(config["lr"]),
                        "steps": steps,
                        "samples": running_batch_size,
                        "epoch": epoch,
                    }
                    ray.tune.report(
                        metrics,
                        checkpoint=checkpoint,
                    )
            print(
                f"Finished training loop with final metrics: loss={final_loss}, mae_norm={final_mae_norm}, rmse_norm={final_rmse_norm}, mae={final_mae}, rmse={final_rmse}")
            return metrics

        tune_batch_options = resolve_tune_batch_size_options(
            num_training_pairs)

        with mlflow_logger.trace("hyperparameter_tuning_and_training"):
            tuner = ray.tune.Tuner(
                train_loop_per_worker,
                param_space={
                    "scaling_config": ray.train.ScalingConfig(num_workers=NUM_WORKERS, use_gpu=False),
                    "datasets": {"train": train_dataset},
                    "lr": ray.tune.loguniform(TUNE_LR_MIN, TUNE_LR_MAX),
                    "num_epochs": ray.tune.choice(list(TUNE_EPOCH_OPTIONS)),
                    "batch_size": ray.tune.choice(tune_batch_options),
                    "x_mean": x_mean.squeeze(0).tolist(),
                    "x_std": x_std.squeeze(0).tolist(),
                    "y_mean": y_mean.squeeze(0).tolist(),
                    "y_std": y_std.squeeze(0).tolist(),
                },
                tune_config=ray.tune.TuneConfig(
                    metric="rmse",
                    mode="min",
                    num_samples=TUNE_NUM_SAMPLES,
                ),
                run_config=ray.tune.RunConfig(
                    storage_path="/storage",  # This is a shared storage path on our cluster
                ),
                _tuner_kwargs={
                  #  "verbose": 1,  # This is required to prevent a crash...
                }
            )
            tune_results = tuner.fit()
            
        mlflow_logger.log_table(tune_results.get_dataframe(), "tune_results.json")

        best_result = tune_results.get_best_result(
            metric="rmse", mode="min")
        mlflow_logger.log_params(
            {
                "tuning.enabled": True,
                "tuning.num_samples": TUNE_NUM_SAMPLES,
                "tuning.best_lr": float(best_result.config["lr"]), 
            }
        )
        mlflow_logger.log_params(best_result.config)
        mlflow_logger.log_metrics(best_result.metrics)
        with mlflow_logger.trace("load_final_checkpoint"):
            with best_result.checkpoint.as_directory() as checkpoint_dir:
                model_state_dict = torch.load(os.path.join(
                    checkpoint_dir, "model.pt"), map_location="cpu")
                model.load_state_dict(model_state_dict)
        mlflow_logger.finish(status="FINISHED")

        return TorchOneHourAheadModel(model, x_mean, x_std, y_mean, y_std)
