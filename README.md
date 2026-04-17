# analytics-operator-ml-example

Example machine-learning analytics operator built on `operator-lib`, with Ray-powered training/tuning and PyTorch inference.

## Why `uv run` is mandatory here

This project must be executed with `uv run`.

- Ray worker processes inherit environment and interpreter context from the launching process.
- If you start with plain `python`, Ray/Tune workers can run with a different interpreter or missing dependencies.
- `uv run` guarantees the locked environment from `pyproject.toml` and `uv.lock` is used consistently by the driver and worker processes.

Use:

```bash
uv run main.py
```

Avoid:

```bash
python main.py
```

## Project structure

- `main.py`: bootstrap/entrypoint for the operator runtime
- `op.py`: operator implementation (inference, training trigger, retraining policy)
- `training.py`: model training + tuning pipeline (Ray + Torch)
- `util.py`: data extraction and timestamp normalization helpers
- `pyproject.toml`: dependency/runtime definition for `uv`
- `.vscode/`: task/debug setup that enforces `uv run`

## `main.py`

`main.py` is intentionally small because it delegates runtime lifecycle to `operator-lib`.

What it does:

1. Imports the local `Operator` class from `op.py`.
2. Imports `OperatorLib` from the external `operator-lib` package.
3. Instantiates `OperatorLib(Operator(), name=...)` in `if __name__ == "__main__":`.

Why this matters:

- `OperatorLib` owns process-level concerns (config wiring, stream handling, lifecycle orchestration, model registry integration).
- Your custom logic stays in `Operator` (inference/training methods), while the framework handles execution flow.
- The `name` identifies this operator implementation in the surrounding platform context.

Design note:

- Keeping `main.py` as a thin bootstrap reduces accidental side effects and makes startup behavior predictable.

## `op.py`

`op.py` defines the operator behavior by subclassing `MLOperator`.

### Classes

- `CustomConfig(Config)`: placeholder for typed custom configuration extensions.
- `Operator(MLOperator)`: concrete implementation used by `main.py`.

### Operator lifecycle methods

#### `init(self, *args, **kwargs)`

- Currently delegates to `super().init(...)`.
- This is the extension point for custom startup logic if you need some.

#### `infer(self, model, data, selector, device_id, timestamp)`

Primary online inference path.

Flow:

1. Reads `value` from incoming `data`.
2. If missing, returns `(None, None)` to signal no output/no model replacement.
3. Builds payload with:
	- `timestamp`
	- `value` coerced to `float`
4. If a trained model exists, tries `model.predict(payload)`.
5. On prediction success: returns `(prediction, None)`.
6. On prediction error: swallows exception and returns `(None, None)`.

Return contract:

- First tuple item: inference result to publish.
- Second tuple item: optional immediate model replacement (unused here, so `None`). In case the model is updated by inference, it can be returned here.

#### `train(self, _, logger)`

Training trigger called by framework.

Flow:

1. Calls `provide_historic_data(datetime.timedelta(days=3))`.
2. Expects at least one Ray Dataset; raises `RuntimeError` otherwise.
3. Launches distributed training via `train_one_hour_ahead_model.remote(datasets, logger)`.
4. Uses `ray.get(...)` to retrieve the trained PythonModel wrapper.
5. Returns the model for registration/use by the framework.

Why this split is good:

- `op.py` decides *when and with what data* to train.
- `training.py` owns *how* to train.

#### `need_retraining(self, _)`

- Returns `True` unconditionally (current placeholder policy).
- Framework will always consider retraining needed.
- Replace with a production policy (time-based, drift-based, quality-based).

## `training.py`

`training.py` contains the ML pipeline implementation.

At a high level it:

1. Parses and merges historical datasets.
2. Builds one-hour-ahead supervised pairs.
3. Computes normalization statistics.
4. Tunes hyperparameters with Ray Tune.
5. Trains a small feed-forward PyTorch regressor.
6. Restores best checkpoint and wraps it as `TorchOneHourAheadModel`.
7. Logs params/metrics/artifacts/traces through `TrainMlflowLogger`.

Key pieces:

- `_StreamingOneHourPairBuilder`: batch-safe pair construction with carry-over rows.
- `train_one_hour_ahead_model` (`@ray.remote`): full distributed training+tuning flow.
- `TorchOneHourAheadModel`: inference wrapper with timestamp parsing and denormalization.

## `pyproject.toml` and `uv`

`pyproject.toml` defines:

- Python version pin: `==3.10.20`
- Core dependencies: `torch`, `ray[client,train]`, `operator-lib`, `tqdm`
- PyTorch CPU wheel index via `extra-index-url`
- `tool.uv.package = false` (workspace is app-style, not a published package)

`uv` usage in this project:

1. Sync environment from lock file:

```bash
uv sync
```

2. Run the app with resolved environment:

```bash
uv run main.py
```

3. Run arbitrary commands in the same environment:

```bash
uv run python -c "import ray, torch; print(ray.__version__)"
```

## Running with VS Code (`.vscode`)

This repo already includes ready-to-use VS Code config.

### Debug run

Debug configuration uses:

- `python: ${workspaceFolder}/.vscode/uv-run-python.sh`

The wrapper script:

1. Activates `.venv`
2. Executes `uv run python "$@"`

This keeps debug sessions aligned with the same `uv` environment and avoids Ray worker mismatch issues.

## Minimal local run checklist

1. Install `uv` if needed.
2. `uv sync`
3. Ensure required env vars are set (or run via provided VS Code task).
4. Start with `uv run main.py`

## Notes

- Do not switch to plain `python main.py` unless you intentionally want to bypass the locked `uv` environment.
- If Ray/Tune behaves inconsistently, first verify the process was launched through `uv run` (terminal, task, and debugger).
