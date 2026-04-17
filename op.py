from operator_lib.util import Config, MLOperator
from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


import ray
import typing
import datetime
from mlflow.pyfunc import PyFuncModel, PythonModel

from training import train_one_hour_ahead_model

class CustomConfig(Config):
    pass
    
class Operator(MLOperator):
    configType = CustomConfig

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

    def infer(self, model: typing.Optional[PyFuncModel], data: typing.Dict[str, typing.Any], selector: str, device_id: str, timestamp: datetime.datetime) -> typing.Tuple[typing.Optional[typing.Any], typing.Optional[PythonModel]]:
        current_value = data.get("value")
        if current_value is None:
            return None, None

        payload = {
            "timestamp": timestamp,
            "value": float(current_value),
        }

        if model is not None:
            try:
                prediction = model.predict(payload)
                return prediction, None
            except Exception:
                pass

        return None, None

    def train(self, _: typing.Optional[PyFuncModel], logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
        datasets = provide_historic_data(
            datetime.timedelta(days=365))
        if len(datasets) == 0:
            raise RuntimeError("Expected at least one ray Dataset!")
        return ray.get(train_one_hour_ahead_model.remote(datasets, logger))

    def need_retraining(self, _: typing.Optional[PyFuncModel]) -> bool:
        return True # You should implement a proper retraining strategy here, e.g. based on the age of the model or the distribution of incoming data.
