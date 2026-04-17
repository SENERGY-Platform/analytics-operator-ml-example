import datetime
import typing


def to_epoch_seconds(value: typing.Any) -> float:
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


def extract_timestamp_and_value(row: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Tuple[float, float]]:
    ts = row.get("timestamp", row.get("time", row.get("ts")))
    value = row.get("value")
    if ts is None or value is None:
        return None
    return to_epoch_seconds(ts), float(value)
