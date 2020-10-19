import datetime
import typing

import pandas as pd
import numpy as np
from sklearn import linear_model


def generate_message_data(
        start_time: typing.Union[datetime.datetime, pd.Timestamp],
        forecast_hour: float,
        forecast_length: int,
        current_record: pd.DataFrame,
        model: linear_model.LinearRegression,
) -> typing.Dict:
    coef = model.coef_[0]
    intercept = model.intercept_

    predict_minutes = model.predict(
        np.reshape(forecast_length, (-1, 1))
    )[0] / 60
    print(predict_minutes)

    start_time = pd.to_datetime(start_time, format="%Y%m%d%H")
    forecast_time = pd.Timedelta(f"{forecast_hour}h")
    valid_time = start_time + forecast_time

    predict_time = pd.Timedelta(f"{forecast_length}h")

    data = {
        "start_time": start_time.isoformat(),
        "request": {
            "forecast_time": forecast_time.isoformat(),
            "valid_time": valid_time.isoformat(),
        },
        "current": {
            "forecast_time": current_record["forecast_time"].isoformat(),
            "valid_time": current_record["valid_time"].isoformat(),
            "ctime": float(current_record["ctime"] / 60),
        },
        "model": {
            "type": "linear",
            "coef": coef,
            "intercept": intercept,
        },
        "predict": {
            "total": {
                "forecast_time": predict_time.isoformat(),
                "ctime": predict_minutes
            }
        }
    }
    return data
