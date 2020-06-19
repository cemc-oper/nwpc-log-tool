import datetime

import pandas as pd
import numpy as np


def generate_message_data(
        start_time: datetime.datetime or pd.Timestamp,
        forecast_hour: int,
        forecast_length: int,
        current_record: pd.DataFrame,
        model,
) -> dict:
    coef = model.coef_[0]
    intercept = model.intercept_

    predict_minutes = model.predict(np.reshape(forecast_length, (-1, 1)))[0] / 60
    print(predict_minutes)

    start_time = pd.to_datetime(start_time, format="%Y%m%d%H")
    forecast_time = pd.Timedelta(f"{forecast_hour}h")
    valid_time = start_time + forecast_time

    data = {
        "start_time": start_time.isoformat(),
        "request": {
            "forecast_time": f"{forecast_hour}h",
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
                "forecast_time": f"{forecast_length}h",
                "ctime": predict_minutes
            }
        }
    }
    return data
