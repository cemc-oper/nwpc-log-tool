import datetime
import re
from pathlib import Path

import pandas as pd
from sklearn import linear_model


def get_step_time_from_file(
        file_path: str or Path,
        start_time: datetime.datetime or pd.Timedelta = None,
) -> pd.DataFrame:
    """
    Get seconds for each step from ecflow job output grapes.1 of GRAPES TYM.

    Example output:

    Timing for processing for step 7165 (2020060723:24:00):          0.32850 elapsed seconds.
    Timing for processing for step 7165 (2020060723:24:00):          0.32837 cpu seconds.
     begin of gcr  5.219601769424760E-002
     RES of gcr  7.680596670152904E-013 in           15 iterations
    Timing for processing for step 7166 (2020060723:25:00):          0.32440 elapsed seconds.
    Timing for processing for step 7166 (2020060723:25:00):          0.32422 cpu seconds.
     begin of gcr  5.215694159471712E-002
     RES of gcr  8.024413137292737E-013 in           15 iterations

    Parameters
    ----------
    file_path: str or Path

    start_time: datetime.datetime or pandas.Timedelta

    Returns
    -------
    pandas.DataFrame:
        table data with "valid_time", "time", "step", "ctime", "forecast_time" and "forecast_hour" as columns,
        and step number as index.
    """

    # elapsed seconds has some problem in one cycle. use cpu seconds instead
    p = re.compile(r"Timing for processing for step\s+(.+) \((.*)\):\s+(.+) cpu seconds\.")
    data = []
    index = []
    with open(file_path) as f:
        for line in f:
            m = p.match(line)
            if m is None:
                continue
            step = int(m.group(1))
            valid_time = pd.to_datetime(m.group(2), format='%Y%m%d%H:%M:%S')
            time = float(m.group(3))
            data.append({
                "valid_time": valid_time,
                "time": time
            })
            index.append(step)
    df = pd.DataFrame(data, index=index)
    df["step"] = df.index
    df["ctime"] = df["time"].cumsum()
    if start_time is None:
        start_time = df["valid_time"].iloc[0]
    df["forecast_time"] = df["valid_time"] - start_time
    df["forecast_hour"] = df["forecast_time"] / pd.Timedelta(hours=1)
    return df


def get_output_time_from_file(file_path: str or Path) -> pd.DataFrame:
    """
    Get seconds for modelvar output from ecflow job output grapes.1 of GRAPES TYM.

    Example output:

    Timing for processing for step 120 (2020060301:59:00):          0.52200 elapsed seconds.
    Timing for processing for step 120 (2020060301:59:00):          0.52184 cpu seconds.
     output modelvar use    1.19464898109436      seconds
      post grib2 compress and output use    1.47139906883240       seconds.
      output grib2 compress and output use   0.219344139099121       seconds.
     begin of gcr  4.231574434342701E-005
     RES of gcr  5.927635314226707E-013 in           17 iterations

    Parameters
    ----------
    file_path: str or Path

    Returns
    -------
    pandas.DataFrame:
        table data with "time" as column.
    """
    p = re.compile(r"output modelvar use\s+([0-9.]*)\s+seconds")
    data = []
    with open(file_path) as f:
        for line in f:
            m = p.search(line)
            if m is None:
                continue
            time = float(m.group(1))
            data.append({
                "time": time
            })
    df = pd.DataFrame(data)
    return df


def train_linear_model(df: pd.DataFrame):
    """
    Train linear regression model for forecast_hour and ctime using scikit-learn.

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    sklearn.linear_model.LinearRegression:

    """
    df = df.copy()
    X = df["forecast_hour"].values.reshape(-1, 1)
    y = df["ctime"]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model
