import datetime
import re
import typing
from pathlib import Path

import pandas as pd
from sklearn import linear_model


def get_step_time_from_file(
        file_path: typing.Union[str, Path],
        start_time: typing.Union[datetime.datetime, pd.Timedelta] = None,
        step_time: pd.Timedelta = None,
) -> pd.DataFrame:
    """
    Get seconds for each step from std.out.0000 of GRAPES GFS.

    Example output:

     begin operational forecast----
     begin of gcr  5.298061077439735E-005
     RES of gcr  1.062585033967659E-023 in           37 iterations
    Timing for processing for step        1:   15.51460 elapsed seconds.
     begin of gcr  6.001155404370011E-005
     RES of gcr  1.006302069134485E-023 in           37 iterations
    Timing for processing for step        2:    0.30060 elapsed seconds.
     begin of gcr  5.512293401798762E-005
     RES of gcr  1.112349462442307E-023 in           37 iterations

    Parameters
    ----------
    file_path: str or Path
    start_time: datetime.datetime or pd.Timedelta
        start time for cycle.
    step_time: pd.Timedelta
        time for each step, `MODEL_DT` for GRAPES GFS.

    Returns
    -------
    pandas.DataFrame:
        table data with "valid_time", "time", "step" and "ctime" as columns, and step number as index.
        If step_time is set, "forecast_time" and "forecast_hour" are added.
        And if step_time and start_time are both set, "valid_time" is added.

    Examples
    --------
    Get step time for GRAPES GFS GMF at 2020050200 cycle.
    >>> from nwpc_log_tool.data_finder import find_local_file
    >>> file_path = find_local_file(
    ...     "grapes_gfs_gmf/log/fcst_long_std_out",
    ...     file_path="2020050200",
    ... )
    >>> df = get_step_time_from_file(
    ...     file_path,
    ...     start_time=pd.to_datetime("2020050200", format="%Y%m%d%H"),
    ...     step_time=pd.Timedelta(seconds=300),
    ... )
    >>> df.head()
          time  step    ctime forecast_time  forecast_hour          valid_time
    1  15.3539     1  15.3539      00:05:00       0.083333 2020-05-02 00:05:00
    2   0.2767     2  15.6306      00:10:00       0.166667 2020-05-02 00:10:00
    3   0.3557     3  15.9863      00:15:00       0.250000 2020-05-02 00:15:00
    4   0.3482     4  16.3345      00:20:00       0.333333 2020-05-02 00:20:00
    5   0.3440     5  16.6785      00:25:00       0.416667 2020-05-02 00:25:00

    """
    p = re.compile(r"Timing for processing for step\s+(.+):\s+(.+) elapsed seconds\.")
    data = []
    index = []
    with open(file_path) as f:
        for line in f:
            m = p.match(line)
            if m is None:
                continue
            step = int(m.group(1))
            time = float(m.group(2))
            data.append({
                "time": time
            })
            index.append(step)
    df = pd.DataFrame(data, index=index)
    df["step"] = df.index
    df["ctime"] = df["time"].cumsum()
    if step_time is not None:
        df["forecast_time"] = step_time * df["step"]
        df["forecast_hour"] = df["forecast_time"] / pd.Timedelta(hours=1)
        if start_time is not None:
            df["valid_time"] = df["forecast_time"] + start_time
    return df


def train_linear_model(
        df: pd.DataFrame,
        x_label: str = "step",
) -> linear_model.LinearRegression:
    """
    Train linear regression model for step and ctime using scikit-learn.

    Parameters
    ----------
    df: pandas.DataFrame
    x_label: str
        label for X data. default is step.
        Set `forecast_hour` if `df` has `forecast_hour` column.

    Returns
    -------
    sklearn.linear_model.LinearRegression:

    Examples
    --------
    Use `df` from the example of ``get_step_time_from_file`` function.

    >>> import numpy as np
    >>> model = train_linear_model(df, x_label="forecast_hour")
    >>> model.predict(np.array([240]).reshape(-1, 1))/60
    [22.79910527]

    """
    X = df[x_label].values.reshape(-1, 1)
    y = df["ctime"]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model
