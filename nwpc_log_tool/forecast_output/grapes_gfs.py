import re
from pathlib import Path

import pandas as pd
from sklearn import linear_model


def get_step_time_from_file(file_path: str or Path):
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

    Returns
    -------
    pandas.DataFrame:
        table data with "valid_time", "time", "step" and "ctime" as columns, and step number as index.
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
    return df


def train_linear_model(df: pd.DataFrame):
    """
    Train linear regression model for step and ctime using scikit-learn.

    Parameters
    ----------
    df: pandas.DataFrame

    Returns
    -------
    sklearn.linear_model.LinearRegression:

    """
    X = df["step"].values.reshape(-1, 1)
    y = df["ctime"]
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model
