import datetime
from pathlib import Path

import pandas as pd

from .grapes_meso import (
    get_step_time_from_file as meso_get_step_time_from_file,
    get_output_time_from_file,
    train_linear_model,
)


def get_step_time_from_file(
        file_path: str or Path,
        start_time: datetime.datetime or pd.Timedelta = None,
        time_type: str = "cpu",
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

    time_type: str

    Returns
    -------
    pandas.DataFrame:
        table data with "valid_time", "time", "step", "ctime", "forecast_time" and "forecast_hour" as columns,
        and step number as index.
    """

    # elapsed seconds has some problem in one cycle. use cpu seconds instead
    return meso_get_step_time_from_file(
        file_path=file_path,
        start_time=start_time,
        time_type=time_type,
    )
