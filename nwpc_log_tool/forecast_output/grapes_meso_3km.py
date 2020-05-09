import re
from pathlib import Path

import pandas as pd


def get_step_time_from_file(file_path: str or Path) -> pd.DataFrame:
    """
    Get seconds for each step from ecflow job output fcst.1 of GRAPES MESO 3KM.

    Example output:

    Timing for processing for step 1 (2020050900:00:00):         10.60030 elapsed seconds.
    Timing for processing for step 1 (2020050900:00:00):          8.34263 cpu seconds.
     begin of gcr  3.311504627275603E-004
     RES of gcr  7.923976079163864E-013 in           37 iterations
     warm start: grid%do_cld = T
    Timing for processing for step 2 (2020050900:00:30):          0.73180 elapsed seconds.
    Timing for processing for step 2 (2020050900:00:30):          0.73085 cpu seconds.
     begin of gcr  1.786058322574234E-004
     RES of gcr  9.312605195708282E-013 in           36 iterations

    Parameters
    ----------
    file_path: str or Path

    Returns
    -------
    pandas.DataFrame:
        table data with "valid_time" and "time" as columns, and step number as index.
    """
    p = re.compile(r"Timing for processing for step\s+(.+) \((.*)\):\s+(.+) elapsed seconds\.")
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
    return df


def get_output_time_from_file(file_path: str or Path) -> pd.DataFrame:
    """
    Get seconds for modelvar output from ecflow job output fcst.1 of GRAPES MESO 3KM.

    Example output:

    Timing for processing for step 129 (2020050900:59:30):          0.77260 elapsed seconds.
    Timing for processing for step 129 (2020050900:59:30):          0.77182 cpu seconds.
     output modelvar use    2.41637611389160      seconds
      post grib2 compress and output use   0.456164121627808       seconds.
     ADJUST TIME STEP: old dt=   23.00000      new dt=   24.00000      MaxCfl=
       1.155885
     begin of gcr  8.673518207560389E-006
     RES of gcr  9.119498186871097E-013 in           24 iterations

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
