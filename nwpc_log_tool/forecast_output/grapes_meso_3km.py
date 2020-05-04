import re

import pandas as pd


def get_step_time_from_file(file_path):
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


def get_output_time_from_file(file_path):
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
