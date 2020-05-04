import re

import pandas as pd


def get_step_time_from_file(file_path):
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
    return df
