#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

def describe_csv(file_path):
    data_csv = pd.read_csv(file_path)
    data_num = data_csv.select_dtypes(include=['number'])

    stats = {
        "Count": data_num.count(),
        "Mean": data_num.mean(),
        "Std": data_num.std(),
        "Min": data_num.min(),
        "25%": data_num.quantile(0.25),
        "50%": data_num.median(),
        "75%": data_num.quantile(0.75),
        "Max": data_num.max(),
    }

    result = pd.DataFrame(stats)
    print(result.T)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Please provide exactly one argument.")
    else:
         describe_csv(sys.argv[1])