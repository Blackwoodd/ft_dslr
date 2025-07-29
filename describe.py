#!/usr/bin/env python3

import sys
import csv
import math

def percentile(values, p):
    values = sorted(values)
    n = len(values)
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    else:
        return values[f] * (c - k) + values[c] * (k - f)

def ecart_type(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)

def describe_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    numeric_cols = []
    for i, val in enumerate(header):
        try:
            float(data[0][i])  # test avec la 1ère valeur
            numeric_cols.append(i)
        except ValueError:
            continue

    list_value = {}
    for i in numeric_cols:
        list_value[i] = []
    for row in data:
        for i in numeric_cols:
            if row[i] != '':
                list_value[i].append(float(row[i]))

    results = {}
    for i in numeric_cols:
        values = list_value[i]
        mean = sum(values) / len(values)
        max_val = max(values)
        min_value = min(values)
        std = ecart_type(values)
        median = percentile(values, 0.5)
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)

        results[header[i]] = {
            "Count": len(values),
            "Mean": mean,
            "Std": std,
            "Min": min_value,
            "25%": q1,
            "50%": median,
            "75%": q3,
            "Max": max_val
        }
    print("Stat".ljust(10), end="")
    for col in results.keys():
        print(col.rjust(12), end="")
    print()

    for stat_name in ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
        print(stat_name.ljust(10), end="")
        for col in results.keys():
            value = results[col][stat_name]
            if isinstance(value, int):
                print(f"{value:12d}", end="")
            else:
                print(f"{value:12.2f}", end="")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Please provide exactly one argument.")
    else:
         describe_csv(sys.argv[1])