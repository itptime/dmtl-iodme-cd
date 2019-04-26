import os

import numpy as np

from io_json import json_read

_run = 0

names = [
    "la_roche_sur_yon",
    "longwy",
    "toulouse",
    "beziers",
    "arras",
    "chalon_sur_saone",
    "lyon",
    "montpellier",
    "nantes",
    "valence",
]

for name in names:
    results_multitask = [None] * 4
    if os.path.isfile(f"results/ovo_{name}_multitask_{_run}.json"):
        results = json_read(f"results/ovo_{name}_multitask_{_run}.json")
        for target in range(4):
            results_multitask[target] = [run[-4 + target] for run in results]
    else:
        print("No file of one versus others results, execute ovo_multitask.py first")

    results_singletask = [None] * 4
    for target in range(4):
        if os.path.isfile(f"results/ovo_{name}_singletask_{target}.json"):
            results_singletask[target] = [
                run[-1]
                for run in json_read(f"results/ovo_{name}_singletask_{target}.json")
            ]
        elif target == 0:
            print(
                "No file of one versus others results, execute ovo_singletask.py first"
            )

    for target in range(4):
        print(
            f"{name} multitask {target}: {np.mean(results_multitask[target])}"
            + f" +/- {np.std(results_multitask[target])}"
        )
        print(
            f"{name} singletask {target}: {np.mean(results_singletask[target])}"
            + f" +/- {np.std(results_singletask[target])}"
        )
