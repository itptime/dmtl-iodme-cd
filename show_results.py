import os

import numpy as np

from io_json import json_read


_run = 2

results_multitask = [None] * 4
if os.path.isfile(f"results/multitask_{_run}.json"):
    results = json_read(f"results/multitask_{_run}.json")
    for target in range(4):
        results_multitask[target] = [run[-4 + target] for run in results]
else:
    print("No file of results, execute multitask.py first")

results_singletask = [None] * 4
for target in range(4):
    if os.path.isfile(f"results/singletask_{target}.json"):
        results_singletask[target] = [
            run[-1] for run in json_read(f"results/singletask_{target}.json")
        ]
    elif target == 0:
        print("No file of results, execute singletask.py first")

for target in range(4):
    print(
        f"multitask {target}: {np.mean(results_multitask[target])}"
        + f" +/- {np.std(results_multitask[target])}"
    )
    print(
        f"singletask {target}: {np.mean(results_singletask[target])}"
        + f" +/- {np.std(results_singletask[target])}"
    )
