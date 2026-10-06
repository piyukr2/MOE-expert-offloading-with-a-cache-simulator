"""Run every policy x cache size on one trace and collect a results table (M5, M6, M9)."""
import pandas as pd

import config
from src.calibrate import load_copy_time
from src.predictor import make_predictor
from src.simulator import simulate, split_trace


def run_grid(trace, dataset, sizes=config.CACHE_SIZES, random_seeds=5, max_test_tokens=None):
    """Returns one row per (cache size, policy). Results are on the TEST half only."""
    copy_time = load_copy_time()
    fit, test = split_trace(trace, config.FIT_FRACTION)
    if max_test_tokens:
        test = test[:max_test_tokens]          # quick runs while developing

    # the predictors learn their counts on the FIT half only
    predictors = {name: make_predictor(name).fit(fit) for name in ("CrossLayer", "NextToken")}

    rows = []

    def add(result, size, variant, seed=0):
        # keep the single-number columns (skip the per-layer lists)
        r = {k: v for k, v in result.items() if not k.endswith("per_layer")}
        rows.append({"dataset": dataset, "slots": size, "variant": variant, "seed": seed, **r})

    for size in sizes:
        add(simulate(test, "NoCache", size, copy_time_s=copy_time), size, "NoCache")
        for seed in range(random_seeds):                       # Random: repeat and average
            add(simulate(test, "Random", size, seed=seed, copy_time_s=copy_time), size, "Random", seed)
        for pol in ("LRU", "LFU", "Belady"):
            add(simulate(test, pol, size, copy_time_s=copy_time), size, pol)
        for name, pred in predictors.items():                  # LRU + prefetching
            add(simulate(test, "LRU", size, predictor=pred, prefetch_p=config.PREFETCH_P,
                         copy_time_s=copy_time), size, f"LRU+{name}")
        print(f"{dataset}: slots={size} done")

    df = pd.DataFrame(rows)
    # average the Random seeds into one row
    return df.groupby(["dataset", "slots", "variant"], as_index=False, sort=False).mean(numeric_only=True)
