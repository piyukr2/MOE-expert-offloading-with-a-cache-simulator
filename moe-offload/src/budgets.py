"""Per-layer cache budgets (step M7).

Uniform split: every layer gets the same number of slots.
Reallocated  : same TOTAL slots, but layers that miss more get more slots.
The split is learned on the fit half of the trace and tested on the test half.
"""
import numpy as np

from src.simulator import simulate


def miss_proportional_slots(miss_rates, total_slots, min_slots=8, max_slots=64):
    """Split total_slots across layers in proportion to each layer's miss rate.

    min_slots = 8 because one token needs top_k = 8 experts at once.
    """
    miss_rates = np.asarray(miss_rates, dtype=float)
    L = len(miss_rates)
    assert min_slots * L <= total_slots <= max_slots * L

    # Give every layer the minimum, then hand out the rest by miss rate
    slots = np.full(L, min_slots)
    extra = total_slots - slots.sum()
    share = miss_rates / miss_rates.sum() if miss_rates.sum() > 0 else np.full(L, 1 / L)
    slots = np.minimum(slots + np.floor(share * extra).astype(int), max_slots)

    # Rounding leaves a few slots over: give them one by one to the
    # layers with the highest miss rate that still have room
    for l in np.argsort(-miss_rates):
        if slots.sum() == total_slots:
            break
        if slots[l] < max_slots:
            slots[l] += 1
    while slots.sum() < total_slots:          # very rare: everyone else was full
        slots[np.argmin(slots)] += 1
    return slots.tolist()


def compare_uniform_vs_reallocated(fit, test, slots_per_layer, policy="LRU", copy_time_s=1.1e-3):
    L = fit.shape[1]
    total = slots_per_layer * L

    # 1) measure each layer's miss rate on the FIT half with a uniform split
    fit_run = simulate(fit, policy, slots_per_layer, copy_time_s=copy_time_s)
    miss_rates = [1 - h for h in fit_run["hit_rate_per_layer"]]

    # 2) reallocate the same total
    new_slots = miss_proportional_slots(miss_rates, total)

    # 3) evaluate both on the unseen TEST half
    uniform = simulate(test, policy, slots_per_layer, copy_time_s=copy_time_s)
    realloc = simulate(test, policy, new_slots, copy_time_s=copy_time_s)
    return {
        "slots_per_layer": slots_per_layer,
        "reallocated_slots": new_slots,
        "fit_miss_rate_per_layer": miss_rates,
        "uniform": uniform,
        "reallocated": realloc,
    }
