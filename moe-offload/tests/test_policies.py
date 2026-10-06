"""Small hand-made traces whose answers we can work out on paper.

A trace is [tokens, layers, top_k]. Most tests use 1 layer and top_k = 1,
so a trace is just a list of expert ids: [1, 2, 1] -> shape (3, 1, 1).
"""
import numpy as np

from src.budgets import miss_proportional_slots
from src.predictor import CrossLayerPredictor, NextTokenPredictor
from src.simulator import simulate, split_trace


def seq(*experts):
    return np.array(experts).reshape(-1, 1, 1)


def hits(trace, policy, slots, **kw):
    return simulate(trace, policy, slots, num_experts=8, **kw)["hits"]


# ---------- hand-computed answers ----------

def test_lru():
    # 1m 2m 1h | 3m evicts 2 (least recent) | 1h | 2m  -> 2 hits
    assert hits(seq(1, 2, 1, 3, 1, 2), "LRU", 2) == 2


def test_lfu_differs_from_lru():
    # after 1,1,2 a new expert 3 arrives:
    #   LFU evicts 2 (used once), LRU evicts 1 (used longest ago)
    t = seq(1, 1, 2, 3, 2)
    assert hits(t, "LFU", 2) == 1
    assert hits(t, "LRU", 2) == 2


def test_belady():
    # cyclic pattern: LRU always evicts the one needed next (0 hits),
    # Belady keeps it and gets 2 hits
    t = seq(1, 2, 3, 1, 2, 3)
    assert hits(t, "LRU", 2) == 0
    assert hits(t, "Belady", 2) == 2


def test_no_cache_never_hits():
    r = simulate(seq(1, 1, 1), "NoCache", 2, num_experts=8)
    assert r["hits"] == 0 and r["misses"] == 3


def test_current_token_experts_are_never_evicted():
    # top_k = 2, 2 slots. Token 1 needs [2, 0]: 0 is a hit, 2 must evict 1 (not 0).
    t = np.array([[0, 1], [2, 0], [0, 2]]).reshape(3, 1, 2)
    assert hits(t, "LRU", 2) == 3


# ---------- invariants on random traces ----------

def random_trace(T=300, L=2, K=8, E=64, seed=0):
    rng = np.random.default_rng(seed)
    # skewed popularity, like real routers (some experts are "hot")
    p = rng.dirichlet(np.full(E, 0.3))
    return np.stack([[rng.choice(E, K, replace=False, p=p) for _ in range(L)] for _ in range(T)])


def test_counts_add_up():
    tr = random_trace()
    for pol in ["Random", "LRU", "LFU", "Belady"]:
        r = simulate(tr, pol, 16)
        assert r["hits"] + r["misses"] == r["requests"] == tr.size


def test_belady_is_best():
    for seed in range(3):
        tr = random_trace(seed=seed)
        best = simulate(tr, "Belady", 16)["hits"]
        for pol in ["Random", "LRU", "LFU"]:
            assert simulate(tr, pol, 16)["hits"] <= best


def test_bigger_cache_never_hurts_lru_or_belady():
    tr = random_trace()
    for pol in ["LRU", "Belady"]:
        h = [simulate(tr, pol, k)["hits"] for k in (8, 16, 32, 64)]
        assert h == sorted(h)
    assert simulate(tr, "LRU", 64)["misses"] <= 64 * tr.shape[1]   # only cold misses


def test_per_layer_slots():
    tr = random_trace(L=2)
    r = simulate(tr, "LRU", [8, 32])
    assert r["hit_rate_per_layer"][1] > r["hit_rate_per_layer"][0]


# ---------- predictors ----------

def test_cross_layer_predictor_learns_a_fixed_mapping():
    # 4 fixed groups of 8 experts; when layer 0 uses group g, layer 1 uses
    # the same group shifted by 32. Perfectly predictable from layer 0.
    rng = np.random.default_rng(0)
    groups = np.arange(32).reshape(4, 8)
    l0 = groups[rng.integers(0, 4, 400)]
    tr = np.stack([l0, l0 + 32], axis=1)
    fit, test = split_trace(tr)
    pred = CrossLayerPredictor(num_layers=2).fit(fit)
    plain = simulate(test, "LRU", 16)
    with_pf = simulate(test, "LRU", 16, predictor=pred, prefetch_p=8)
    assert with_pf["prediction_precision"] == 1.0
    assert with_pf["misses_per_layer"][1] == 0               # every layer-1 expert was prefetched
    assert with_pf["misses"] < plain["misses"]


def test_next_token_predictor_runs():
    tr = random_trace()
    fit, test = split_trace(tr)
    pred = NextTokenPredictor(num_layers=2).fit(fit)
    r = simulate(test, "LRU", 16, predictor=pred, prefetch_p=4)
    assert r["prefetches"] > 0 and 0 <= r["prediction_precision"] <= 1
    # 8 slots are all taken by the current token: prefetch must be skipped, not crash
    assert simulate(test, "LRU", 8, predictor=pred, prefetch_p=4)["prefetches"] == 0


# ---------- per-layer budgets ----------

def test_budget_keeps_total_and_limits():
    rates = [0.9, 0.1, 0.5, 0.0]
    for total in (32, 64, 100, 256):
        s = miss_proportional_slots(rates, total)
        assert sum(s) == total
        assert all(8 <= x <= 64 for x in s)
    s = miss_proportional_slots(rates, 100)
    assert s[0] > s[2] > s[1] >= s[3]       # more misses -> more slots
