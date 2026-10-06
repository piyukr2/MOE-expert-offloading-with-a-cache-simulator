"""Eviction policies for one layer's expert cache.

Every policy answers one question: "the cache is full and a new expert must
come in -- which expert do we throw out?"

The simulator calls three methods:
    on_use(expert, t)            an expert was requested at token t (hit or miss)
    choose_victim(candidates, t) pick one expert to evict from `candidates`
    on_evict(expert)             the expert left the cache

`candidates` never contains experts that the current token still needs.
"""
import random

import numpy as np


class Policy:
    name = "base"

    def on_use(self, expert, t):
        pass

    def on_evict(self, expert):
        pass

    def choose_victim(self, candidates, t):
        raise NotImplementedError


class RandomPolicy(Policy):
    """Evict a random expert. The 'no intelligence' baseline."""
    name = "Random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def choose_victim(self, candidates, t):
        return self.rng.choice(sorted(candidates))


class LRUPolicy(Policy):
    """Evict the Least Recently Used expert."""
    name = "LRU"

    def __init__(self):
        self.last_used = {}          # expert -> last token index it was used

    def on_use(self, expert, t):
        self.last_used[expert] = t

    def choose_victim(self, candidates, t):
        # ties (same last use) -> smallest expert id, so results are repeatable
        return min(candidates, key=lambda e: (self.last_used.get(e, -1), e))


class LFUPolicy(Policy):
    """Evict the Least Frequently Used expert (counts all past uses).
    Ties are broken by LRU."""
    name = "LFU"

    def __init__(self):
        self.count = {}
        self.last_used = {}

    def on_use(self, expert, t):
        self.count[expert] = self.count.get(expert, 0) + 1
        self.last_used[expert] = t

    def choose_victim(self, candidates, t):
        return min(candidates, key=lambda e: (self.count.get(e, 0), self.last_used.get(e, -1), e))


class BeladyPolicy(Policy):
    """Oracle: evict the expert whose next use is farthest in the future.
    It cheats by looking at the future trace, so it is the best any policy
    can do (upper bound on hit rate)."""
    name = "Belady"

    def __init__(self, layer_trace, num_experts=64):
        # layer_trace: int array [tokens, top_k] for THIS layer
        # uses[e] = sorted list of token indices where expert e is used
        self.uses = [np.nonzero((layer_trace == e).any(axis=1))[0] for e in range(num_experts)]

    def next_use(self, e, t):
        """First token after t that uses expert e (infinity if never)."""
        i = np.searchsorted(self.uses[e], t, side="right")
        return self.uses[e][i] if i < len(self.uses[e]) else float("inf")

    def choose_victim(self, candidates, t):
        return max(sorted(candidates), key=lambda e: self.next_use(e, t))


POLICY_NAMES = ["Random", "LRU", "LFU", "Belady"]


def make_policy(name, layer_trace=None, seed=0, num_experts=64):
    if name == "Random":
        return RandomPolicy(seed)
    if name == "LRU":
        return LRUPolicy()
    if name == "LFU":
        return LFUPolicy()
    if name == "Belady":
        return BeladyPolicy(layer_trace, num_experts)
    raise ValueError(f"unknown policy {name}")
