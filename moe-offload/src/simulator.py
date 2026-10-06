"""Cache simulator: replays an expert trace against a GPU cache of K slots per layer.

Trace format: int array [tokens, layers, top_k] -- expert ids chosen by the router.

For each token t, for each layer L:
    the router needs top_k experts.
    expert already in layer L's cache -> HIT  (free)
    expert not in the cache           -> MISS (copy it from CPU; evict one if full)
Optionally a predictor prefetches experts *before* they are requested.

Each layer has its own cache (OLMoE's experts are different weights in every layer).
The same LayerCache class is used by the real offloaded model (src/offload.py),
so the simulator and the real GPU cache follow exactly the same rules.
"""
import numpy as np

from src.policies import make_policy


class LayerCache:
    """The expert cache of ONE layer: which experts are on the GPU right now."""

    def __init__(self, slots, policy):
        self.slots = slots            # how many experts fit on the GPU for this layer
        self.policy = policy          # decides what to evict (src/policies.py)
        self.cache = set()            # experts currently on the GPU
        self.prefetched = set()       # prefetched experts that were not used yet
        self.hits = self.misses = 0
        self.prefetches = self.useful_prefetches = 0

    def request(self, needed, t):
        """Token t needs these experts. Returns the list of (expert, evicted_expert)
        loads that must happen; evicted_expert is None if a slot was free."""
        needed = [int(e) for e in needed]
        missing = []

        # 1) count hits first, so we never evict an expert this token still needs
        for e in needed:
            if e in self.cache:
                self.hits += 1
                if e in self.prefetched:            # a prefetch paid off
                    self.useful_prefetches += 1
                    self.prefetched.discard(e)
            else:
                self.misses += 1
                missing.append(e)
            self.policy.on_use(e, t)

        # 2) bring in the missing experts, evicting others if the cache is full
        loads = []
        for e in missing:
            victim = self._make_room(t, keep=set(needed))
            self.cache.add(e)
            loads.append((e, victim))
        return loads

    def prefetch(self, experts, t, keep=()):
        """Copy predicted experts in early. Never evicts experts in `keep`.
        Returns the list of (expert, evicted_expert) loads."""
        loads, keep = [], set(keep)
        for e in experts:
            e = int(e)
            if e in self.cache:
                continue
            if len(self.cache) >= self.slots and not self.cache - keep:
                break                               # no room for a prefetch: skip it
            victim = self._make_room(t, keep)
            self.cache.add(e)
            self.policy.on_use(e, t)                # treat a prefetched expert as "just used"
            self.prefetched.add(e)
            self.prefetches += 1
            keep.add(e)                             # don't evict what we just prefetched
            loads.append((e, victim))
        return loads

    def _make_room(self, t, keep):
        """If the cache is full, evict one expert (not in `keep`) and return it."""
        if len(self.cache) < self.slots:
            return None
        victim = self.policy.choose_victim(self.cache - keep, t)
        self.cache.discard(victim)
        self.prefetched.discard(victim)
        self.policy.on_evict(victim)
        return victim


def simulate(trace, policy="LRU", slots=16, predictor=None, prefetch_p=4,
             copy_time_s=1.1e-3, seed=0, num_experts=64):
    """Run one simulation and return a dict of results.

    slots      : int (same for every layer) or list with one value per layer
    predictor  : fitted object from src/predictor.py, or None
    copy_time_s: measured seconds to copy one expert CPU -> GPU (step M3)
    """
    T, L, K = trace.shape
    slots = [slots] * L if np.isscalar(slots) else list(slots)
    assert len(slots) == L
    assert min(slots) >= K, "each layer needs room for at least the top_k experts of one token"

    # "NoCache" = flush after every token: every single request is a copy
    if policy == "NoCache":
        return _summary("NoCache", T, K, np.zeros(L, int), np.full(L, T * K), 0, 0, [], copy_time_s)

    layers = [LayerCache(slots[l], make_policy(policy, trace[:, l], seed + l, num_experts))
              for l in range(L)]
    precisions = []

    for t in range(T):
        for l in range(L):
            layers[l].request(trace[t, l], t)

            # optional: prefetch what the predictor thinks comes next
            if predictor is None:
                continue
            guess = predictor.predict(t, l, trace[t, l], prefetch_p)
            if guess is None:
                continue
            target_layer, target_token, experts = guess
            if target_token < T:      # how many guesses were right? (for reporting only)
                actual = set(trace[target_token, target_layer].tolist())
                precisions.append(len(actual & set(experts.tolist())) / len(experts))
            # same layer: the current token's experts must stay
            keep = trace[t, l].tolist() if target_layer == l else ()
            layers[target_layer].prefetch(experts, t, keep)

    hits = np.array([c.hits for c in layers])
    misses = np.array([c.misses for c in layers])
    return _summary(policy, T, K, hits, misses, sum(c.prefetches for c in layers),
                    sum(c.useful_prefetches for c in layers), precisions, copy_time_s)


def _summary(policy, T, K, hits, misses, n_prefetch, n_useful, precisions, copy_time_s):
    total = int(hits.sum() + misses.sum())
    m = int(misses.sum())
    return {
        "policy": policy,
        "requests": total,
        "hits": int(hits.sum()),
        "misses": m,                                   # copies the GPU must WAIT for
        "hit_rate": float(hits.sum()) / total,
        "prefetches": n_prefetch,                      # copies done ahead of time
        "useful_prefetches": n_useful,
        "transfers": m + n_prefetch,                   # all copies (bandwidth / energy)
        "stall_time_s": m * copy_time_s,               # estimated time lost waiting
        "stall_ms_per_token": 1000 * m * copy_time_s / T,
        "prediction_precision": float(np.mean(precisions)) if precisions else None,
        "hit_rate_per_layer": (hits / (hits + misses)).tolist(),
        "misses_per_layer": misses.tolist(),
    }


def split_trace(trace, fit_fraction=0.5):
    """First part (fit) for learning counts/budgets, rest (test) for reporting."""
    cut = int(len(trace) * fit_fraction)
    return trace[:cut], trace[cut:]
