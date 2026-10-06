"""Count-based predictors that guess which experts will be needed next,
so they can be copied to the GPU *before* the router asks for them.

No neural network: we just count co-occurrences in the "fit" part of a
trace and look the counts up on the unseen "test" part.

Two variants:
  CrossLayerPredictor : experts of layer L  (token t) -> experts of layer L+1 (token t)
  NextTokenPredictor  : experts of layer L  (token t) -> experts of layer L   (token t+1)
"""
import numpy as np


class CrossLayerPredictor:
    name = "CrossLayer"

    def __init__(self, num_layers=16, num_experts=64):
        # counts[L, a, b] = how often expert a in layer L came with expert b in layer L+1
        self.counts = np.zeros((num_layers - 1, num_experts, num_experts), dtype=np.int64)
        self.num_layers = num_layers

    def fit(self, trace):
        """trace: int array [tokens, layers, top_k]"""
        for L in range(self.num_layers - 1):
            for now, nxt in zip(trace[:, L], trace[:, L + 1]):
                self.counts[L][np.ix_(now, nxt)] += 1
        return self

    def predict(self, t, layer, experts, p):
        """Return (target_layer, target_token, top-p expert ids), or None."""
        if layer == self.num_layers - 1:
            return None
        scores = self.counts[layer][experts].sum(axis=0)
        return layer + 1, t, np.argsort(-scores)[:p]


class NextTokenPredictor:
    name = "NextToken"

    def __init__(self, num_layers=16, num_experts=64):
        self.counts = np.zeros((num_layers, num_experts, num_experts), dtype=np.int64)
        self.num_layers = num_layers

    def fit(self, trace):
        for L in range(self.num_layers):
            for now, nxt in zip(trace[:-1, L], trace[1:, L]):
                self.counts[L][np.ix_(now, nxt)] += 1
        return self

    def predict(self, t, layer, experts, p):
        scores = self.counts[layer][experts].sum(axis=0)
        return layer, t + 1, np.argsort(-scores)[:p]


def make_predictor(name, num_layers=16, num_experts=64):
    if name == "CrossLayer":
        return CrossLayerPredictor(num_layers, num_experts)
    if name == "NextToken":
        return NextTokenPredictor(num_layers, num_experts)
    raise ValueError(f"unknown predictor {name}")
