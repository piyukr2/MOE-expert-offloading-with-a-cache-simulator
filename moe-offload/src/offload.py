"""Real expert offloading for OLMoE (step M8).

Normal OLMoE keeps all 64 experts of every layer on the GPU (~13 GB).
Here we keep ALL experts in CPU memory and give each layer only K "slots" on
the GPU. Before an expert runs, we make sure it is in a slot:
    already in a slot -> hit, use it
    not in a slot     -> miss, copy it from CPU into a slot (evicting another)

Which expert to evict is decided by the same LayerCache + policy code as the
simulator, so the real miss count can be checked against the simulated one.

Written for transformers 5.x, where each layer's experts are one module
(`layer.mlp.experts`) holding two big tensors:
    gate_up_proj : [64, 2 * 1024, 2048]
    down_proj    : [64, 2048, 1024]
"""
import numpy as np
import torch
import torch.nn as nn

import config
from src.policies import make_policy
from src.simulator import LayerCache


class OffloadedExperts(nn.Module):
    """Drop-in replacement for OlmoeExperts that keeps only `slots` experts on the GPU."""

    def __init__(self, experts, slots, policy="LRU", device="cuda"):
        super().__init__()
        self.act_fn = experts.act_fn
        self.device = device
        pin = device == "cuda"      # pinned CPU memory makes CPU -> GPU copies faster

        # All experts stay in CPU memory. Plain attributes (not nn.Parameters),
        # so model.to("cuda") does NOT move them to the GPU.
        self.cpu_gate_up = experts.gate_up_proj.data.contiguous()
        self.cpu_down = experts.down_proj.data.contiguous()
        if pin:
            try:
                self.cpu_gate_up = self.cpu_gate_up.pin_memory()
                self.cpu_down = self.cpu_down.pin_memory()
            except RuntimeError as err:     # not enough pinnable memory: slower but works
                print("could not pin memory, using normal memory:", err)

        # The GPU slot pool: room for `slots` experts
        self.gpu_gate_up = torch.empty((slots,) + self.cpu_gate_up.shape[1:],
                                       dtype=self.cpu_gate_up.dtype, device=device)
        self.gpu_down = torch.empty((slots,) + self.cpu_down.shape[1:],
                                    dtype=self.cpu_down.dtype, device=device)

        self.cache = LayerCache(slots, make_policy(policy))   # same rules as the simulator
        self.slot_of = {}                     # expert id -> slot number
        self.free_slots = list(range(slots))
        self.t = 0                            # token counter (the simulator's "time")
        self.log = []                         # expert ids requested per token (for checking)

    def load(self, expert, evicted):
        """Copy one expert from CPU into a GPU slot (reusing the evicted expert's slot)."""
        slot = self.slot_of.pop(evicted) if evicted is not None else self.free_slots.pop()
        self.gpu_gate_up[slot].copy_(self.cpu_gate_up[expert], non_blocking=True)
        self.gpu_down[slot].copy_(self.cpu_down[expert], non_blocking=True)
        self.slot_of[expert] = slot

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # hidden_states: [tokens, 2048]; top_k_index / top_k_weights: [tokens, 8]
        out = torch.zeros_like(hidden_states)
        chosen = top_k_index.tolist()

        # One token at a time, exactly like the simulator (simple, and in
        # batch-1 decoding there is only one token per call anyway)
        for i, experts in enumerate(chosen):
            for expert, evicted in self.cache.request(experts, self.t):
                self.load(expert, evicted)
            self.log.append(experts)
            self.t += 1

            x = hidden_states[i]
            for k, e in enumerate(experts):
                s = self.slot_of[e]
                gate, up = torch.nn.functional.linear(x, self.gpu_gate_up[s]).chunk(2, dim=-1)
                y = torch.nn.functional.linear(self.act_fn(gate) * up, self.gpu_down[s])
                out[i] += top_k_weights[i, k] * y
        return out


def offload_model(model, slots, policy="LRU", device="cuda"):
    """Replace every layer's experts with OffloadedExperts, then move the rest
    of the model (attention, router, embeddings: ~1.3 GB) to the GPU.

    `model` must be loaded on the CPU first, e.g.
        AutoModelForCausalLM.from_pretrained(config.MOE_MODEL, dtype=torch.float16)
    """
    slots = [slots] * len(model.model.layers) if np.isscalar(slots) else slots
    for layer, k in zip(model.model.layers, slots):
        layer.mlp.experts = OffloadedExperts(layer.mlp.experts, k, policy, device)
    return model.to(device).eval()


def offload_stats(model):
    """Total hits/misses over all layers, plus the logged trace [tokens, layers, 8]."""
    experts = [layer.mlp.experts for layer in model.model.layers]
    hits = sum(e.cache.hits for e in experts)
    misses = sum(e.cache.misses for e in experts)
    trace = np.array([e.log for e in experts]).transpose(1, 0, 2)
    return {"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses), "trace": trace}


def reset_stats(model):
    """Clear counters (e.g. after a warm-up run). The cache contents are kept."""
    for layer in model.model.layers:
        e = layer.mlp.experts
        e.cache.hits = e.cache.misses = 0
        e.log = []
