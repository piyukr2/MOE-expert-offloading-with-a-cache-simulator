"""Check the real offloading code on a TINY random OLMoE that runs on a CPU.

The tiny model has the same structure as OLMoE-1B-7B but only 4 layers,
16 experts and top-4 routing, so the test takes a few seconds.
Skipped automatically if torch / transformers are not installed.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.offload import offload_model, offload_stats
from src.simulator import simulate
from src.traces import route

E, K, LAYERS = 16, 4, 4


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.OlmoeConfig(vocab_size=100, hidden_size=64, intermediate_size=32,
                                   num_hidden_layers=LAYERS, num_attention_heads=4,
                                   num_key_value_heads=4, num_experts=E, num_experts_per_tok=K)
    torch.manual_seed(0)
    return transformers.OlmoeForCausalLM(cfg).eval()


@torch.no_grad()
def test_route_returns_expert_ids(tiny_model):
    ids = torch.randint(0, 100, (2, 10))
    experts, logits = route(tiny_model, ids, top_k=K)
    assert experts.shape == (2, 10, LAYERS, K)
    assert logits.shape == (2, 10, 100)


@torch.no_grad()
def test_offloaded_model_gives_same_output(tiny_model):
    ids = torch.randint(0, 100, (1, 12))
    expected = tiny_model(ids).logits
    small = offload_model(copy.deepcopy(tiny_model), slots=6, device="cpu")
    assert torch.allclose(small(ids).logits, expected, atol=1e-4)


@torch.no_grad()
def test_real_misses_match_simulator(tiny_model):
    # generate token by token, like real decoding
    small = offload_model(copy.deepcopy(tiny_model), slots=6, device="cpu")
    small.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=30, do_sample=False)
    real = offload_stats(small)

    sim = simulate(real["trace"], "LRU", 6, num_experts=E)
    assert real["misses"] == sim["misses"]
    assert real["hits"] == sim["hits"]
    assert 0 < real["hit_rate"] < 1
