"""Compare OLMoE with dense models on the same text (step M3).

For each model we measure:
  - total and active parameters, FLOPs per token (~ 2 x active params)
  - perplexity on both datasets (quality)
  - prefill speed (tokens/s) and energy (J/token) while computing perplexity
  - decode speed (tokens/s) and energy, one token at a time, batch 1
  - peak GPU memory

Models are loaded one at a time and freed afterwards (only 16 GB on a T4).
"""
import gc
import math

import torch

import config
from src.data import make_chunks
from src.energy import EnergyMeter
from src.traces import chunk_loss


def load_model(model_id, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        precision = "fp16"
    except torch.cuda.OutOfMemoryError:
        # Dense 7B is ~13.8 GB in fp16: if it does not fit, use 8-bit (needs bitsandbytes)
        gc.collect(); torch.cuda.empty_cache()
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=device)
        precision = "int8"
    return model.eval(), tok, precision


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    expert = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    if expert == 0:                         # dense model: everything is active
        return total, total
    # MoE: only TOP_K of NUM_EXPERTS experts run per token
    active = total - expert + expert * config.TOP_K // config.NUM_EXPERTS
    return total, active


@torch.no_grad()
def perplexity_and_speed(model, chunks, batch_size=1):
    device = next(model.parameters()).device
    loss_sum, n_tok = 0.0, 0
    with EnergyMeter() as m:
        for b in range(0, len(chunks), batch_size):
            ids = chunks[b:b + batch_size].to(device)
            loss, n = chunk_loss(model(ids).logits, ids)    # same loss code as the traces step
            loss_sum, n_tok = loss_sum + loss, n_tok + n
        torch.cuda.synchronize()
    n_in = chunks.numel()
    return {"perplexity": math.exp(loss_sum / n_tok),
            "prefill_tok_per_s": n_in / m.seconds,
            "prefill_J_per_tok": m.joules / n_in if m.available else None}


@torch.no_grad()
def decode_over_text(model, ids):
    """Decode like a chatbot does: ONE token per step, reusing the KV cache.
    We feed real text (not the model's own guesses) so every model and every
    cache setting sees exactly the same tokens."""
    device = next(model.parameters()).device
    ids = ids.view(1, -1).to(device)
    past = None
    for i in range(ids.shape[1]):
        out = model(ids[:, i:i + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values


@torch.no_grad()
def decode_speed(model, ids, n_tokens=128):
    """Tokens per second and energy per token for batch-1 decoding."""
    decode_over_text(model, ids[:8])                  # warm-up (CUDA start-up costs)
    with EnergyMeter() as m:
        decode_over_text(model, ids[:n_tokens])
        torch.cuda.synchronize()
    return {"decode_tok_per_s": n_tokens / m.seconds,
            "decode_J_per_tok": m.joules / n_tokens if m.available else None}


def run_baseline(model_id, n_chunks=50):
    torch.cuda.reset_peak_memory_stats()
    model, tok, precision = load_model(model_id)
    total, active = count_params(model)
    row = {"model": model_id, "precision": precision,
           "total_params_B": total / 1e9, "active_params_B": active / 1e9,
           "GFLOPs_per_token": 2 * active / 1e9,
           "weights_GB": torch.cuda.memory_allocated() / 1e9}
    for name in ("wikitext", "code"):
        chunks = make_chunks(tok, name)[:n_chunks]   # same text for every model (same tokenizer)
        for k, v in perplexity_and_speed(model, chunks).items():
            row[f"{name}_{k}"] = v
        if name == "wikitext":
            row.update(decode_speed(model, chunks[0]))
    row["peak_GPU_GB"] = torch.cuda.max_memory_allocated() / 1e9
    del model
    gc.collect(); torch.cuda.empty_cache()        # free the GPU for the next model
    return row
