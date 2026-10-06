"""Record which experts OLMoE's router picks for every token (steps M1, M2).

Output: int16 array [tokens, layers, top_k], saved as .npz.

Why one forward pass is enough: the model is causal, so the routing for token t
only depends on tokens 0..t. Feeding the whole chunk at once (teacher forcing)
gives exactly the same expert choices as generating token by token.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config


def load_moe(device="cuda", dtype=torch.float16):
    """Load OLMoE. fp16 because the T4 has no bf16 support."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MOE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.MOE_MODEL, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    return model, tok


@torch.no_grad()
def route(model, input_ids, top_k=config.TOP_K):
    """One forward pass. Returns (experts [batch, seq, layers, top_k], logits)."""
    out = model(input_ids, output_router_logits=True)
    # router_logits: one tensor per layer, shape [batch*seq, num_experts]
    per_layer = [r.topk(top_k, dim=-1).indices for r in out.router_logits]
    experts = torch.stack(per_layer, dim=1)                       # [batch*seq, layers, k]
    b, s = input_ids.shape
    return experts.view(b, s, len(per_layer), top_k), out.logits


def chunk_loss(logits, input_ids):
    """Sum of next-token cross-entropy and number of predicted tokens (for perplexity).
    One sequence at a time: the float32 copy of all logits would not fit next to OLMoE on a T4."""
    loss = 0.0
    for row_logits, row_ids in zip(logits, input_ids):
        loss += F.cross_entropy(row_logits[:-1].float(), row_ids[1:], reduction="sum").item()
    return loss, input_ids[:, 1:].numel()


def collect_traces(model, chunks, out_path, batch_size=config.BATCH_SIZE, save_every=10):
    """Run all chunks through the model and save the expert trace.

    Resume-safe: progress is saved to `out_path` every `save_every` batches,
    and a rerun continues from where it stopped (useful if Kaggle disconnects).
    """
    n, seq = chunks.shape
    L, K = config.NUM_LAYERS, config.TOP_K
    trace = np.zeros((n, seq, L, K), dtype=np.int16)
    done, total_loss, total_tok = 0, 0.0, 0
    if os.path.exists(out_path):
        old = np.load(out_path)
        done = int(old["chunks_done"])
        trace[:done] = old["trace"].reshape(-1, seq, L, K)[:done]
        total_loss, total_tok = float(old["loss_sum"]), int(old["loss_tokens"])
        print(f"resuming from chunk {done}/{n}")

    def save():
        np.savez_compressed(out_path, trace=trace[:done].reshape(-1, L, K), chunks_done=done,
                            loss_sum=total_loss, loss_tokens=total_tok)

    device = next(model.parameters()).device
    start = time.time()
    for i, b in enumerate(range(done, n, batch_size)):
        ids = chunks[b:b + batch_size].to(device)
        experts, logits = route(model, ids)
        trace[b:b + len(ids)] = experts.cpu().numpy()
        l, t = chunk_loss(logits, ids)
        total_loss, total_tok = total_loss + l, total_tok + t
        done = b + len(ids)
        if (i + 1) % save_every == 0:
            save()
            print(f"{done}/{n} chunks  {time.time() - start:.0f}s  ppl={np.exp(total_loss / total_tok):.2f}")
    save()
    print(f"done: {done * seq} tokens, perplexity {np.exp(total_loss / total_tok):.2f}")
    return trace.reshape(-1, L, K)
