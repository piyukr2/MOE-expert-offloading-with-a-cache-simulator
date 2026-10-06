# Plan: MoE Expert Offloading with a Cache Simulator (40 h, Kaggle T4)

## Context
The project plan (`MoE_Expert_Offloading_Project_Plan.docx`) asks: when an MoE model doesn't fit on a small GPU, which expert-caching policy best reduces slow CPU→GPU copies? The new requirement is to also show **a clear gain over a vanilla (dense, non-MoE) LLM**, while staying within 40 hours on one Kaggle T4 (16 GB), keeping code in GitHub and pulling it into Kaggle, and keeping the code easy for students to follow, built one milestone at a time.

The repo is empty apart from the .docx. Everything below is new code.

## The "gain over vanilla" story (what we will prove)
Compare three models on the same text, all from AI2's OLMo family so that only the architecture differs:

| Model | HF id | Total params | Active params / token |
|---|---|---|---|
| Dense small | `allenai/OLMo-1B-0724-hf` | ~1.2B | ~1.2B |
| **MoE (ours)** | `allenai/OLMoE-1B-7B-0924` | ~6.9B | ~1.3B |
| Dense large ("vanilla") | `allenai/OLMo-7B-0724-hf` | ~6.9B | ~6.9B |

Claims we can measure on a T4:
1. **Compute:** OLMoE uses ~5× fewer FLOPs per token than dense 7B (computed from active params, then checked with measured tokens/s with all weights on the GPU).
2. **Quality:** OLMoE perplexity is far better than dense 1B and close to dense 7B on both datasets. Same compute as 1B, quality near 7B.
3. **GPU memory:** with expert offloading plus a cache of 25% of experts, OLMoE needs ~4–5 GB of GPU memory instead of ~14 GB for dense 7B. That leaves room for a cheaper GPU, larger batches, or longer context.
4. **Energy per token:** measured with `pynvml` power sampling during runs. Expected to be lower for OLMoE than for dense 7B.
5. **Caching makes offloading practical:** the best policy (vs Random/no-cache) cuts CPU→GPU traffic substantially, and we report how close it gets to the Oracle.

Honest limitation for the report: offloaded OLMoE will be **slower per token than dense models that fit fully on the GPU**, because of PCIe copies. The gain is memory, compute and energy, and the caching study measures how much of the speed penalty a good policy removes. This is stated in the report, not hidden.

Useful numbers (verify in M1/M3): 16 MoE layers × 64 experts, top-8. One expert ≈ 3 × 2048 × 1024 params ≈ 12.6 MB in fp16. All experts ≈ 12.9 GB, non-expert weights ≈ 1.3 GB. T4 PCIe (pinned memory) ≈ 10–12 GB/s, so about 1 ms per expert copy.

## Datasets (inference input only, no training)
- **Natural language:** `Salesforce/wikitext`, config `wikitext-103-raw-v1`, test + validation splits.
- **Code (Python):** `codeparrot/codeparrot-clean-valid` (ungated). Fallback: `code_search_net` python.
- ~200k tokens each, in chunks of 512 tokens, taken from a fixed random sample (seed 0) of documents.
- Split each trace in time: first 50% = "fit" (predictor counts and per-layer budgets), last 50% = "test" (all reported numbers).

## Key simplifications for students
- **No hooks needed for traces:** call the model with `output_router_logits=True`. It returns per-layer router logits, and `topk(8)` gives the chosen experts. A teacher-forced forward pass produces the same routing as token-by-token generation, because the model is causal. Explain this in the notebook.
- Each layer has its **own cache** of K slots, simulated independently. Total budget = K × 16.
- Plain Python + NumPy for the simulator. No classes deeper than one `Policy` base class.
- **One `LayerCache` class** decides hits, misses and evictions for both the simulator and the real offloaded model, so they follow the same rules and their miss counts can be compared exactly.
- **Tested logic lives in `src/*.py`. Running, analysing and plotting happen in notebooks** (one per milestone, all run on Kaggle), with markdown explanations and simple comments.
- Every milestone ends with a notebook that runs top to bottom and a short `docs/Mx.md` note covering what was done, the result, and a plot.

## Repo layout
```
moe-offload/
  README.md                 # how to run each milestone on Kaggle (+ locally)
  requirements.txt          # transformers 5.x, accelerate, datasets, numpy, pandas, matplotlib, nvidia-ml-py, pytest
  config.py                 # model ids, dataset ids, token budget, cache sizes, paths, seeds
  src/
    data.py                 # load + tokenize + chunk both datasets
    traces.py               # load OLMoE, record router top-8 → int16 array [tokens, layers, 8]; save .npz
    calibrate.py            # time CPU→GPU copy of an expert-sized tensor (pinned vs not)
    baselines.py            # perplexity, tokens/s, peak VRAM, energy for one model
    energy.py               # tiny pynvml power sampler (background thread)
    policies.py             # Random, LRU, LFU, Belady
    predictor.py            # CrossLayer and NextToken count-based predictors
    simulator.py            # LayerCache + simulate(trace, policy, slots, predictor) → hits, misses, cost
    budgets.py              # uniform vs miss-rate-proportional per-layer allocation
    experiments.py          # run every policy x cache size on a trace
    offload.py              # (M8) real offloaded OLMoE: K expert slots per layer on the GPU
    files.py                # find files produced by earlier notebooks (/kaggle/input/...)
  tests/                    # test_policies.py (hand-made traces), test_offload.py (tiny random OLMoE on CPU)
  notebooks/M0_… … M9_…ipynb  # one Kaggle notebook per milestone
  docs/                     # TEMPLATE.md → M1.md … M9.md
  results/                  # final CSV/PNG copied from Kaggle outputs (committed); traces are NOT committed
```

## Kaggle + GitHub workflow (set up in M0)
- Develop locally → `git push` to GitHub. Each notebook's first cell clones the repo into `/kaggle/working/moe-offload` (or runs `git pull` if it's already there) and runs `pip install -r requirements.txt`. For a private repo, store a GitHub token in **Kaggle Secrets** as `GITHUB_TOKEN`.
- GPU notebooks (M0–M3, M8): **GPU T4 ×1**, internet ON. CPU notebooks (M4–M7, M9): accelerator **None**, which saves GPU quota.
- Everything a notebook creates goes to `/kaggle/working/` (`traces/`, `results/`). Traces are saved once as a **Kaggle Dataset** (`moe-traces`). Later notebooks add the traces dataset or earlier notebooks' outputs as **inputs**, and `src/files.find_file()` finds them under `/kaggle/input`.
- Final CSVs and plots are downloaded and committed to `results/` for the report.
- Budget: Kaggle allows ~30 GPU-hours/week and 12 h per session. GPU work is ~4–5 GPU-hours total (M1–M3, M8).

## Milestones (iterative, ~40 h total)

**M0. Setup (2 h)**: GitHub repo, layout, `config.py`, a Kaggle notebook that clones the repo and prints `torch.cuda.get_device_name()`.
✅ Done when: the Kaggle notebook pulls the latest commit and sees the T4.

**M1. Load OLMoE and see routing (3 h)**: load OLMoE in **fp16** (T4 has no bf16), run 2 sentences, print the top-8 experts per layer for each token, and compute perplexity on 20 chunks.
✅ Done when: we get a `[tokens, 16, 8]` array of expert IDs and a sensible perplexity.

**M2. Collect traces (4 h, ~1–2 GPU-h)**: `src/data.py` + `src/traces.py`. Batch the chunks, save after every N batches (resume-safe), and produce `traces_wikitext.npz` and `traces_code.npz`.
✅ Done when: both files exist on Kaggle and are published as a Kaggle Dataset.

**M3. Calibrate + dense baselines (4 h, ~2 GPU-h)**:
- `calibrate.py`: median time to copy a 12.6 MB fp16 tensor CPU→GPU, pinned vs pageable (100 repeats after warm-up).
- `baselines.py`: for OLMo-1B, OLMoE, and OLMo-7B, measure perplexity on both datasets (same 50 chunks), prefill tokens/s, decode tokens/s (128 new tokens, batch 1), peak VRAM (`torch.cuda.max_memory_allocated`), and joules/token (`energy.py`). Load one model at a time and free it before the next. If OLMo-7B fp16 OOMs, fall back to 8-bit via bitsandbytes and note it.
✅ Done when: `results/baselines.csv` and a bar chart of the gains (claims 1–4) exist.

**M4. Trace analysis (3 h, CPU)**: expert usage histogram per layer (hot/cold), reuse rate between consecutive tokens, entropy per layer, and wikitext vs code comparison.
✅ Done when: plots exist and answer "can caching help at all?".

**M5. Simulator + Random/LRU/LFU (5 h, CPU)**: `simulator.py` loops over tokens → layers → 8 requested experts. A hit happens if the expert is in that layer's slot set. Otherwise it's a miss: evict via policy and insert. `cost_seconds = misses × measured_copy_time`. Write unit tests on tiny traces with hand-computed answers. Cache sizes K ∈ {8, 16, 32} of 64 (12.5/25/50%). Also add the **"no cache" baseline** (every request is a miss, i.e. flush every token).
✅ Done when: tests pass and a hit-rate table is produced for 3 policies × 3 sizes × 2 datasets.

**M6. Belady Oracle + Predictor (5 h, CPU)**:
- Belady: precompute the next-use index for each request, then evict the expert with the farthest next use.
- Predictor: **cross-layer Markov** counts learned on the fit half, predicting layer L+1's experts from layer L's experts for the same token. Prefetch the top-P predicted experts (P = 4) into LRU before layer L+1 runs. Prefetched experts count as transfers but are off the critical path, so we report misses and "stall misses" separately. Also add a same-layer variant (token t → t+1) for comparison.
✅ Done when: all 5 policies are in one table, plus prediction precision/recall.

**M7. Per-layer budgets (3 h, CPU)**: on the fit half, run uniform LRU and record miss rate per layer. Reallocate the same total slots in proportion to miss rate (minimum 8, because one token needs 8 experts at once; maximum 64). Evaluate on the test half.
✅ Done when: uniform vs reallocated hit rate and cost are compared (reporting "no gain" is an acceptable result).

**M8. Real offloaded inference (5 h, ~2 GPU-h)**: `offload.py` keeps all expert weights in pinned CPU memory and gives each layer a GPU slot pool of K experts with LRU (and the best policy from M6, if time allows). It replaces the experts' forward with "ensure resident → compute". Run decode with batch 1 and measure real tokens/s, peak VRAM, and energy at K = 16 and 32.
✅ Done when: (a) the real miss count matches the simulator's for the same text, which validates the simulator, and (b) the offloaded OLMoE's VRAM, speed, and energy are added to the baselines table.
*Stretch / cut first if behind schedule:* if M8 slips, the report uses formula-estimated latency, as in the original plan.

**M9. Full experiments + write-up (6 h)**: run all policies × sizes × datasets (Random averaged over 5 seeds). Final plots: hit rate vs cache size, transfer cost per policy, per-layer results, and the gains-vs-dense chart. README, report, slides.

| M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3 | 4 | 4 | 3 | 5 | 5 | 3 | 5 | 6 | **40 h** |

## Risks & fallbacks
- OLMoE in fp16 needs ~14 GB of CPU RAM to load. Kaggle provides ~29 GB, so it's fine. Use `low_cpu_mem_usage=True`.
- If OLMo-7B fp16 doesn't fit on the GPU, use 8-bit quantization and note it in the report.
- If `output_router_logits` isn't returned in the installed transformers version, fall back to a forward hook on each `mlp.gate` module.
- If the code dataset is unavailable, use `code_search_net` python.

## Verification
- `pytest tests/` locally: policies on hand-made traces (e.g. LRU on `[1,2,3,1,4]` with K=2 gives a known hit count, and Belady ≥ every other policy on random traces).
- Sanity invariants: hits + misses = total requests, hit rate is monotonic in K for LRU and Belady, and Belady ≥ LRU/LFU/Random.
- M1: perplexity on wikitext is in a reasonable range (roughly 10–20).
- M8: real offload miss count equals the simulator's for the same trace (exact match for LRU).
- Each milestone notebook runs top to bottom on Kaggle from a fresh `git clone`.
