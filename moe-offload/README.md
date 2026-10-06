# MoE Expert Offloading with a Cache Simulator

A Mixture-of-Experts model (OLMoE-1B-7B) uses only 8 of its 64 experts per layer for each token,
so it needs **~5x less compute per token** than a dense model of the same size (OLMo-7B).
But all experts must still be stored, which takes ~14 GB of GPU memory.

We keep the experts in CPU memory, give each layer a small **cache of K experts on the GPU**,
and study which caching policy (Random, LRU, LFU, a count-based predictor, and the Belady
Oracle) avoids the most slow CPU→GPU copies.

See [PLAN.md](PLAN.md) for the full plan and the reasoning behind each step.
New to the project? Follow [READING_GUIDE.md](READING_GUIDE.md): it lists what to read, in order.

## Repository layout

```
config.py            all settings (models, datasets, cache sizes, paths)
src/                 reusable code (tested)
  data.py            load WikiText + Python code, cut into 512-token chunks
  traces.py          load OLMoE, record the 8 experts per token per layer
  calibrate.py       measure CPU->GPU copy time of one expert
  energy.py          measure GPU energy (joules)
  baselines.py       perplexity, speed, memory, energy of a model
  policies.py        Random, LRU, LFU, Belady eviction policies
  predictor.py       count-based predictors for prefetching
  simulator.py       LayerCache + simulate(): replay a trace against a cache
  budgets.py         equal vs per-layer cache sizes
  experiments.py     run all policies x cache sizes on a trace
  offload.py         REAL offloading: OLMoE with K expert slots per layer on the GPU
  files.py           find files made by earlier notebooks
notebooks/           one Kaggle notebook per milestone (M0 ... M9)
tests/               unit tests (run: python -m pytest -q tests)
docs/                short notes per milestone (what we did, what we saw)
```

## Running on Kaggle

1. Push this repo to GitHub.
2. In each notebook, set `REPO_URL` in the first cell to your repo (for a private repo, see the comment in that cell).
3. Upload the notebook to Kaggle (**File → Import Notebook**) and run it top to bottom.
   The first cell clones/pulls the latest code, so after a `git push` you only need to rerun it.

Run the notebooks in order:

| Notebook | Accelerator | Add as input | Produces | Time |
|---|---|---|---|---|
| M0_setup | GPU T4 | – | (checks only) | 5 min |
| M1_load_olmoe_and_routing | GPU T4 | – | (exploration) | 10 min |
| M2_collect_traces | GPU T4 | – | `traces_wikitext.npz`, `traces_code.npz` → save as dataset **moe-traces** | 30–60 min |
| M3_calibrate_and_baselines | GPU T4 | – | `calibration.json`, `baselines.csv` | 1.5–2 h |
| M4_trace_analysis | None (CPU) | moe-traces | usage plots | 5 min |
| M5_simulator_basic_policies | None | moe-traces, M3 output | `M5_basic_policies.csv` | 10–20 min |
| M6_oracle_and_predictor | None | moe-traces, M3 output | `policy_results.csv` | ~1 h |
| M7_per_layer_budgets | None | moe-traces, M3 output | `budgets.csv` | 10–20 min |
| M8_real_offloading | GPU T4 | M3 output | `offload.csv` | ~1 h |
| M9_final_results | None | M3, M6, M7, M8 outputs | final tables and figures | 5 min |

"Add as input" means: **Add Input → Your Work → Notebooks** (pick that notebook's saved version),
or **Datasets → moe-traces**. Use **Save Version → Save & Run All** so a notebook's output is kept.

GPU time in total: about 4–5 hours (Kaggle gives ~30 GPU-hours per week).

## Working locally

The simulator, tests and CPU notebooks (M4–M7, M9) also run on a laptop:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Locally, outputs go to `outputs/` (not committed). Copy final CSVs/plots you want to keep into `results/`.
