# Reading Guide: understanding the project top-down

Read in this order: **background → why → what → how it fits together → each piece in detail**.
Each step says what to read, what to look for, and a question you should be able to
answer before moving on. Total: about 4–5 hours, plus Level 0 if MoE is new to you.

---

## Level 0 · Background: Transformers, Mixture of Experts, caching (3–5 h, before anything else)

Never worked with MoE before? Start here. Read **must-read** items in order; the rest are optional.

### 0.1 Transformers (skip if you already know how GPT-style models work)
- **Must-read:** [Transformers, the tech behind LLMs, 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) (video, ~30 min).
  Visual intuition for tokens, attention and feed-forward layers.
- [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/): the same ideas with pictures.
- [Let's build GPT, Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video, ~2 h): builds a small GPT in PyTorch.
  Good if you want to understand the code side.
- [Hugging Face LLM Course, chapter 1](https://huggingface.co/learn/llm-course/chapter1/1): how the `transformers`
  library loads and runs models (we use it in `src/traces.py` and `src/baselines.py`).

> Key point for this project: every Transformer layer = **attention** + a **feed-forward network (FFN)**.
> An MoE model replaces that one FFN with **many FFNs ("experts")** and a **router** that picks a few per token.

### 0.2 Mixture of Experts
- **Must-read:** [A Visual Guide to Mixture of Experts, Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts).
  The easiest introduction: experts, router, top-k, load balancing, with diagrams.
- **Must-read:** [Mixture of Experts Explained, Hugging Face blog](https://huggingface.co/blog/moe).
  Read the TL;DR, "What is a Mixture of Experts?", "What is Sparsity?" and "When to use sparse MoEs vs dense models?".
  Note the TL;DR point that **all experts must be loaded in memory** even though only a few compute per
  token: that is exactly the problem our project works on.
- Papers, if you want the original sources (read abstract + figures first):
  - [Shazeer et al. 2017, Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538): the paper that introduced the modern MoE layer.
  - [Fedus et al. 2021, Switch Transformers](https://arxiv.org/abs/2101.03961): MoE in Transformers, top-1 routing.
  - [Jiang et al. 2024, Mixtral of Experts](https://arxiv.org/abs/2401.04088): a well-known open MoE model (8 experts, top-2).
  - [Fedus et al. 2022, A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667): a survey, for the big picture.

### 0.3 Our model: OLMoE
- **Must-read:** [OLMoE model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924): what "1B-7B" means, 64 experts, top-8.
- [OLMoE paper, Muennighoff et al. 2024](https://arxiv.org/abs/2409.02060): read the abstract, Figure 1, and the
  section analysing **expert routing / specialization**. It shows experts specialise by domain (e.g. code vs text),
  which is why we test on two datasets.
- [OLMoE in the transformers docs](https://huggingface.co/docs/transformers/model_doc/olmoe): the code API we call.

### 0.4 Caching and offloading
- **Must-read:** [Cache replacement policies, Wikipedia](https://en.wikipedia.org/wiki/Cache_replacement_policies):
  read the sections on **Bélády's algorithm**, **LRU** and **LFU**. These are our policies.
- **Must-read:** [Eliseev & Mazur 2023, Fast Inference of MoE Language Models with Offloading](https://arxiv.org/abs/2312.17238)
  ([code](https://github.com/dvmazur/mixtral-offloading)): the closest work to our project. They keep experts in CPU
  memory, use an **LRU expert cache**, and **prefetch** experts by guessing the next layer's experts.
  Our M5, M6 and M8 follow the same ideas on a smaller model.
- [Huang et al. 2023, Towards MoE Deployment: Mitigating Inefficiencies in MoE Inference](https://arxiv.org/abs/2303.06182):
  see **Expert Buffering**: keep hot experts in GPU memory and the rest in CPU memory, the same idea as our cache.

> ✅ Before moving on you should be able to answer:
> - What does the router do, and what does "top-8 of 64" mean?
> - Why does an MoE with 7B parameters compute like a 1B model?
> - What do LRU, LFU and Belady evict?

---

## Level 1 · The problem and the goal (30 min)

**1.1 [`../MoE_Expert_Offloading_Project_Plan.docx`](../MoE_Expert_Offloading_Project_Plan.docx)**: the original idea.
Read sections 1–2 (Project Idea, Goals).

**1.2 [PLAN.md](PLAN.md)**: the plan we are actually following. Read these sections:
- *Context*: what changed from the original idea (show a gain over a dense model, 40 h, Kaggle T4).
- *The "gain over vanilla" story*: the three models we compare and the 5 claims we want to prove.
  This is the heart of the project.
- *Datasets*: why WikiText and Python code, and why there is **no training**.

> ✅ You should be able to answer:
> - Why does an MoE model need less *compute* but not less *memory* than a dense model of the same size?
> - What is a cache "hit" and a "miss" here, and why is a miss slow?
> - What will we compare OLMoE against, and which numbers prove a gain?

---

## Level 2 · The design: how the pieces fit together (30 min)

**2.1 [PLAN.md](PLAN.md)**: read *Key simplifications for students*, *Repo layout* and the *Milestones*
(just the first line of each, M0–M9).

**2.2 [README.md](README.md)**: the table of notebooks: which runs on GPU and which on CPU,
and which outputs feed which later step.

Keep this data flow in mind for the rest of the reading:

```
           GPU (Kaggle T4)                             CPU (no GPU needed)
 ┌──────────────────────────────────┐     ┌──────────────────────────────────────────┐
 │ M1 look at the router            │     │                                          │
 │ M2 text ──OLMoE──> trace files ──┼────>│ M4 analyse traces (hot experts? reuse?)  │
 │    [tokens, 16 layers, 8 experts]│     │ M5 simulator: Random / LRU / LFU         │
 │ M3 copy time per expert ─────────┼────>│ M6 + Belady Oracle + predictor           │
 │    dense vs MoE baselines ───┐   │     │ M7 per-layer cache sizes                 │
 │ M8 REAL offloading (checks   │   │     │                                          │
 │    the simulator) ───────────┼───┼────>│ M9 final tables and plots                │
 └──────────────────────────────┴───┘     └──────────────────────────────────────────┘
```

The key idea of the design: the expensive GPU work (running the model) is done **once**
and saved as a *trace*. All the cache experiments then replay that trace cheaply on the CPU.

> ✅ You should be able to answer:
> - Why do we record traces instead of running the model for every policy?
> - What is the shape of a trace, and what does `trace[t, l]` contain?

---

## Level 3 · Settings (5 min)

**3.1 [config.py](config.py)**: every number the project uses: model names, 16 layers × 64 experts,
top-8, token budget, cache sizes `[8, 16, 32]`, and where files are saved on Kaggle vs your laptop.

---

## Level 4 · The core idea in code: the cache simulator (1–1.5 h)

This is the most important code. Read it slowly.

**4.1 [src/policies.py](src/policies.py)**: *who gets evicted?*
- Read the module docstring: the 3 methods every policy has (`on_use`, `choose_victim`, `on_evict`).
- Then `LRUPolicy` → `LFUPolicy` → `RandomPolicy` → `BeladyPolicy`.
- Look for: Belady is the only one that reads the **future** (`next_use`).

**4.2 [src/simulator.py](src/simulator.py)**: *hit or miss?*
- `LayerCache.request()`: the whole hit/miss/evict logic for one layer and one token.
  Notice why hits are counted **before** loading misses (never evict an expert the current token needs).
- `LayerCache.prefetch()`: copying predicted experts in early.
- `simulate()`: loops over tokens → layers and calls `request()`; optionally asks a predictor.
- `_summary()`: what each result number means (`misses` vs `prefetches` vs `stall_ms_per_token`).

**4.3 [tests/test_policies.py](tests/test_policies.py)**: the best way to check your understanding.
For `test_lru`, `test_lfu_differs_from_lru` and `test_belady`, work out the answer **on paper first**,
then compare with the comment.

**4.4 [notebooks/M5_simulator_basic_policies.ipynb](notebooks/M5_simulator_basic_policies.ipynb)**, section 1:
watch LRU step by step on a tiny example.

> ✅ You should be able to answer:
> - On the trace 1, 2, 3, 1, 2, 3 with 2 slots, why does LRU get 0 hits and Belady 2?
> - Why does every policy give the same result when there are only 8 slots?
> - Why can't a real system use Belady?

---

## Level 5 · Smarter caching (45 min)

**5.1 [src/predictor.py](src/predictor.py)**: a predictor is just a **table of counts**.
- `CrossLayerPredictor.fit()`: counts which layer-(L+1) experts appear together with layer-L experts.
- `predict()`: sums the counts for the current experts and picks the top P.
- `NextTokenPredictor`: same idea, but token t → token t+1.

**5.2 [src/budgets.py](src/budgets.py)**: give layers that miss more a bigger cache.
- `miss_proportional_slots()`: split a fixed total of slots by miss rate (min 8, max 64).
- `compare_uniform_vs_reallocated()`: learn on the *fit* half, test on the unseen *test* half.

**5.3 [src/experiments.py](src/experiments.py)**: `run_grid()` runs every policy × cache size on one
trace. This produces the main results table.

> ✅ You should be able to answer:
> - Why are the predictor's counts learned on the first half of the trace and tested on the second?
> - Why do prefetched copies not count as "stall" time, but still count as "transfers"?

---

## Level 6 · Getting real data from the model (45 min)

**6.1 [src/data.py](src/data.py)**: text → 512-token chunks (`iter_texts`, `make_chunks`).

**6.2 [src/traces.py](src/traces.py)**:
- Module docstring: why one forward pass gives the same routing as token-by-token generation.
- `route()`: one forward pass with `output_router_logits=True` → top-8 experts per layer.
- `collect_traces()`: batches, resume-safe saving, perplexity on the side.

**6.3 [notebooks/M1_load_olmoe_and_routing.ipynb](notebooks/M1_load_olmoe_and_routing.ipynb)**: see the
model's shape, the size of one expert (12.6 MB), and the experts each token picks.

**6.4 [notebooks/M2_collect_traces.ipynb](notebooks/M2_collect_traces.ipynb)**: how traces are made and saved on Kaggle.

---

## Level 7 · Measuring the gain over a dense model (45 min)

**7.1 [src/calibrate.py](src/calibrate.py)**: `time_copy()` measures one expert's CPU→GPU copy
(pinned vs normal memory). This turns "misses" into "milliseconds".

**7.2 [src/energy.py](src/energy.py)**: `EnergyMeter`: energy = power × time, sampled in the background.

**7.3 [src/baselines.py](src/baselines.py)**:
- `count_params()`: total vs **active** parameters (only 8 of 64 experts count for MoE).
- `perplexity_and_speed()`: quality and prefill speed/energy.
- `decode_over_text()` / `decode_speed()`: chatbot-style decoding, one token at a time.
- `run_baseline()`: puts it all together for one model.

**7.4 [notebooks/M3_calibrate_and_baselines.ipynb](notebooks/M3_calibrate_and_baselines.ipynb)**: the gain table.

---

## Level 8 · Real offloading: proving the simulator is right (45 min)

Read this last: it combines everything above.

**8.1 [src/offload.py](src/offload.py)**:
- Module docstring: experts in CPU memory, K slots per layer on the GPU.
- `OffloadedExperts.__init__`: CPU copy of all 64 experts + GPU slot pool.
- `OffloadedExperts.forward`: for each token, ask `LayerCache.request()` (the **same** code as the
  simulator), copy in missing experts with `load()`, then compute.
- `offload_model()`: swap every layer's experts for `OffloadedExperts`.

**8.2 [tests/test_offload.py](tests/test_offload.py)**: on a tiny random model: same outputs as the
original model, and real misses == simulated misses.

**8.3 [notebooks/M8_real_offloading.ipynb](notebooks/M8_real_offloading.ipynb)**: the same check on the
real OLMoE, plus memory/speed/energy.

> ✅ You should be able to answer:
> - Why does sharing `LayerCache` between simulator and real model matter?
> - Why is offloaded OLMoE slower than full OLMoE, and what does it gain in exchange?

---

## Level 9 · Results (15 min)

**9.1** Notebooks [M4](notebooks/M4_trace_analysis.ipynb), [M6](notebooks/M6_oracle_and_predictor.ipynb),
[M7](notebooks/M7_per_layer_budgets.ipynb): analysis and plots (skim the markdown and plot code).

**9.2 [notebooks/M9_final_results.ipynb](notebooks/M9_final_results.ipynb)**: final tables, headline
numbers, and the *Limitations* section. These go straight into the report.

**9.3 Helpers you can skip:** [src/files.py](src/files.py) (finds files from earlier Kaggle notebooks),
[notebooks/M0_setup.ipynb](notebooks/M0_setup.ipynb) (setup check).

---

## Quick reference: which file answers which question

| Question | File |
|---|---|
| Which expert gets evicted? | `src/policies.py` |
| Is this request a hit or a miss? | `src/simulator.py` → `LayerCache.request` |
| What will be needed next? | `src/predictor.py` |
| How many slots per layer? | `src/budgets.py` |
| Which experts did the model really use? | `src/traces.py` → `route` |
| How slow is one copy? | `src/calibrate.py` |
| Is MoE better than dense? | `src/baselines.py`, M3 notebook |
| Does it work on a real GPU? | `src/offload.py`, M8 notebook |
