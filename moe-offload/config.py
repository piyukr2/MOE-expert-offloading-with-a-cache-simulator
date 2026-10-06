"""All project settings in one place. Change values here, not inside src/."""
import os

# ---------- Models (all from AI2's OLMo family) ----------
MOE_MODEL = "allenai/OLMoE-1B-7B-0924"          # 64 experts/layer, top-8
DENSE_SMALL = "allenai/OLMo-1B-0724-hf"         # same compute per token as OLMoE
DENSE_LARGE = "allenai/OLMo-7B-0724-hf"         # same total size as OLMoE ("vanilla")

NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8

# ---------- Datasets (inference input only, no training) ----------
WIKI_DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1")
CODE_DATASET = "codeparrot/codeparrot-clean-valid"
CODE_FALLBACK = ("code_search_net", "python")

TOKENS_PER_DATASET = 200_000
CHUNK_LEN = 512
BATCH_SIZE = 1          # chunks per forward pass; OLMoE fills ~13 of the T4's 14.5 GB, so keep this small
SEED = 0

# ---------- Cache experiments ----------
CACHE_SIZES = [8, 16, 32]      # slots per layer, out of 64 (12.5%, 25%, 50%)
PREFETCH_P = 4                 # experts the predictor prefetches per layer
FIT_FRACTION = 0.5             # first half of a trace = fit, second half = test

# Fallback copy time per expert (seconds), replaced by the M3 measurement
DEFAULT_COPY_TIME_S = 1.1e-3

# ---------- Paths ----------
# On Kaggle everything we create goes to /kaggle/working (it becomes the
# notebook's "Output", which later notebooks can add as an input).
# On a laptop it goes to ./outputs inside the repo.
ON_KAGGLE = os.path.exists("/kaggle")
OUTPUT_DIR = "/kaggle/working" if ON_KAGGLE else os.path.join(os.path.dirname(__file__), "outputs")
TRACE_DIR = os.path.join(OUTPUT_DIR, "traces")       # big trace files (.npz)
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")    # small results: csv, json, png
INPUT_DIR = "/kaggle/input"                          # where Kaggle puts attached datasets/outputs

os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
