"""Find files made by earlier steps.

On Kaggle, each notebook saves its files to /kaggle/working. A later notebook
can use them by clicking "Add Input" and choosing that notebook's output (or
the traces dataset). Kaggle then puts the files somewhere under /kaggle/input.
find_file() searches our own outputs first, then everything under /kaggle/input.
"""
import glob
import os

import numpy as np

import config


def find_file(name):
    places = [os.path.join(config.RESULTS_DIR, name), os.path.join(config.TRACE_DIR, name)]
    places += sorted(glob.glob(os.path.join(config.INPUT_DIR, "**", name), recursive=True))
    for path in places:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found. Run the step that makes it, or add its "
                            f"notebook output / dataset as an input on Kaggle.")


def load_trace(dataset):
    """dataset = 'wikitext' or 'code'. Returns int array [tokens, layers, top_k]."""
    return np.load(find_file(f"traces_{dataset}.npz"))["trace"]
