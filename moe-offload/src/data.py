"""Load the two text datasets and cut them into fixed-length token chunks.

We do NOT train anything. These chunks are only fed through the model so we
can record which experts the router picks for every token.
"""
import torch
from datasets import load_dataset

import config


def iter_texts(name):
    """Yield raw text documents for 'wikitext' or 'code'."""
    if name == "wikitext":
        repo, cfg = config.WIKI_DATASET
        for split in ("test", "validation"):
            # wikitext stores one line per row; glue the lines back into one long text
            yield "".join(load_dataset(repo, cfg, split=split)["text"])

    elif name == "code":
        try:
            # streaming = read files one by one instead of downloading the whole dataset
            ds = load_dataset(config.CODE_DATASET, split="train", streaming=True)
            ds = ds.shuffle(seed=config.SEED, buffer_size=10_000)    # random files, not the first N
            for row in ds:
                yield row["content"]
        except Exception as err:      # dataset moved or unavailable -> fallback
            print("code dataset failed, using fallback:", err)
            repo, cfg = config.CODE_FALLBACK
            for row in load_dataset(repo, cfg, split="test", streaming=True):
                yield row["whole_func_string"]
    else:
        raise ValueError(name)


def make_chunks(tokenizer, name, n_tokens=config.TOKENS_PER_DATASET, chunk_len=config.CHUNK_LEN):
    """Tokenize documents, join them with an end-of-text token, cut into
    chunks of chunk_len tokens, and return a random sample as [n_chunks, chunk_len]."""
    ids = []
    for text in iter_texts(name):
        ids += tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(ids) > 2 * n_tokens:          # read enough text, stop early
            break

    n_all = len(ids) // chunk_len
    chunks = torch.tensor(ids[: n_all * chunk_len]).view(n_all, chunk_len)

    # pick a random (but repeatable) sample of chunks, kept in their original order
    n_want = min(n_tokens // chunk_len, n_all)
    g = torch.Generator().manual_seed(config.SEED)
    pick = torch.randperm(n_all, generator=g)[:n_want].sort().values
    return chunks[pick]
