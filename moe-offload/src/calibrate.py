"""Measure how long it takes to copy one expert from CPU to GPU (step M3).

One OLMoE expert = 3 weight matrices of 2048 x 1024 (gate, up, down)
= 6.3M numbers = 12.6 MB in fp16.
"""
import json
import os
import time

import config

EXPERT_PARAMS = 3 * 2048 * 1024


def time_copy(pinned, repeats=100, warmup=10, numel=EXPERT_PARAMS):
    import torch      # imported here so load_copy_time() works on a laptop without torch
    cpu = torch.empty(numel, dtype=torch.float16)
    if pinned:
        cpu = cpu.pin_memory()       # page-locked RAM: the GPU can read it directly (faster)
    gpu = torch.empty(numel, dtype=torch.float16, device="cuda")
    times = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu.copy_(cpu, non_blocking=pinned)
        torch.cuda.synchronize()     # wait until the copy has really finished
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]    # median is robust to the odd slow run


def calibrate(save=True):
    import torch
    res = {"expert_bytes": EXPERT_PARAMS * 2, "gpu": torch.cuda.get_device_name()}
    for pinned in (False, True):
        t = time_copy(pinned)
        key = "pinned" if pinned else "pageable"
        res[f"copy_time_s_{key}"] = t
        res[f"bandwidth_GBps_{key}"] = res["expert_bytes"] / t / 1e9
    res["copy_time_s"] = res["copy_time_s_pinned"]     # what the simulator uses
    if save:
        with open(os.path.join(config.RESULTS_DIR, "calibration.json"), "w") as f:
            json.dump(res, f, indent=2)
    return res


def load_copy_time():
    """Measured copy time if M3 was run, else the default from config."""
    from src.files import find_file
    try:
        with open(find_file("calibration.json")) as f:
            return json.load(f)["copy_time_s"]
    except FileNotFoundError:
        print("calibration.json not found, using default copy time", config.DEFAULT_COPY_TIME_S)
        return config.DEFAULT_COPY_TIME_S


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
