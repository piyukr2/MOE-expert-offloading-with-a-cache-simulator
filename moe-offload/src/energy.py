"""Measure GPU energy: sample the power draw in a background thread and add it up.

Usage:
    with EnergyMeter() as m:
        run_model()
    print(m.joules, m.seconds)
"""
import threading
import time


class EnergyMeter:
    def __init__(self, gpu_index=0, interval_s=0.05):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.joules = 0.0
        self.seconds = 0.0
        self.available = True

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as err:       # no NVIDIA GPU / no driver -> just measure time
            print("energy not available:", err)
            self.available = False
        self._stop = threading.Event()
        self._start = time.perf_counter()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000   # mW -> W
            now = time.perf_counter()
            self.joules += watts * (now - last)                              # energy = power x time
            last = now

    def __exit__(self, *exc):
        self._stop.set()
        if self.available:
            self._thread.join()
        self.seconds = time.perf_counter() - self._start
        return False
