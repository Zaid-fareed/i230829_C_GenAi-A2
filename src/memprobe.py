"""Peak-memory probe for CPU runs (and CUDA if present).

PyTorch's CPU allocator exposes no peak counter, so we sample the process
resident-set size (RSS) every few milliseconds on a helper thread while the
measured region runs, and report ``peak_rss - rss_before``.  Torch kernels
release the GIL, so the sampler keeps running during matmuls.  Each benchmark
point is run in a fresh subprocess so allocator caching from earlier points does
not hide or inflate the number.
"""
from __future__ import annotations

import threading
import time

import psutil
import torch


class PeakRSS:
    def __init__(self, interval=0.002, use_cuda=None):
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self.base = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        self.cuda_used = torch.cuda.is_available() and getattr(self, "use_cuda", False)
        if self.cuda_used:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); self.cuda_base = torch.cuda.memory_allocated()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *a):
        if self.cuda_used:
            torch.cuda.synchronize()
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def delta_mb(self):
        if torch.cuda.is_available() and self.cuda_used:
            return (torch.cuda.max_memory_allocated() - self.cuda_base) / 2**20   # exact GPU allocator peak
        return (self.peak - self.base) / 2**20

    @property
    def peak_mb(self):
        return self.peak / 2**20
