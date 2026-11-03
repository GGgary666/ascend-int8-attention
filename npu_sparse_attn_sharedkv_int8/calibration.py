"""Observe actual post-RoPE cache updates without changing BF16 execution.

Collection is explicitly gated by <directory>/collect. Create this marker only
after server warmup, remove it, then send one final request to flush each rank.
This mode deliberately synchronizes during checkpoints; never benchmark it.
"""

import atexit
import os
from pathlib import Path
import time

import torch


class CacheObserver:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.marker = self.directory / "collect"
        self.stats = {}
        self.dirty = False
        self.last_save = time.monotonic()
        atexit.register(self.flush)

    @torch.inference_mode()
    def observe(self, prefix, kind, cache, updates, slots):
        if not self.marker.exists():
            self.flush()
            return
        if updates is None or updates.shape[0] == 0:
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Calibration requires --enforce-eager; marker must be absent during warmup")
        if updates.shape[-1] != 512 or cache.shape[-1] != 512:
            raise ValueError("Calibration expects the D=512 shared KV cache")
        if slots.ndim == 1:
            valid = (slots >= 0) & (slots < cache.shape[0] * cache.shape[1])
        else:
            valid = ((slots[:, 0] >= 0) & (slots[:, 0] < cache.shape[0])
                     & (slots[:, 1] >= 0) & (slots[:, 1] < cache.shape[1]))
        values = updates.reshape(-1, 512).float()
        values = torch.where(valid[:, None], values, 0.0)
        maximum = values.abs().amax(0)
        count = valid.sum()
        key = f"{prefix}.{kind}"
        if key in self.stats:
            old = self.stats[key]
            old[0].copy_(torch.maximum(old[0], maximum))
            old[1].add_(count)
        else:
            self.stats[key] = [maximum, count]
        self.dirty = True
        if time.monotonic() - self.last_save >= 10:
            self.flush()

    def flush(self):
        if not self.dirty:
            return
        keys = sorted(self.stats)
        maxima = torch.stack([self.stats[k][0] for k in keys]).cpu()
        counts = torch.stack([self.stats[k][1] for k in keys]).cpu()
        if not torch.isfinite(maxima).all():
            raise RuntimeError("Non-finite real-model cache values encountered during calibration")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        payload = {"version": 1, "rank": rank, "pid": os.getpid(),
                   "stats": {k: {"maximum": maxima[i], "rows": int(counts[i])} for i, k in enumerate(keys)}}
        path = self.directory / f"rank_{rank:02d}.pt"
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.dirty = False
        self.last_save = time.monotonic()
