"""Optional rank-zero validation using CPU shadows of real BF16 cache writes.

Only selected layers are observed. No shadow NPU allocation is made. Enable
<SHAREDKV_VALIDATE_DIR>/collect after warmup in an eager validation server;
remove it before any performance measurement. Reference softmax is CPU FP64.
"""
import json
import os
from pathlib import Path
import re

import torch


class ModelValidator:
    def __init__(self, prefix):
        self.prefix = prefix
        directory = os.environ.get("SHAREDKV_VALIDATE_DIR")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        selected = prefix.startswith("mtp.") or ".mtp." in prefix or bool(re.search(r"\.layers\.(0|1|4|30|31|57|58|60)\.", prefix))
        self.directory = Path(directory) if directory and rank == 0 and selected else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.pages = {"ori": {}, "cmp": {}}
        self.calls = 0
        self.was_enabled = False

    def enabled(self):
        enabled = self.directory is not None and (self.directory / "collect").exists()
        if enabled and torch.npu.is_current_stream_capturing():
            raise RuntimeError("Real-model attention validation requires --enforce-eager")
        if not enabled and self.was_enabled:
            self.pages = {"ori": {}, "cmp": {}}
        self.was_enabled = enabled
        return enabled

    def write(self, cache, updates, slots, kind, scale):
        if not self.enabled() or updates is None or not updates.shape[0]:
            return
        values, ids = updates.detach().cpu().reshape(-1, 512), slots.cpu().long()
        block = cache.shape[1]
        if ids.ndim == 1:
            ids = torch.stack((ids // block, ids % block), -1)
        valid = ((ids[:, 0] >= 0) & (ids[:, 0] < cache.shape[0]) & (ids[:, 1] >= 0) & (ids[:, 1] < block))
        for row in valid.nonzero().flatten().tolist():
            page, offset = ids[row].tolist()
            if page not in self.pages[kind]:
                self.pages[kind][page] = (torch.empty(block, 512, dtype=torch.bfloat16), torch.zeros(block, dtype=torch.bool))
            data, present = self.pages[kind][page]
            data[offset].copy_(values[row])
            present[offset] = True
        values = values[valid].float()
        if not values.numel():
            return
        scale_cpu = scale.detach().cpu()
        quant = (values * scale_cpu.double().reciprocal().float()).round().clamp(-127, 127)
        record = dict(type="cache", layer=self.prefix, kind=kind, elements=values.numel(),
                      clipped=int((values.abs() > scale_cpu * 127).sum()),
                      squared_error=float((values - quant * scale_cpu).square().sum()),
                      squared_reference=float(values.square().sum()))
        self.append(record)

    def append(self, record):
        with (self.directory / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")

    def read(self, kind, slots, block):
        result = []
        for slot in slots.tolist():
            page, offset = divmod(slot, block)
            data = self.pages[kind].get(page)
            if data is None or not data[1][offset]:
                return None
            result.append(data[0][offset])
        return torch.stack(result).double() if result else torch.empty(0, 512, dtype=torch.float64)

    def attention(self, case, output):
        if not self.enabled():
            return
        self.calls += 1
        # Sample both prefill and decode without retaining every model step.
        if self.calls not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256):
            return
        cu, lens = case.cu_q.cpu(), case.seq_lens.cpu()
        batch = next((i for i in range(len(lens)) if cu[i + 1] > cu[i]), None)
        if batch is None:
            return
        row = int(cu[batch + 1]) - 1
        position = int(lens[batch]) - 1
        if case.ori_indices is not None:
            slots = case.ori_indices[row, 0].cpu().long()
            slots = slots[(slots >= 0) & (slots < case.ori.shape[0] * case.ori.shape[1])]
        else:
            index = torch.arange(max(0, position + 1 - case.window), position + 1)
            table = case.ori_block_table[batch].cpu().long()
            slots = table[index // case.ori.shape[1]] * case.ori.shape[1] + index % case.ori.shape[1]
        original = self.read("ori", slots, case.ori.shape[1])
        compressed = torch.empty(0, 512, dtype=torch.float64)
        if case.ratio > 1:
            count = (position + 1) // case.ratio
            index = case.cmp_indices[row, 0].cpu().long() if case.ratio == 4 else torch.arange(count)
            index = index[(index >= 0) & (index < count)]
            table = case.cmp_block_table[batch].cpu().long()
            slots = table[index // case.cmp.shape[1]] * case.cmp.shape[1] + index % case.cmp.shape[1]
            compressed = self.read("cmp", slots, case.cmp.shape[1])
        if original is None or compressed is None:
            self.append(dict(type="attention_skipped", layer=self.prefix, call=self.calls,
                             reason="Selected cache rows predate validation collection"))
            return
        query, sink = case.q[row].detach().cpu().double(), case.sink.detach().cpu().double()
        kv = torch.cat((original, compressed))
        scores = query @ kv.T * case.softmax_scale
        probability = torch.softmax(torch.cat((scores, sink[:, None]), -1), -1)[:, :-1]
        expected = probability @ kv
        actual = output[row].detach().cpu().double()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise RuntimeError(f"Non-finite attention output in {self.prefix}")
        cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1)
        difference = actual - expected
        record = dict(type="attention", layer=self.prefix, call=self.calls, ratio=case.ratio,
                      query_tokens=int(cu[batch + 1] - cu[batch]), sequence_length=int(lens[batch]),
                      original_keys=len(original), compressed_keys=len(compressed),
                      min_head_cosine=float(cosine.min()), mean_head_cosine=float(cosine.mean()),
                      relative_l2=float(difference.norm() / expected.norm().clamp_min(1e-30)),
                      max_abs=float(difference.abs().max()))
        self.append(record)
        path = self.directory / f"{self.prefix}.call_{self.calls}.pt"
        torch.save(dict(query=query, original=original, compressed=compressed, sink=sink,
                        softmax_scale=case.softmax_scale, actual=actual, reference=expected), path)
