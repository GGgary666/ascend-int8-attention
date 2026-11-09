"""Preserved FIFO replay telemetry helpers; MemCache counters are added explicitly."""

import asyncio
import json
import re
import time
from . import replay_metrics as replay
from .capacity_client import get_json

COUNTERS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:ascend_store_get_success_bytes_total",
    "vllm:ascend_store_get_failed_bytes_total",
    "vllm:ascend_store_put_success_bytes_total",
    "vllm:ascend_store_put_failed_bytes_total",
    "vllm:ascend_store_get_success_keys_total",
    "vllm:ascend_store_get_failed_keys_total",
    "vllm:ascend_store_put_success_keys_total",
    "vllm:ascend_store_put_failed_keys_total",
)


def counters_by_engine(text):
    values = {key: {} for key in COUNTERS}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" not in line:
            continue
        name, rest = line.split("{", 1)
        if name not in values:
            continue
        match = re.search(r'(?:^|,)engine="([^"]+)"', rest)
        if not match:
            continue
        engine = match.group(1)
        reason = re.search(r'(?:^|,)finished_reason="([^"]+)"', rest)
        label = reason.group(1) if reason else "total"
        try:
            value = float(line.split()[-1])
        except ValueError:
            continue
        values[name].setdefault(engine, {})[label] = value
    return values


def delta_counters(before, after):
    out = {}
    for name in COUNTERS:
        out[name] = {}
        for engine in ("0", "1"):
            a, b = before[name].get(engine, {}), after[name].get(engine, {})
            out[name][engine] = {key: b.get(key, 0) - a.get(key, 0) for key in a.keys() | b.keys()}
    return out


def counter_total(delta, name):
    return sum(v for groups in delta[name].values() for v in groups.values())


def finished_ok(delta):
    return sum(
        value
        for reasons in delta["vllm:request_success_total"].values()
        for reason, value in reasons.items()
        if reason not in ("abort", "error")
    )


async def settled_metrics(session, base, *, before=None, expected=None, timeout=45):
    deadline = time.monotonic() + timeout
    last_signature = None
    stable = 0
    while time.monotonic() < deadline:
        text = await replay.fetch_metrics(session, base)
        gauges = replay.per_engine_gauges(text)
        parsed = counters_by_engine(text)
        idle = all(
            gauges.get(rank, {}).get("num_requests_running") == 0
            and gauges.get(rank, {}).get("num_requests_waiting") == 0
            for rank in ("0", "1")
        )
        signature = json.dumps(parsed, sort_keys=True)
        stable = stable + 1 if idle and signature == last_signature else 0
        enough = (
            before is None
            or expected is None
            or finished_ok(delta_counters(before, parsed)) >= expected
        )
        if idle and stable >= 1 and enough:
            return text, parsed
        last_signature = signature
        await asyncio.sleep(1)
    raise TimeoutError("per-DP metrics did not settle or finished-request counts did not arrive")


class Timeline:
    def __init__(self, path, anchor, concurrency):
        self.stream = path.open("w", encoding="utf-8")
        self.anchor = anchor
        self.limit = concurrency
        self.active = 0
        self.events = []
        self.sent_ids = []

    def send(self, ctx, timestamp):
        self.active += 1
        if self.active > self.limit:
            raise RuntimeError(f"client concurrency {self.active} exceeded {self.limit}")
        self.sent_ids.append(ctx["id"])
        self.add("send", ctx["id"], timestamp)

    def finish(self, ctx, timestamp):
        if ctx.get("sent_at") is None:
            return
        self.active -= 1
        if self.active < 0:
            raise RuntimeError("negative in-flight count")
        self.add("finish", ctx["id"], timestamp)

    def add(self, kind, identity, timestamp):
        value = {
            "kind": kind,
            "id": identity,
            "offset_s": timestamp - self.anchor,
            "in_flight": self.active,
        }
        self.events.append(value)
        self.stream.write(json.dumps(value) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()

    def stats(self):
        if not self.events:
            return {"max": 0, "mean": None, "events": 0}
        area = 0.0
        for a, b in zip(self.events, self.events[1:]):
            area += (b["offset_s"] - a["offset_s"]) * a["in_flight"]
        span = self.events[-1]["offset_s"] - self.events[0]["offset_s"]
        return {
            "max": max(x["in_flight"] for x in self.events),
            "mean": area / span if span > 0 else None,
            "events": len(self.events),
        }


async def sample_metrics(
    session, base, out, anchor, stop, timeline, *, capacity_host=None, capacity_port=17000
):
    samples, errors = [], []
    with (out / "telemetry.jsonl").open("w", encoding="utf-8") as f:
        while True:
            try:
                raw = await replay.fetch_metrics(session, base)
                gauges = replay.per_engine_gauges(raw)
                required = {"num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"}
                if any(not required.issubset(gauges.get(rank, {})) for rank in ("0", "1")):
                    raise ValueError(f"missing DP gauges: {gauges}")
                item = {
                    "offset_s": time.perf_counter() - anchor,
                    "client_in_flight": timeline.active,
                    "engines": gauges,
                    "counters_by_engine": counters_by_engine(raw),
                }
                if capacity_host:
                    capacity = await asyncio.to_thread(
                        get_json,
                        host=capacity_host,
                        port=capacity_port,
                        path="/api/v1/capacity/segment_remaining",
                    )
                    assert capacity["status"] == 200
                    assert (
                        capacity["body"]["data_source_ready"] and not capacity["body"]["degraded"]
                    )
                    segments = [
                        s
                        for s in capacity["body"]["segments"]
                        if s["medium"] == "DRAM" and s["total_bytes"]
                    ]
                    assert len(segments) == 2 and all(
                        s["total_bytes"] == 256 * 1024**3 for s in segments
                    )
                    for segment in segments:
                        segment["payload_bytes"] = (
                            segment["used_bytes"] - segment["fragmentation_bytes"]
                        )
                        assert segment["payload_bytes"] >= 0
                    item["memcache_capacity"] = capacity
                samples.append(item)
            except Exception as exc:
                item = {"offset_s": time.perf_counter() - anchor, "error": repr(exc)}
                errors.append(item)
            f.write(json.dumps(item) + "\n")
            f.flush()
            if stop.is_set():
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
    return samples, errors
