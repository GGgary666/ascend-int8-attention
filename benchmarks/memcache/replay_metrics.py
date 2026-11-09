"""Historical replay metric definitions, preserved for comparable reporting."""

import math
import re


def quantiles(values):
    a = sorted(values)
    if not a:
        return None

    def p(q):
        x = q * (len(a) - 1)
        lo, hi = math.floor(x), math.ceil(x)
        return a[lo] + (a[hi] - a[lo]) * (x - lo)

    return {
        "mean": sum(a) / len(a),
        "p50": p(0.5),
        "p90": p(0.9),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": a[-1],
    }


async def fetch_metrics(session, base):
    async with session.get(base + "/metrics") as response:
        response.raise_for_status()
        return await response.text()


def per_engine_gauges(text):
    watched = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"}
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" not in line:
            continue
        name = line.split("{", 1)[0]
        if name not in watched:
            continue
        match = re.search(r'(?:\{|,)engine="([^"]+)"', line)
        if match:
            try:
                result.setdefault(match.group(1), {})[name.removeprefix("vllm:")] = float(
                    line.split()[-1]
                )
            except ValueError:
                continue
    return result


def telemetry_summary(samples):
    result = {}
    engines = sorted({engine for sample in samples for engine in sample["engines"]})
    for engine in engines:
        result[engine] = {}
        for metric in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            values = [
                sample["engines"][engine][metric]
                for sample in samples
                if metric in sample["engines"].get(engine, {})
            ]
            result[engine][metric] = quantiles(values)
    return result
