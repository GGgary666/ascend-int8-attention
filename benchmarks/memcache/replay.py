#!/usr/bin/env python3
"""Replay the Thursday interval with one immutable global head-of-line queue."""

import argparse
import asyncio
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import aiohttp
import numpy as np

from .read_dataset import iter_requests
from .scheduling import OrderedDispatcher, ready_at, may_skip_idle
from . import replay_metrics as replay
from . import replay_runtime as replay128

HERE = Path(__file__).resolve().parent
DATA = None  # Set explicitly by the CLI; the dataset is not distributed with source.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(stage, run_id):
    if not run_id or "REPLACE_WITH" in run_id:
        raise ValueError("a unique, concrete run_id is required")
    rows = list(iter_requests(DATA, run_id=run_id))
    rows.sort(key=lambda x: x["metadata"]["replay_order"])
    dp_counts = collections.Counter()
    seen_identities = set()
    for item in rows:
        m = item["metadata"]
        m["is_long_history_followup"] = m["is_long_private_history_followup"]
        m["first_identity_request"] = m["identity_scope_id"] not in seen_identities
        seen_identities.add(m["identity_scope_id"])
        # Keep the original metadata and add aliases used by the stream collector.
        m.update(
            benchmark_id=m["sample_id"],
            planned_merge_order=m["replay_order"],
            dp_order=dp_counts[m["dp_rank"]],
            stage="replay",
            busy_think_time_s=m["busy_parent_gap_s"],
            replay_prefix_tokens=m["common_checkpoint_tokens_4096"],
            is_continuation=m["is_history_followup"],
            kv_reuse_planned=m["is_history_followup"],
            conversation_parent_request_id=m["availability_parent_id"],
            conversation_think_time_s=m["busy_parent_gap_s"],
            arrival_policy="after_parent_completion" if m["availability_parent_id"] else "absolute",
            replay_arrival_offset_s=m["busy_arrival_offset_s"],
        )
        dp_counts[m["dp_rank"]] += 1
        assert item["request"]["request_id"] == m["sample_id"]
    known = {item["metadata"]["id"]: item for item in rows}
    if len(known) != len(rows):
        raise ValueError("duplicate source request")
    for item in rows:
        m, b = item["metadata"], item["request"]
        if (
            hashlib.sha256(np.asarray(b["prompt"], dtype="<u4").tobytes()).hexdigest()
            != m["prompt_sha256"]
        ):
            raise ValueError("prompt fingerprint mismatch")
        if len(b["prompt"]) != m["input_tokens"] or b["max_tokens"] != m["output_tokens"]:
            raise ValueError("frozen lengths changed")
        if len(b["prompt"]) + b["max_tokens"] > 135000:
            raise ValueError("full request exceeds model context")
        if m["arrival_policy"] == "after_parent_completion":
            p = known[m["conversation_parent_request_id"]]["metadata"]
            assert p["replay_order"] < m["replay_order"]
            assert p["dp_rank"] == m["dp_rank"] and p["identity_scope_id"] == m["identity_scope_id"]
        if m["prefix_reference_id"]:
            parent = known[m["prefix_reference_id"]]
            k = m["frozen_shared_prefix_tokens"]
            assert b["prompt"][:k] == parent["request"]["prompt"][:k]
            assert b["cache_salt"] == parent["request"]["cache_salt"]
        assert item["headers"]["X-data-parallel-rank"] == str(m["dp_rank"])
    assert [r["metadata"]["replay_order"] for r in rows] == list(range(len(rows)))
    return rows


def metric_group(rows):
    known = [r for r in rows if r["cached_tokens"] is not None]
    return {
        "requests": len(rows),
        "cache_count_coverage": len(known),
        "hit_requests": sum(r["cached_tokens"] > 0 for r in known),
        "cached_tokens": sum(r["cached_tokens"] for r in known),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "modeled_prefix_tokens": sum(r["replay_prefix_tokens"] for r in rows),
        "ttft_ms": replay.quantiles([r["ttft_ms"] for r in rows]),
        "tpot_ms": replay.quantiles([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
    }


async def run(o):
    # Share the model driver's advisory lock with functional/performance tests.
    driver_lock = None
    if getattr(o, "driver_lock", None):
        import fcntl

        driver_lock = Path(o.driver_lock).open("a")
        fcntl.flock(driver_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = load(o.stage, o.run_id)
    clock = "busy"
    plan = json.loads((DATA / "run_plan.json").read_text())
    per_dp_limit = plan["max_client_inflight_per_dp"]
    global_limit = plan["max_global_client_inflight"]
    out = Path(o.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "stage": o.stage,
        "precision": o.precision,
        "run_id": o.run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base": o.base,
        "client_concurrency_limit": global_limit,
        "max_client_inflight_per_dp": per_dp_limit,
        "clock": clock,
        "strict_global_order": True,
        "routing": "frozen payload headers",
        "request_id": "original frozen payload field retained",
        "arrival": "max(frozen arrival, actual parent completion + frozen think time)",
        "idle_skip": "only when client and both DP engines have no running or waiting work",
        "payload_sha256": sha(DATA / "payloads.jsonl.gz"),
        "run_plan_sha256": sha(DATA / "run_plan.json"),
        "client_sha256": sha(__file__),
        "counter_scope": "local prefix hits + external prefix hits; both retained separately",
        "native_capacity_host": getattr(o, "capacity_host", None),
        "native_capacity_port": getattr(o, "capacity_port", 17000),
        "wall_definition": "first request headers sent to last full response completion; startup/warmup excluded",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    trace = aiohttp.TraceConfig()
    timeline = None
    anchor = skipped = 0.0

    async def sent(_session, context, _params):
        ctx = context.trace_request_ctx
        if ctx is not None:
            ctx["sent_at"] = time.perf_counter()
            ctx["logical_sent_s"] = ctx["sent_at"] - anchor + skipped
            timeline.send(ctx, ctx["sent_at"])
            ctx["sent_event"].set()

    trace.on_request_headers_sent.append(sent)
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=3600)
    records, failures, skips = [], [], []
    done_logical, done_records, failed_ids = {}, {}, set()
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0),
        trace_configs=[trace],
        read_bufsize=8 * 1024 * 1024,
    ) as session:
        before_raw, before = await replay128.settled_metrics(session, o.base)
        (out / "metrics-before.txt").write_text(before_raw)
        anchor, cpu_start = time.perf_counter(), time.process_time()
        timeline = replay128.Timeline(out / "concurrency-events.jsonl", anchor, global_limit)
        stop = asyncio.Event()
        sampling = asyncio.create_task(
            replay128.sample_metrics(
                session,
                o.base,
                out,
                anchor,
                stop,
                timeline,
                capacity_host=getattr(o, "capacity_host", None),
                capacity_port=getattr(o, "capacity_port", 17000),
            )
        )
        dispatcher = OrderedDispatcher(
            [item["metadata"] for item in rows],
            max_inflight_per_dp=per_dp_limit,
            max_global_inflight=global_limit,
        )
        lookup = {item["metadata"]["id"]: item for item in rows}
        tasks = {}
        submit_counts = collections.Counter()
        global_submits = 0
        with (
            (out / "requests.jsonl").open("w") as stream,
            (out / "failures.jsonl").open("w") as errs,
            (out / "submissions.jsonl").open("w") as submissions,
            (out / "idle-skips.jsonl").open("w") as idle_stream,
        ):

            async def send_one(item, ctx):
                m, body = item["metadata"], item["request"]
                first = last = usage = returned_prompt = returned_id = finish_reason = None
                generated = []
                try:
                    async with session.post(
                        o.base + "/v1/completions",
                        json=body,
                        headers=item["headers"],
                        trace_request_ctx=ctx,
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                f"HTTP {response.status}: {(await response.text())[:1500]}"
                            )
                        async for line in response.content:
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                break
                            event = json.loads(data)
                            if event.get("error"):
                                raise RuntimeError(str(event["error"]))
                            if event.get("id"):
                                returned_id = event["id"]
                            if event.get("usage"):
                                usage = event["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("finish_reason") is not None:
                                    finish_reason = choice["finish_reason"]
                                if choice.get("prompt_token_ids") is not None:
                                    returned_prompt = choice["prompt_token_ids"]
                                new = choice.get("token_ids") or []
                                if new:
                                    now = time.perf_counter()
                                    first = now if first is None else first
                                    last = now
                                    generated.extend(new)
                    end = time.perf_counter()
                    if ctx["sent_at"] is None or usage is None or first is None:
                        raise ValueError("missing stream/usage/timing metadata")
                    if (
                        returned_id != "cmpl-" + body["request_id"]
                        or returned_prompt != body["prompt"]
                    ):
                        raise ValueError("response request ID or echoed prompt differs")
                    if (
                        usage["prompt_tokens"] != m["input_tokens"]
                        or usage["completion_tokens"] != m["output_tokens"]
                        or len(generated) != m["output_tokens"]
                    ):
                        raise ValueError(f"incomplete output: {usage}, returned={len(generated)}")
                    if finish_reason != "length":
                        raise ValueError(
                            f"unexpected finish reason for full fixed output: {finish_reason!r}"
                        )
                    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    if cached is not None and not 0 <= cached <= m["input_tokens"]:
                        raise ValueError("invalid cached token count")
                    rec = {
                        **m,
                        "request_id": body["request_id"],
                        "service_request_id": returned_id,
                        "dp_submit_index": ctx["dp_submit_index"],
                        "actual_submit_s": ctx["sent_at"] - anchor,
                        "global_submit_index": ctx["global_submit_index"],
                        "first_token_s": first - anchor,
                        "end_s": end - anchor,
                        "success": True,
                        "prompt_tokens": usage["prompt_tokens"],
                        "cached_tokens": cached,
                        "submitted_offset_s": ctx["sent_at"] - anchor,
                        "first_token_offset_s": first - anchor,
                        "finished_offset_s": end - anchor,
                        "logical_submitted_s": ctx["logical_sent_s"],
                        "logical_ready_s": ctx["logical_ready_s"],
                        "logical_finished_s": end - anchor + skipped,
                        "ttft_s": first - ctx["sent_at"],
                        "ttft_ms": 1000 * (first - ctx["sent_at"]),
                        "tpot_ms": 1000 * (last - first) / (len(generated) - 1)
                        if len(generated) > 1
                        else None,
                        "seconds": end - ctx["sent_at"],
                        "usage": usage,
                        "finish_reason": finish_reason,
                        "generated_token_ids": generated,
                        "generated_sha256": hashlib.sha256(
                            np.asarray(generated, dtype="<u4").tobytes()
                        ).hexdigest(),
                    }
                    records.append(rec)
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                k: rec[k]
                                for k in ["request_id", "ttft_ms", "tpot_ms", "cached_tokens"]
                            }
                        ),
                        flush=True,
                    )
                    return rec
                except Exception as exc:
                    fail = {
                        **m,
                        "request_id": body["request_id"],
                        "success": False,
                        "error": repr(exc),
                        "partial_output_tokens": len(generated),
                        "sent": ctx["sent_at"] is not None,
                        "finished_offset_s": time.perf_counter() - anchor,
                    }
                    failures.append(fail)
                    errs.write(json.dumps(fail) + "\n")
                    errs.flush()
                    print(
                        json.dumps({"failure": body["request_id"], "error": repr(exc)}), flush=True
                    )
                    return None
                finally:
                    if ctx["sent_at"] is not None:
                        timeline.finish(ctx, time.perf_counter())
                    ctx["sent_event"].set()

            while not dispatcher.empty() or tasks:
                # First observe completion, so causal readiness is independent of the loop's wake latency.
                for task in list(tasks):
                    if task.done():
                        item = tasks.pop(task)
                        rec = task.result()
                        ident = item["metadata"]["id"]
                        if rec is None:
                            failed_ids.add(ident)
                        else:
                            done_logical[ident] = rec["logical_finished_s"]
                            done_records[ident] = rec
                now = time.perf_counter() - anchor + skipped
                failed_reference = any(
                    r["availability_parent_id"] in failed_ids for r in dispatcher.pending
                )
                inflight = collections.Counter(
                    item["metadata"]["dp_rank"] for item in tasks.values()
                )
                ready = (
                    [] if failed_reference else dispatcher.take_ready(now, done_logical, inflight)
                )
                for m in ready:
                    item = lookup[m["id"]]
                    ctx = {
                        "id": m["benchmark_id"],
                        "sent_at": None,
                        "sent_event": asyncio.Event(),
                        "logical_ready_s": ready_at(m, done_logical),
                        "global_submit_index": global_submits,
                        "dp_submit_index": submit_counts[m["dp_rank"]],
                    }
                    assert global_submits == m["replay_order"], "global source order changed"
                    global_submits += 1
                    assert ctx["dp_submit_index"] == m["dp_order"], "per-DP frozen order changed"
                    submit_counts[m["dp_rank"]] += 1
                    task = asyncio.create_task(send_one(item, ctx))
                    tasks[task] = item
                    await ctx["sent_event"].wait()
                    if ctx["sent_at"] is not None:
                        sub = {
                            "request_id": m["benchmark_id"],
                            "id": m["id"],
                            "replay_order": m["replay_order"],
                            "dp_submit_index": ctx["dp_submit_index"],
                            "global_submit_index": ctx["global_submit_index"],
                            "dp_rank": m["dp_rank"],
                            "offset_s": ctx["sent_at"] - anchor,
                            "logical_sent_s": ctx["logical_sent_s"],
                            "logical_ready_s": ctx["logical_ready_s"],
                        }
                        submissions.write(json.dumps(sub) + "\n")
                        submissions.flush()
                if dispatcher.empty() and not tasks:
                    break
                if failed_reference:
                    if tasks:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        continue
                    raise RuntimeError(
                        "history reference failed; stopped new submissions and drained in-flight work: "
                        + repr(failed_ids)
                    )
                now = time.perf_counter() - anchor + skipped
                inflight = collections.Counter(
                    item["metadata"]["dp_rank"] for item in tasks.values()
                )
                next_ready = dispatcher.next_release(done_logical, inflight)
                if tasks:
                    wait = max(0.001, next_ready - now) if next_ready is not None else None
                    await asyncio.wait(tasks, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
                    continue
                if next_ready is None:
                    raise RuntimeError(
                        "unschedulable descendants after failure: " + repr(failed_ids)
                    )
                if next_ready <= now:
                    continue
                # No active client requests; verify both actual DP engines before advancing the replay clock.
                try:
                    raw = await replay.fetch_metrics(session, o.base)
                    gauges = replay.per_engine_gauges(raw)
                    if any(
                        not {"num_requests_running", "num_requests_waiting"}.issubset(
                            gauges.get(rank, {})
                        )
                        for rank in ("0", "1")
                    ):
                        raise ValueError("missing DP gauges for idle skip")
                except Exception as exc:
                    with (out / "idle-state-errors.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "offset_s": time.perf_counter() - anchor,
                                    "error": repr(exc),
                                    "clock_advanced": False,
                                }
                            )
                            + "\n"
                        )
                    await asyncio.sleep(0.5)
                    continue
                running = {int(r): gauges[r]["num_requests_running"] for r in ("0", "1")}
                waiting = {int(r): gauges[r]["num_requests_waiting"] for r in ("0", "1")}
                now = time.perf_counter() - anchor + skipped
                allowed = may_skip_idle(
                    client_inflight=timeline.active, dp_running=running, dp_waiting=waiting
                )
                jump = max(0.0, next_ready - now) if allowed else 0.0
                if jump:
                    event = {
                        "wall_offset_s": time.perf_counter() - anchor,
                        "logical_before_s": now,
                        "skipped_s": jump,
                        "engines": gauges,
                    }
                    skipped += jump
                    skips.append(event)
                    idle_stream.write(json.dumps(event) + "\n")
                    idle_stream.flush()
                else:
                    await asyncio.sleep(min(0.2, max(0.001, next_ready - now)))
        cpu_seconds = time.process_time() - cpu_start
        stop.set()
        samples, errors = await sampling
        timeline.close()
        after_raw, after = await replay128.settled_metrics(
            session, o.base, before=before, expected=len(records)
        )
        (out / "metrics-after.txt").write_text(after_raw)
    delta = replay128.delta_counters(before, after)
    wall = timeline.events[-1]["offset_s"] - timeline.events[0]["offset_s"]
    rank_counts = {str(rank): sum(r["dp_rank"] == rank for r in records) for rank in (0, 1)}
    rank_success = {
        rank: sum(
            v
            for reason, v in delta["vllm:request_success_total"][rank].items()
            if reason not in ("abort", "error")
        )
        for rank in ("0", "1")
    }
    required = all(
        all(rank in before[n] and rank in after[n] for rank in ("0", "1"))
        for n in replay128.COUNTERS
    )
    groups = {
        "all": records,
        "continuation": [r for r in records if r["is_continuation"]],
        "long_history": [r for r in records if r["is_long_history_followup"]],
        "short_history": [
            r for r in records if r["is_history_followup"] and not r["is_long_history_followup"]
        ],
        "extra_reuse": [r for r in records if r["kv_reuse_planned"] and not r["is_continuation"]],
        "cold": [r for r in records if not r["kv_reuse_planned"]],
    }
    causal_ok = True
    for r in records:
        if r["logical_submitted_s"] + 1e-5 < r["logical_ready_s"]:
            causal_ok = False
        if r["conversation_parent_request_id"]:
            p = done_records[r["conversation_parent_request_id"]]
            if (
                r["submitted_offset_s"] + 1e-5
                < p["finished_offset_s"] + r["conversation_think_time_s"]
            ):
                causal_ok = False
    # A fully idle skip may remove physical think time; logical causality is the invariant in that case.
    if skips:
        causal_ok = all(r["logical_submitted_s"] + 1e-5 >= r["logical_ready_s"] for r in records)
    cached_complete = all(r["cached_tokens"] is not None for r in records)
    cached_sum = sum(r["cached_tokens"] for r in records) if cached_complete else None
    counter = replay128.counter_total
    checks = {
        "complete": len(records) == len(rows) and not failures,
        "cache_accounting": cached_complete,
        "causal_arrivals": causal_ok,
        "dp_success_counters": rank_counts == rank_success,
        "counter_coverage": required,
        "telemetry": not errors and bool(samples),
        "no_client_work_left": timeline.active == 0,
        "prompt_counter": counter(delta, "vllm:prompt_tokens_total")
        == sum(r["input_tokens"] for r in records),
        "output_counter": counter(delta, "vllm:generation_tokens_total")
        == sum(r["output_tokens"] for r in records),
        "cache_counter": cached_complete
        and (
            counter(delta, "vllm:prefix_cache_hits_total")
            + counter(delta, "vllm:external_prefix_cache_hits_total")
        )
        == cached_sum,
        "memcache_no_failed_io": all(
            counter(delta, name) == 0
            for name in replay128.COUNTERS
            if "ascend_store" in name and "_failed_" in name
        ),
    }
    for rank in (0, 1):
        sent_rows = sorted(
            [r for r in records if r["dp_rank"] == rank], key=lambda r: r["actual_submit_s"]
        )
        checks[f"dp{rank}_fixed_order"] = [r["dp_order"] for r in sent_rows] == list(
            range(len(sent_rows))
        )
    ordered = sorted(records, key=lambda r: r["actual_submit_s"])
    checks["global_fixed_order"] = [r["replay_order"] for r in ordered] == list(range(len(rows)))
    checks["global_submit_indices"] = all(
        r["global_submit_index"] == r["replay_order"] for r in records
    )
    # Independent histories may legitimately hit the shared public template.
    checks["first_identity_cold"] = all(
        r["cached_tokens"] == 0 for r in records if r.get("first_identity_request")
    )
    summary = {
        "precision": o.precision,
        "stage": o.stage,
        "run_id": o.run_id,
        "planned": len(rows),
        "completed": len(records),
        "failed": len(failures),
        "unfinished": len(rows) - len(records) - len(failures),
        "checks": checks,
        "valid_for_comparison": all(checks.values()),
        "wall_seconds": wall,
        "output_tokens_per_second": sum(r["output_tokens"] for r in records) / wall,
        "requests_per_second": len(records) / wall,
        "actual_input_tokens": sum(r["input_tokens"] for r in records),
        "actual_output_tokens": sum(r["output_tokens"] for r in records),
        "cached_tokens_sum": cached_sum,
        "input_token_hit_fraction": cached_sum / sum(r["input_tokens"] for r in records)
        if cached_complete
        else None,
        "groups": {k: metric_group(v) for k, v in groups.items()},
        "rank_completed": rank_counts,
        "rank_success_counter": rank_success,
        "concurrency": timeline.stats(),
        "logical_idle_skipped_s": skipped,
        "idle_skip_events": len(skips),
        "client_cpu_seconds": cpu_seconds,
        "client_cpu_cores_mean": cpu_seconds / wall,
        "client_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_engine_telemetry": replay.telemetry_summary(samples),
        "telemetry_errors": errors,
        "counter_delta_by_engine": delta,
        "preemptions": counter(delta, "vllm:num_preemptions_total"),
        "local_cached_tokens": counter(delta, "vllm:prefix_cache_hits_total"),
        "external_cached_tokens": counter(delta, "vllm:external_prefix_cache_hits_total"),
        "memcache_io": {
            name: counter(delta, name) for name in replay128.COUNTERS if "ascend_store" in name
        },
        "speculative_acceptance_rate": counter(delta, "vllm:spec_decode_num_accepted_tokens_total")
        / max(1, counter(delta, "vllm:spec_decode_num_draft_tokens_total")),
        "payload_sha256": config["payload_sha256"],
        "run_plan_sha256": config["run_plan_sha256"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("replay integrity checks failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("replay",), default="replay")
    p.add_argument("--precision", choices=("int8", "bf16"), required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--driver-lock", required=True)
    p.add_argument("--capacity-host", required=True)
    p.add_argument("--capacity-port", type=int, default=17000)
    p.add_argument("--dataset", type=Path, required=True)
    options = p.parse_args()
    DATA = options.dataset.resolve(strict=True)
    asyncio.run(run(options))
