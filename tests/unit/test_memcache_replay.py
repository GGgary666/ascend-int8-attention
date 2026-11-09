"""Verify the network sender obeys the global FIFO, including across DPs."""

import asyncio
import collections
import json
from types import SimpleNamespace
import pytest
import socket

pytest.importorskip("aiohttp")
pytest.importorskip("numpy")
from aiohttp import web
from benchmarks.memcache import replay as client


@pytest.mark.parametrize("fail_parent", [False, True])
@pytest.mark.parametrize("external_hit", [False, True])
def test_global_head_blocks_other_dp_and_failed_reference_stops_trial(
    tmp_path, monkeypatch, fail_parent, external_hit
):
    (tmp_path / "run_plan.json").write_text(
        json.dumps(dict(max_client_inflight_per_dp=16, max_global_client_inflight=32))
    )
    (tmp_path / "payloads.jsonl.gz").write_bytes(
        b"mock payload identity; no real dataset is submitted"
    )
    monkeypatch.setattr(client, "DATA", tmp_path)

    async def exercise():
        counts = {rank: {name: 0 for name in client.replay128.COUNTERS} for rank in ("0", "1")}
        active = collections.Counter()
        sent = []

        async def metrics(request):
            output = []
            for rank in ("0", "1"):
                for name, value in counts[rank].items():
                    labels = f'engine="{rank}"'
                    if name == "vllm:request_success_total":
                        labels += ',finished_reason="length"'
                    output.append(f"{name}{{{labels}}} {value}")
                for name, value in [
                    ("num_requests_running", active[rank]),
                    ("num_requests_waiting", 0),
                    ("kv_cache_usage_perc", 0),
                ]:
                    output.append(f'vllm:{name}{{engine="{rank}"}} {value}')
            return web.Response(text="\n".join(output) + "\n")

        async def complete(request):
            body = await request.json()
            rank = request.headers["X-data-parallel-rank"]
            rid = body["request_id"]
            sent.append(rid)
            if rid == "parent" and fail_parent:
                return web.Response(status=500, text="intentional mock failure")
            active[rank] += 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await asyncio.sleep(0.10 if rid == "parent" else 0.01)
            event = {
                "id": "cmpl-" + rid,
                "choices": [{"prompt_token_ids": body["prompt"], "token_ids": [10]}],
            }
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            await asyncio.sleep(0.01)
            event["choices"] = [{"token_ids": [11], "finish_reason": "length"}]
            await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            cached = 3 if external_hit and rid == "child" else 0
            event["choices"] = []
            event["usage"] = {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": cached},
            }
            await response.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
            await response.write_eof()
            active[rank] -= 1
            for name, n in [
                ("vllm:request_success_total", 1),
                ("vllm:prompt_tokens_total", 3),
                ("vllm:prefix_cache_queries_total", 3),
                ("vllm:generation_tokens_total", 2),
            ]:
                counts[rank][name] += n
            counts[rank]["vllm:external_prefix_cache_hits_total"] += cached
            if cached:
                counts[rank]["vllm:ascend_store_get_success_bytes_total"] += 64
            return response

        app = web.Application()
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/completions", complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, socket.gethostbyname("localhost"), 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        items = []
        for order, (rid, dp, dporder, arrival, parent, gap) in enumerate(
            [
                ("parent", 0, 0, 0, None, 0),
                ("child", 0, 1, 0, "parent", 0.05),
                ("other", 1, 0, 0.03, None, 0),
                ("late", 1, 1, 10, None, 0),
            ]
        ):
            m = dict(
                id=rid,
                sample_id=rid,
                benchmark_id=rid,
                dp_rank=dp,
                dp_order=dporder,
                replay_order=order,
                availability_parent_id=parent,
                busy_arrival_offset_s=arrival,
                busy_parent_gap_s=gap,
                conversation_parent_request_id=parent,
                conversation_think_time_s=gap,
                input_tokens=3,
                output_tokens=2,
                replay_prefix_tokens=0,
                is_continuation=bool(parent),
                is_history_followup=bool(parent),
                is_long_history_followup=False,
                kv_reuse_planned=False,
                prefix_reference_id=None,
            )
            items.append(
                {
                    "metadata": m,
                    "headers": {"X-data-parallel-rank": str(dp)},
                    "request": {"request_id": rid, "prompt": [1, 2, 3], "max_tokens": 2},
                }
            )
        monkeypatch.setattr(client, "load", lambda *args: items)
        out = tmp_path / "run"
        options = SimpleNamespace(
            stage="replay",
            precision="int8",
            run_id="test",
            output=str(out),
            base=f"http://localhost:{port}",
        )
        try:
            if fail_parent:
                with pytest.raises(RuntimeError, match="history reference failed"):
                    await client.run(options)
                assert sent == ["parent"]
                assert len((out / "failures.jsonl").read_text().splitlines()) == 1
            else:
                await client.run(options)
                s = json.loads((out / "summary.json").read_text())
                rs = {
                    r["id"]: r
                    for r in map(json.loads, (out / "requests.jsonl").read_text().splitlines())
                }
                assert s["valid_for_comparison"] and s["completed"] == 4
                assert s["local_cached_tokens"] == 0 and s["external_cached_tokens"] == (
                    3 if external_hit else 0
                )
                assert all(
                    row["finish_reason"] == "length" and row["generated_token_ids"] == [10, 11]
                    for row in rs.values()
                )
                assert sent == ["parent", "child", "other", "late"]
                assert rs["other"]["actual_submit_s"] > rs["parent"]["end_s"]
                assert (
                    rs["child"]["logical_submitted_s"] >= rs["parent"]["logical_finished_s"] + 0.05
                )
                assert s["logical_idle_skipped_s"] > 9 and s["wall_seconds"] < 2
        finally:
            await runner.cleanup()

    asyncio.run(exercise())
