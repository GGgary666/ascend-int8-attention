"""Opt-in profiler annotations; no synchronization or model/scheduler changes."""
from functools import wraps
import json
import os
from pathlib import Path
import time

_installed = False


def describe_batch(runner, scheduled):
    new = {r.req_id: r for r in scheduled.scheduled_new_reqs}
    cached = scheduled.scheduled_cached_reqs
    computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
    rows = []
    for request_id, count in scheduled.num_scheduled_tokens.items():
        state = runner.requests.get(request_id, new.get(request_id))
        prompt = len(state.prompt_token_ids) if state is not None and state.prompt_token_ids is not None else None
        position = computed.get(request_id, getattr(state, "num_computed_tokens", None))
        prefill = min(count, max(0, prompt - position)) if prompt is not None and position is not None else None
        rows.append(dict(scheduled=count, prompt=prompt, computed=position, prefill=prefill))
    return dict(requests=len(rows), tokens=scheduled.total_num_scheduled_tokens,
                prefill_tokens=sum(r["prefill"] or 0 for r in rows),
                unknown_rows=sum(r["prefill"] is None for r in rows), rows=rows)


def wrap_runner(original, phase, directory, record_function):
    @wraps(original)
    def observed(self, *args, **kwargs):
        worker = getattr(self, "_sharedkv_diag_worker", None)
        profiler = getattr(worker, "profiler", None)
        if worker is None or not getattr(worker, '_sharedkv_profile_selected', worker.rank == 0) or profiler is None or not profiler.is_running:
            return original(self, *args, **kwargs)
        if phase == "target_forward":
            scheduled = args[0] if args else kwargs["scheduler_output"]
            info = describe_batch(self, scheduled)
            self._sharedkv_diag_step = getattr(self, "_sharedkv_diag_step", 0) + 1
            self._sharedkv_diag_info = info
            self._sharedkv_diag_execution = None
        else:
            info = getattr(self, "_sharedkv_diag_info", {})
        step = getattr(self, "_sharedkv_diag_step", 0)
        marker = f"sharedkv_step|{step}|{phase}|b{info.get('requests')}|t{info.get('tokens')}|p{info.get('prefill_tokens')}"
        started = time.time_ns()
        with record_function(marker):
            result = original(self, *args, **kwargs)
        ended = time.time_ns()
        record = dict(step=step, phase=phase, marker=marker, start_ns=started, end_ns=ended,
                      rank=worker.rank, local_rank=getattr(worker, 'local_rank', None),
                      host_submission_ms=(ended-started)/1e6, batch=info,
                      execution=getattr(self, "_sharedkv_diag_execution", None),
                      note="Host submission interval, not synchronized device latency")
        with (directory / f"steps_rank{worker.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return result
    return observed


def install(directory):
    global _installed
    if _installed:
        return
    import torch
    import vllm_ascend
    vllm_ascend._ensure_global_patch()
    from vllm_ascend.worker.worker import NPUWorker
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original_execute = NPUWorker.execute_model
    selected_ranks = os.environ.get('SHAREDKV_PROFILE_TP_RANKS')
    if selected_ranks:
        selected_ranks = set(map(int, selected_ranks.split(',')))
        original_profile = NPUWorker.profile

        @wraps(original_profile)
        def profile(self, *args, **kwargs):
            from vllm.distributed import get_tensor_model_parallel_rank
            self._sharedkv_profile_selected = get_tensor_model_parallel_rank() in selected_ranks
            if self._sharedkv_profile_selected:
                return original_profile(self, *args, **kwargs)

        NPUWorker.profile = profile

    @wraps(original_execute)
    def worker_execute(self, *args, **kwargs):
        self.model_runner._sharedkv_diag_worker = self
        return original_execute(self, *args, **kwargs)

    original_determine = NPUModelRunner._determine_batch_execution_and_padding

    @wraps(original_determine)
    def determine(self, *args, **kwargs):
        result = original_determine(self, *args, **kwargs)
        worker = getattr(self, "_sharedkv_diag_worker", None)
        profiler = getattr(worker, "profiler", None)
        if worker is not None and getattr(worker, '_sharedkv_profile_selected', worker.rank == 0) and profiler is not None and profiler.is_running:
            descriptor = result[1]
            self._sharedkv_diag_execution = dict(graph_mode=str(result[0]),
                padded_tokens=descriptor.num_tokens, padded_requests=descriptor.num_reqs,
                max_query_length=kwargs.get("max_num_scheduled_tokens"))
        return result

    NPUWorker.execute_model = worker_execute
    NPUModelRunner._determine_batch_execution_and_padding = determine
    for method, phase in (("execute_model", "target_forward"), ("sample_tokens", "sample_and_draft")):
        setattr(NPUModelRunner, method, wrap_runner(getattr(NPUModelRunner, method), phase,
                                                  directory, torch.profiler.record_function))
    _installed = True
