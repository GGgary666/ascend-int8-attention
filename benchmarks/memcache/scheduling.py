"""Strict global source order; no network calls and no dense-time mode."""


def ready_at(row, completed_at):
    parent = row["availability_parent_id"]
    if parent and parent not in completed_at:
        return None
    return max(
        row["busy_arrival_offset_s"],
        completed_at[parent] + row["busy_parent_gap_s"] if parent else 0.0,
    )


class OrderedDispatcher:
    """Use one global FIFO across both DPs; never overtake a blocked head."""

    def __init__(self, rows, max_inflight_per_dp=16, max_global_inflight=32):
        self.pending = sorted(rows, key=lambda r: r["replay_order"])
        self.dp_limit = max_inflight_per_dp
        self.global_limit = max_global_inflight

    def take_ready(self, now, completed_at, inflight_by_dp):
        counts = {d: inflight_by_dp.get(d, 0) for d in (0, 1)}
        ready = []
        while self.pending:
            row = self.pending[0]
            dp = row["dp_rank"]
            when = ready_at(row, completed_at)
            if (
                when is None
                or when > now + 1e-8
                or counts[dp] >= self.dp_limit
                or sum(counts.values()) >= self.global_limit
            ):
                break
            ready.append(self.pending.pop(0))
            counts[dp] += 1
        return ready

    def next_release(self, completed_at, inflight_by_dp):
        if not self.pending:
            return None
        head = self.pending[0]
        if (
            inflight_by_dp.get(head["dp_rank"], 0) >= self.dp_limit
            or sum(inflight_by_dp.values()) >= self.global_limit
        ):
            return None
        return ready_at(head, completed_at)

    def empty(self):
        return not self.pending


def may_skip_idle(*, client_inflight, dp_running, dp_waiting):
    """Unknown server state is not proof of idleness. Apply only to isolated runs."""
    return client_inflight == 0 and all(
        dp_running.get(d) == 0 and dp_waiting.get(d) == 0 for d in (0, 1)
    )
