"""Temporary event-based self-test metrics, not the team's benchmark harness."""

from dataclasses import asdict, dataclass
from typing import Iterable

from interfaces import Event, EventKind, Request


@dataclass(frozen=True)
class Metrics:
    ttft_p50: float | None
    ttft_p99: float | None
    tpot_p50: float | None
    tpot_p99: float | None
    requests_per_second: float
    tokens_per_second: float
    goodput: float
    rejection_rate: float
    completed_requests: int
    rejected_requests: int
    duration: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute_metrics(requests: Iterable[Request], events: Iterable[Event]) -> Metrics:
    requests = {request.id: request for request in requests}
    events = list(events)
    arrivals = {event.request_id: event.timestamp for event in events
                if event.kind == EventKind.ARRIVAL}
    first = {event.request_id: event.timestamp for event in events
             if event.kind == EventKind.FIRST_TOKEN}
    finishes = {event.request_id: event.timestamp for event in events
                if event.kind == EventKind.FINISH}
    rejected = {event.request_id for event in events if event.kind == EventKind.REJECT}
    tokens: dict[str, list[float]] = {}
    for event in events:
        if event.kind == EventKind.TOKEN:
            tokens.setdefault(event.request_id, []).append(event.timestamp)
    # Assumption: exclude initial idle time; include queueing, execution, and idle
    # gaps between the first arrival and final event. Empty/zero windows yield 0/s.
    duration = max((event.timestamp for event in events), default=0) - min(arrivals.values(), default=0)
    ttfts, tpots = [], []
    good = 0
    for rid in finishes:
        ttft = first[rid] - arrivals[rid]
        times = tokens[rid]
        # Assumption: percentiles and goodput use per-request mean TPOT, excluding
        # TTFT. Single-output requests have TPOT=0 (no inter-token gap).
        tpot = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 0.0
        ttfts.append(ttft)
        tpots.append(tpot)
        request = requests[rid]
        # Assumption: absent SLOs impose no constraint; equality meets the SLO.
        good += ((request.ttft_slo is None or ttft <= request.ttft_slo)
                 and (request.tpot_slo is None or tpot <= request.tpot_slo))
    def rate(count: int) -> float:
        return count / duration if duration > 0 else 0.0

    return Metrics(
        _percentile(ttfts, 50), _percentile(ttfts, 99),
        _percentile(tpots, 50), _percentile(tpots, 99),
        rate(len(finishes)), rate(sum(map(len, tokens.values()))), rate(good),
        len(rejected) / len(arrivals) if arrivals else 0.0,
        len(finishes), len(rejected), duration,
    )
