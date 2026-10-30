"""Event-based per-request and aggregate serving metrics."""

from dataclasses import asdict, dataclass
from typing import Iterable

from interfaces import Event, EventKind, Request


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    status: str
    arrival_time: float
    first_token_time: float | None
    finish_time: float | None
    output_tokens: int
    ttft: float | None
    mean_tpot: float | None
    e2e_latency: float | None
    ttft_slo: float | None
    tpot_slo: float | None
    meets_ttft_slo: bool | None
    meets_tpot_slo: bool | None
    meets_slo: bool

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class Metrics:
    ttft_p50: float | None
    ttft_p99: float | None
    # Per-request mean TPOT percentiles, retained for baseline compatibility.
    tpot_p50: float | None
    tpot_p99: float | None
    # Percentiles over every inter-token interval across completed requests.
    token_tpot_p50: float | None
    token_tpot_p99: float | None
    e2e_p50: float | None
    e2e_p99: float | None
    requests_per_second: float
    tokens_per_second: float
    goodput: float
    slo_attainment_rate: float
    completion_rate: float
    rejection_rate: float
    completed_requests: int
    rejected_requests: int
    incomplete_requests: int
    total_requests: int
    output_tokens: int
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


def compute_request_metrics(
    requests: Iterable[Request], events: Iterable[Event]
) -> list[RequestMetrics]:
    """Compute one row per request from its lifecycle events."""
    requests = list(requests)
    events = list(events)
    first = {
        event.request_id: event.timestamp
        for event in events
        if event.kind == EventKind.FIRST_TOKEN
    }
    finishes = {
        event.request_id: event.timestamp
        for event in events
        if event.kind == EventKind.FINISH
    }
    rejected = {
        event.request_id for event in events if event.kind == EventKind.REJECT
    }
    tokens: dict[str, list[float]] = {}
    for event in events:
        if event.kind == EventKind.TOKEN:
            tokens.setdefault(event.request_id, []).append(event.timestamp)

    rows = []
    for request in requests:
        rid = request.id
        token_times = tokens.get(rid, [])
        if rid in rejected:
            status = "rejected"
        elif rid in finishes:
            status = "completed"
        else:
            status = "incomplete"

        first_time = first.get(rid)
        finish_time = finishes.get(rid)
        ttft = first_time - request.arrival_time if first_time is not None else None
        mean_tpot = None
        if token_times:
            # A single-token response has no inter-token gap and uses TPOT=0.
            mean_tpot = (
                (token_times[-1] - token_times[0]) / (len(token_times) - 1)
                if len(token_times) > 1 else 0.0
            )
        e2e = finish_time - request.arrival_time if finish_time is not None else None
        meets_ttft = (
            None if request.ttft_slo is None or ttft is None
            else ttft <= request.ttft_slo
        )
        meets_tpot = (
            None if request.tpot_slo is None or mean_tpot is None
            else mean_tpot <= request.tpot_slo
        )
        # Missing SLOs impose no constraint. Rejected/incomplete requests fail.
        meets_slo = (
            status == "completed"
            and (request.ttft_slo is None or meets_ttft is True)
            and (request.tpot_slo is None or meets_tpot is True)
        )
        rows.append(RequestMetrics(
            request_id=rid,
            status=status,
            arrival_time=request.arrival_time,
            first_token_time=first_time,
            finish_time=finish_time,
            output_tokens=len(token_times),
            ttft=ttft,
            mean_tpot=mean_tpot,
            e2e_latency=e2e,
            ttft_slo=request.ttft_slo,
            tpot_slo=request.tpot_slo,
            meets_ttft_slo=meets_ttft,
            meets_tpot_slo=meets_tpot,
            meets_slo=meets_slo,
        ))
    return rows


def compute_metrics(requests: Iterable[Request], events: Iterable[Event]) -> Metrics:
    requests = list(requests)
    events = list(events)
    per_request = compute_request_metrics(requests, events)
    arrivals = [
        event.timestamp for event in events if event.kind == EventKind.ARRIVAL
    ]
    tokens: dict[str, list[float]] = {}
    for event in events:
        if event.kind == EventKind.TOKEN:
            tokens.setdefault(event.request_id, []).append(event.timestamp)
    # Exclude initial idle time; include queueing and execution after first arrival.
    duration = (
        max((event.timestamp for event in events), default=0) - min(arrivals)
        if arrivals else 0.0
    )
    completed = [row for row in per_request if row.status == "completed"]
    rejected = [row for row in per_request if row.status == "rejected"]
    incomplete = [row for row in per_request if row.status == "incomplete"]
    ttfts = [row.ttft for row in completed if row.ttft is not None]
    tpots = [row.mean_tpot for row in completed if row.mean_tpot is not None]
    e2es = [row.e2e_latency for row in completed if row.e2e_latency is not None]
    token_tpots = [
        later - earlier
        for row in completed
        for earlier, later in zip(
            tokens.get(row.request_id, []), tokens.get(row.request_id, [])[1:]
        )
    ]
    good = sum(row.meets_slo for row in completed)
    output_tokens = sum(row.output_tokens for row in per_request)
    total = len(per_request)

    def rate(count: int) -> float:
        return count / duration if duration > 0 else 0.0

    def fraction(count: int) -> float:
        return count / total if total else 0.0

    return Metrics(
        ttft_p50=_percentile(ttfts, 50),
        ttft_p99=_percentile(ttfts, 99),
        tpot_p50=_percentile(tpots, 50),
        tpot_p99=_percentile(tpots, 99),
        token_tpot_p50=_percentile(token_tpots, 50),
        token_tpot_p99=_percentile(token_tpots, 99),
        e2e_p50=_percentile(e2es, 50),
        e2e_p99=_percentile(e2es, 99),
        requests_per_second=rate(len(completed)),
        tokens_per_second=rate(output_tokens),
        goodput=rate(good),
        slo_attainment_rate=fraction(good),
        completion_rate=fraction(len(completed)),
        rejection_rate=fraction(len(rejected)),
        completed_requests=len(completed),
        rejected_requests=len(rejected),
        incomplete_requests=len(incomplete),
        total_requests=total,
        output_tokens=output_tokens,
        duration=duration,
    )
