"""Scheduler adapter around the team's GPT2/KVCache, without new model kernels.

Unequal input spans are grouped by length and executed in separate forwards.
This is a correctness/integration path, not a fused Sarathi mixed-batch kernel.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import random
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np

from interfaces import BatchPlan, Request, StepResult
from model import GPT2, KVCache


def weights_sha256(model: GPT2) -> str:
    fingerprint = hashlib.sha256()
    for name, parameter in model.named_parameters():
        fingerprint.update(name.encode())
        fingerprint.update(parameter.value.to_numpy().tobytes())
    return fingerprint.hexdigest()


@dataclass
class _RequestProgress:
    prompt: list[int]
    output_len: int
    prefilled: int = 0
    generated: list[int] = field(default_factory=list)
    slot: int | None = None


class MiniTorchEngine:
    def __init__(self, model: GPT2, max_slots: int,
                 prompts: Mapping[str, Sequence[int]] | None = None,
                 prompt_seed: int = 0,
                 synchronize: Callable[[], None] | None = None,
                 metadata: Mapping[str, object] | None = None) -> None:
        self.model = model
        self.cache = KVCache(model.config, max_slots)
        self.max_slots = max_slots
        self._prompts = prompts
        self.prompt_seed = prompt_seed
        self._synchronize = synchronize or (lambda: None)
        self._requests: dict[str, _RequestProgress] = {}
        # Assumption: this adapter is inference-only. Parameter.update() in the
        # current runtime enables gradients, so detach the loaded/random weights.
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.value = parameter.value.detach()
        metadata = dict(metadata or {})
        # The factory supplies a cached checksum; hashing large models is slow.
        if "weights_sha256" not in metadata:
            metadata["weights_sha256"] = weights_sha256(self.model)
        self.metadata: dict[str, object] = {
            "engine": "minitorch", "execution_mode": "grouped_by_input_length",
            "model_config": asdict(model.config), "precision": "float32",
            "prompt_source": "provided_token_ids" if prompts is not None else "synthetic_length_replay",
            "prompt_seed": prompt_seed, "stopping_rule": "fixed_output_len_ignore_eos",
            "kv_storage": "host_numpy", "kv_cache_bytes": self.cache.num_bytes,
            "warmup_steps": 0, **metadata,
        }
        self.forward_calls = 0

    @property
    def prompt_tokens(self) -> dict[str, list[int]]:
        return {rid: list(request.prompt) for rid, request in self._requests.items()}

    @property
    def output_tokens(self) -> dict[str, list[int]]:
        return {rid: list(request.generated) for rid, request in self._requests.items()}

    def register(self, request: Request) -> None:
        if request.id in self._requests:
            raise ValueError(f"duplicate engine request: {request.id}")
        # Assumption: fixed output lengths make baseline comparisons deterministic;
        # only the engine sees this stopping length, never the scheduler.
        if request.prompt_len + request.output_len > self.model.config.n_positions:
            raise ValueError(f"request {request.id} exceeds model context length")
        if self._prompts is None:
            # Stable per-ID seed: unaffected by admission order, scheduler, or
            # Python hash randomization. These are length-replay inputs, not text.
            digest = hashlib.sha256(f"{self.prompt_seed}:{request.id}".encode()).digest()
            rng = random.Random(int.from_bytes(digest, "big"))
            prompt = [rng.randrange(self.model.config.n_vocab) for _ in range(request.prompt_len)]
        else:
            if request.id not in self._prompts:
                raise ValueError(f"missing prompt token IDs for {request.id}")
            prompt = list(self._prompts[request.id])
        if len(prompt) != request.prompt_len:
            raise ValueError(f"prompt length mismatch for {request.id}")
        if any(type(token) is not int or not 0 <= token < self.model.config.n_vocab for token in prompt):
            raise ValueError(f"invalid prompt token ID for {request.id}")
        self._requests[request.id] = _RequestProgress(prompt, request.output_len)

    def warmup(self, steps: int = 20) -> None:
        """Warm the existing model path, then return all slots to a clean state."""
        if type(steps) is not int or steps < 0:
            raise ValueError("warmup steps must be a nonnegative integer")
        if self._requests:
            raise ValueError("warmup must happen before request registration")
        slots = [self.cache.allocate() for _ in range(self.max_slots)]
        try:
            for _ in range(steps):
                if self.cache.length(slots[0]) >= self.model.config.n_positions:
                    for slot in slots:
                        self.cache.free(slot)
                    slots = [self.cache.allocate() for _ in range(self.max_slots)]
                starts = [self.cache.length(slot) for slot in slots]
                count = min(2, self.model.config.n_positions) if starts[0] == 0 else 1
                self.model(np.zeros((self.max_slots, count), dtype=np.int64), starts, self.cache, slots).to_numpy()
            self._synchronize()
        finally:
            for slot in slots:
                self.cache.free(slot)
        self.metadata["warmup_steps"] = steps

    def _validate(self, plan: BatchPlan) -> None:
        ids = [rid for rid, _, _ in plan.items]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("plan must be nonempty with one span per request")
        if plan.token_budget is not None and plan.total_tokens > plan.token_budget:
            raise ValueError("plan exceeds token budget")
        new_slots = 0
        for rid, start, count in plan.items:
            if rid not in self._requests:
                raise ValueError(f"unregistered request: {rid}")
            request = self._requests[rid]
            if len(request.generated) == request.output_len:
                raise ValueError(f"request already finished: {rid}")
            if type(start) is not int or type(count) is not int or count < 1:
                raise ValueError("positions/counts must be integers and counts positive")
            if request.prefilled < len(request.prompt):
                if start != request.prefilled or count > len(request.prompt) - request.prefilled:
                    raise ValueError("invalid prefill span")
            elif start != len(request.prompt) + len(request.generated) - 1 or count != 1:
                raise ValueError("invalid decode span")
            if request.slot is None:
                new_slots += 1
            elif self.cache.length(request.slot) != start:
                raise ValueError("engine progress and KV cache disagree")
        if new_slots > self.cache.free_slots:
            raise ValueError("plan exceeds resident request-slot capacity")

    def execute(self, plan: BatchPlan) -> StepResult:
        self._synchronize()
        started = perf_counter()
        self._validate(plan)
        # Snapshot progress so a failure in a later forward cannot leave earlier
        # groups half-applied; cache contents past lengths[slot] are never read.
        saved = {rid: (self._requests[rid].prefilled, len(self._requests[rid].generated),
                       self._requests[rid].slot) for rid, _, _ in plan.items}
        saved_lengths = self.cache.lengths.copy()
        groups: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
        for rid, start, count in plan.items:
            request = self._requests[rid]
            if request.slot is None:
                request.slot = self.cache.allocate()
            groups[count].append((rid, start, count))
        emitted: set[str] = set()
        finished: set[str] = set()
        try:
            self._run_groups(groups, emitted, finished)
        except Exception:
            for rid, (prefilled, generated, slot) in saved.items():
                request = self._requests[rid]
                if slot is None and request.slot is not None:
                    self.cache.free(request.slot)
                request.prefilled, request.slot = prefilled, slot
                del request.generated[generated:]
            self.cache.lengths[:] = saved_lengths
            raise
        # Release slots only after every forward succeeded.
        for rid in finished:
            request = self._requests[rid]
            self.cache.free(request.slot)
            request.slot = None
        self._synchronize()
        # Assumption: all outputs in a logical plan are timestamped at its end,
        # including when the temporary implementation uses several forwards.
        duration = perf_counter() - started
        ids = [rid for rid, _, _ in plan.items]
        return StepResult(duration, tuple(rid for rid in ids if rid in emitted),
                          tuple(rid for rid in ids if rid in finished))

    def _run_groups(self, groups: Mapping[int, list[tuple[str, int, int]]],
                    emitted: set[str], finished: set[str]) -> None:
        for count, items in groups.items():
            inputs, starts, slots = [], [], []
            for rid, start, _ in items:
                request = self._requests[rid]
                inputs.append(request.prompt[start:start + count]
                              if request.prefilled < len(request.prompt)
                              else [request.generated[-1]])
                starts.append(start)
                slots.append(request.slot)
            # Assumption: equal query lengths use the team's batched forward;
            # different lengths use multiple forwards with no fabricated padding.
            logits = self.model(np.asarray(inputs, dtype=np.int64), starts, self.cache, slots).to_numpy()
            self.forward_calls += 1
            for row, (rid, _, _) in enumerate(items):
                request = self._requests[rid]
                if request.prefilled < len(request.prompt):
                    request.prefilled += count
                if request.prefilled == len(request.prompt):
                    request.generated.append(int(np.argmax(logits[row, -1])))
                    emitted.add(rid)
                    if len(request.generated) == request.output_len:
                        finished.add(rid)
