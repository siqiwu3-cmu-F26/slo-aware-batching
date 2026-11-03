"""Optional real-engine loading shared by the smoke driver and benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import Any, Callable

from interfaces import Engine
from .mock_engine import MockEngine


@dataclass(frozen=True)
class EngineConfig:
    engine: str = "mock"
    model_name: str = "gpt2"
    backend: str = "cpu"
    model_seed: int = 0
    prompt_seed: int = 0
    prompt_file: str | None = None
    warmup_steps: int = 20

    def __post_init__(self) -> None:
        if self.engine not in {"mock", "minitorch"}:
            raise ValueError("engine must be mock or minitorch")
        if self.backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=("mock", "minitorch"), default="mock")
    parser.add_argument("--model", dest="model_name", default="gpt2",
                        help="tiny (random smoke model), gpt2, gpt2-medium, or local checkpoint")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu",
                        help="cuda selects the team's fused CudaKernelOps backend")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--prompts", dest="prompt_file", help="JSON mapping request ID to prompt token IDs")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--clock", dest="clock_mode", choices=("virtual", "wall"),
                        help="default: virtual for mock, wall for MiniTorch")


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(**{name: getattr(args, name) for name in EngineConfig.__dataclass_fields__})


def runtime_metadata() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "numba", "torch", "transformers", "pycuda"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "host": platform.node(), "packages": packages}


@lru_cache(maxsize=1)
def _load_model(model_name: str, backend_name: str, model_seed: int) -> tuple[Any, dict[str, object],
                                                                              Callable[[], None] | None]:
    # Benchmark matrices hold one model per output directory; reuse it across rows
    # instead of reloading weights and recomputing the checksum for every run.
    try:
        import numpy as np
        from minitorch import FastOps, TensorBackend
        from model import GPT2, GPT2Config, hf_state_dict, load_hf_weights
        from .minitorch_engine import weights_sha256
    except ImportError as error:
        raise RuntimeError("MiniTorch dependencies missing; install requirements-engine-cpu.txt "
                           "(and requirements-engine.txt for pretrained weights)") from error
    metadata = runtime_metadata()
    from numba import get_num_threads
    metadata["numba_threads"] = get_num_threads()
    metadata.update(model_name=model_name, backend=backend_name, model_seed=model_seed)
    synchronize = None
    ops = FastOps
    if backend_name == "cuda":
        try:
            from minitorch.cuda_kernel_ops import CudaKernelOps
            import pycuda.driver as cuda
        except (ImportError, OSError) as error:
            raise RuntimeError("CUDA backend requires pycuda, torch, and built libraries; "
                               "run kernels/compile_cuda.sh on a CUDA machine") from error
        ops = CudaKernelOps
        synchronize = cuda.Context.synchronize
        device = cuda.Context.get_device()
        metadata.update(gpu=device.name(), cuda_driver=cuda.get_driver_version())
    backend = TensorBackend(ops)
    # Assumption: construction is reproducible and does not perturb workload RNGs.
    state = np.random.get_state()
    try:
        np.random.seed(model_seed)
        if model_name == "tiny":
            model = GPT2(GPT2Config.tiny(), backend)
            metadata["weights"] = "random_smoke_only"
        else:
            model_config, weights = hf_state_dict(model_name)
            model = GPT2(model_config, backend)
            load_hf_weights(model, weights)
            metadata["weights"] = "pretrained"
    finally:
        np.random.set_state(state)
    metadata["weights_sha256"] = weights_sha256(model)
    return model, metadata, synchronize


def create_engine(config: EngineConfig, max_slots: int, a: float = 0.01,
                  b: float = 0.001) -> Engine:
    if config.engine == "mock":
        return MockEngine(max_slots, a, b)
    # Keep all model dependencies optional for scheduler/mock-only users.
    model, metadata, synchronize = _load_model(config.model_name, config.backend, config.model_seed)
    from .minitorch_engine import MiniTorchEngine
    prompts = None
    if config.prompt_file is not None:
        prompts = json.loads(Path(config.prompt_file).read_text(encoding="utf-8"))
        if not isinstance(prompts, dict) or any(not isinstance(v, list) for v in prompts.values()):
            raise ValueError("prompt file must map request IDs to lists of token IDs")
    # Each engine gets a fresh KV cache and warmup; only the weights are shared.
    engine = MiniTorchEngine(model, max_slots, prompts, config.prompt_seed, synchronize, dict(metadata))
    engine.warmup(config.warmup_steps)
    return engine
