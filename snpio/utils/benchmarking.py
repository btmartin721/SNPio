import os
import time
import tracemalloc
from contextlib import ContextDecorator
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd


@dataclass
class ResourceMetrics:
    """Stores memory and execution time for one benchmark run."""

    memory_footprint_mib: float
    execution_time_s: float


def _subprocess_benchmark(conn, func, args, kwargs):
    import sys
    import time
    import traceback
    import tracemalloc

    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        tracemalloc.start()
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        conn.send(("ok", peak / (1024**2), end - start))
    except Exception:
        tb = traceback.format_exc()
        conn.send(("error", tb))
    finally:
        conn.close()


class Benchmark:
    """Benchmark class for measuring performance of functions and code blocks."""

    global_resource_data: Dict[str, List[ResourceMetrics]] = {}

    @staticmethod
    def measure_once(func: Callable[..., Any], *args, **kwargs) -> ResourceMetrics:
        """Measure execution time and memory for a single function call."""
        tracemalloc.start()
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ResourceMetrics(peak / (1024**2), end - start)

    @staticmethod
    def run_repeated(
        func: Callable[..., Any], repeats: int, *args, **kwargs
    ) -> List[ResourceMetrics]:
        """Run a function multiple times and collect performance metrics."""
        return [Benchmark.measure_once(func, *args, **kwargs) for _ in range(repeats)]

    @staticmethod
    def run_repeated_subprocess(
        name: str,
        func: Callable[..., Any],
        repeats: int = 5,
        *args,
        **kwargs,
    ) -> List[ResourceMetrics]:
        """Benchmark function in isolated subprocesses."""
        from tqdm import tqdm

        metrics = []
        for i in tqdm(range(repeats), desc=f"[Subprocess] {name}", unit="rep"):
            parent_conn, child_conn = Pipe()
            p = Process(
                target=_subprocess_benchmark, args=(child_conn, func, args, kwargs)
            )
            p.start()
            msg = parent_conn.recv()
            p.join()

            if msg[0] == "ok":
                _, mem_mib, exec_time = msg
                metrics.append(ResourceMetrics(mem_mib, exec_time))
            elif msg[0] == "error":
                raise RuntimeError(f"[Subprocess error on iteration {i}] {msg[1]}")
            else:
                raise ValueError(f"Unexpected message format: {msg}")

        return metrics

    @staticmethod
    def measure_block(obj: Any | None, name: str) -> ContextDecorator:
        """Context manager for measuring performance of code blocks."""

        class _Measurer(ContextDecorator):
            def __enter__(_self):
                _self._start = time.perf_counter()
                tracemalloc.start()
                return _self

            def __exit__(_self, exc_type, exc_val, exc_tb):
                exec_time = time.perf_counter() - _self._start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                metrics = ResourceMetrics(peak / (1024**2), exec_time)

                if obj is None:
                    Benchmark.global_resource_data.setdefault(name, []).append(metrics)
                else:
                    if not hasattr(obj, "resource_data"):
                        obj.resource_data = {}
                    obj.resource_data.setdefault(name, []).append(metrics)

                return False  # propagate exceptions

        return _Measurer()

    @classmethod
    def collect_all(cls, *objs: Any) -> Dict[str, List[ResourceMetrics]]:
        """Merge global and object-specific resource metrics."""
        combined: Dict[str, List[ResourceMetrics]] = {}
        for method, runs in cls.global_resource_data.items():
            combined.setdefault(method, []).extend(runs)
        for obj in objs:
            for method, runs in getattr(obj, "resource_data", {}).items():
                combined.setdefault(method, []).extend(runs)
        return combined

    @classmethod
    def save_performance(
        cls,
        resource_data: Dict[str, List[ResourceMetrics]] | None = None,
        objs: List[Any] | None = None,
        save_dir: Path | None = None,
        outfile_prefix: str = "",
    ) -> Tuple[pd.DataFrame, Dict[str, List[ResourceMetrics]]]:
        """Save and plot benchmark metrics.

        Args:
            resource_data (dict, optional): Resource data to save.
            objs (list, optional): List of objects with resource data.
            save_dir (Path, optional): Directory to save the output files.
            outfile_prefix (str, optional): Prefix for output files.

        Returns:
            Tuple[pd.DataFrame, dict]: DataFrame of metrics and merged resource data.
        """
        merged = (
            cls.collect_all(*(objs or []))
            if resource_data is None
            else dict(resource_data)
        )
        for method, runs in cls.global_resource_data.items():
            merged.setdefault(method, []).extend(runs)

        records = []
        for method, runs in merged.items():
            for i, m in enumerate(runs):
                records.append(
                    {
                        "method": method,
                        "run_id": i,
                        "memory_footprint": m.memory_footprint_mib,
                        "execution_time": m.execution_time_s,
                    }
                )

        df = pd.DataFrame(records)

        outdir = save_dir or Path("./performance_plots")
        outdir.mkdir(exist_ok=True, parents=True)
        df.to_csv(outdir / f"{outfile_prefix}_metrics.csv", index=False)
        df.to_json(outdir / f"{outfile_prefix}_metrics.json", orient="records")
