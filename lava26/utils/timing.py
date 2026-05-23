import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PhaseRecord:
    name: str
    start: float = 0.0
    end: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.end is not None:
            return self.end - self.start
        return time.monotonic() - self.start


class BudgetTracker:
    def __init__(self, total_s: float, preprocess_s: float, reserve_s: float):
        self.total_s = total_s
        self.preprocess_s = preprocess_s
        self.reserve_s = reserve_s
        self._start = time.monotonic()
        self._phases: Dict[str, PhaseRecord] = {}

    def start_phase(self, name: str) -> None:
        self._phases[name] = PhaseRecord(name=name, start=time.monotonic())

    def end_phase(self, name: str) -> float:
        rec = self._phases.get(name)
        if rec:
            rec.end = time.monotonic()
            return rec.elapsed
        return 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def remaining(self) -> float:
        return max(0.0, self.total_s - self.elapsed() - self.reserve_s)

    def is_budget_exhausted(self) -> bool:
        return self.remaining() <= 0.0

    def is_fast_path(self, fast_path_threshold_s: float) -> bool:
        return self.remaining() <= fast_path_threshold_s

    @contextmanager
    def question_timer(self, timeout_s: float):
        if not hasattr(signal, "SIGALRM"):
            # Windows fallback — no alarm, just yield (per-question timeout not enforced)
            yield
            return

        def _handler(signum, frame):
            raise TimeoutError(f"Question timed out after {timeout_s}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout_s))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def print_summary(self) -> None:
        total_elapsed = self.elapsed()
        print("\n" + "=" * 60)
        print("  TIMING SUMMARY")
        print("=" * 60)
        for name, rec in self._phases.items():
            elapsed = rec.elapsed
            print(f"  {name:<30} {elapsed:>8.1f}s  ({elapsed/60:.1f} min)")
        print("-" * 60)
        print(f"  {'TOTAL':<30} {total_elapsed:>8.1f}s  ({total_elapsed/60:.1f} min)")
        print("=" * 60)
