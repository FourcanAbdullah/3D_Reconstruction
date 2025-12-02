from __future__ import annotations

from contextlib import contextmanager
from typing import List, Tuple, Dict

import time


class StepTimer:
    """Utility to measure and record named step durations."""
    def __init__(self):
        self.records = []
        self._start = None

    @contextmanager
    def track(self, name: str):
        import time
        start = time.time()
        yield
        elapsed = time.time() - start
        self.records.append((name, elapsed))

    def summary(self):
        total = sum(t for _, t in self.records)
        return {
            "steps": [{"name": n, "seconds": round(t, 2)} for n, t in self.records],
            "total_seconds": round(total, 2)
        }