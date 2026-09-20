"""In-process sliding-window rate limiter.

Chosen over `slowapi`/Redis deliberately: this project already lost real
time to dependency-resolution problems, the deployment target is a single
instance, and the whole mechanism is forty lines that are easy to read and
test. The trade-off is explicit - counters live in one process, so with
multiple replicas each replica enforces its own limit, and everything
resets on restart. Moving to Redis means replacing this class, not the
call sites. See docs/DESIGN.md.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

# Stop the bucket map itself becoming a memory-growth vector under a spray
# of spoofed client keys.
MAX_TRACKED_CLIENTS = 10_000


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds).

        A request that is allowed is recorded; one that is rejected is not,
        so a client hammering the endpoint can't push its own window
        forward and extend its own lockout indefinitely.
        """
        if self.limit <= 0:  # 0 or negative disables limiting
            return True, 0

        now = time.monotonic()
        with self._lock:
            timestamps = self._hits.get(key)
            if timestamps is None:
                timestamps = deque()
                self._hits[key] = timestamps
            self._hits.move_to_end(key)

            while timestamps and now - timestamps[0] >= self.window:
                timestamps.popleft()

            if len(timestamps) >= self.limit:
                retry_after = max(1, int(self.window - (now - timestamps[0])) + 1)
                return False, retry_after

            timestamps.append(now)

            # Evict least-recently-seen clients once the map is oversized.
            while len(self._hits) > MAX_TRACKED_CLIENTS:
                self._hits.popitem(last=False)

            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
