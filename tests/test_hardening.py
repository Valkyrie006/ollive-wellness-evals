"""Covers the two bounded-resource mechanisms that only matter once the app
is reachable by people you don't control: the rate limiter and the session
store. Both are hand-rolled (see docs/DESIGN.md for why), so both need
tests that pin their actual behaviour rather than trusting a library.
"""
import time

from agents.core import SessionStore
from api.ratelimit import RateLimiter

# --------------------------------------------------------------------------
# rate limiter
# --------------------------------------------------------------------------

def test_allows_up_to_the_limit_then_blocks():
    rl = RateLimiter(limit=3, window_seconds=60)
    assert [rl.check("ip-a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = rl.check("ip-a")
    assert allowed is False
    assert retry_after >= 1


def test_buckets_are_per_client():
    rl = RateLimiter(limit=1, window_seconds=60)
    assert rl.check("ip-a")[0] is True
    assert rl.check("ip-a")[0] is False
    assert rl.check("ip-b")[0] is True, "one client's burst must not block another"


def test_window_slides_so_clients_recover():
    rl = RateLimiter(limit=2, window_seconds=1)
    assert rl.check("ip-a")[0] is True
    assert rl.check("ip-a")[0] is True
    assert rl.check("ip-a")[0] is False
    time.sleep(1.05)
    assert rl.check("ip-a")[0] is True, "window should have slid open again"


def test_rejected_requests_do_not_extend_the_lockout():
    """A blocked request must not be recorded, otherwise a client hammering
    the endpoint keeps pushing its own window forward and locks itself out
    for as long as it keeps trying."""
    rl = RateLimiter(limit=1, window_seconds=2)
    assert rl.check("ip-a")[0] is True
    for _ in range(20):
        rl.check("ip-a")
    time.sleep(2.05)
    assert rl.check("ip-a")[0] is True


def test_limit_of_zero_disables_limiting():
    rl = RateLimiter(limit=0, window_seconds=60)
    assert all(rl.check("ip-a")[0] for _ in range(50))


# --------------------------------------------------------------------------
# session store
# --------------------------------------------------------------------------

def test_round_trips_messages():
    store = SessionStore(max_sessions=10, ttl_seconds=60)
    store.set("s1", [{"role": "user", "content": "hi"}])
    assert store.get("s1") == [{"role": "user", "content": "hi"}]


def test_unknown_session_is_empty_not_an_error():
    assert SessionStore().get("never-seen") == []


def test_entries_expire_after_ttl():
    store = SessionStore(max_sessions=10, ttl_seconds=1)
    store.set("s1", [{"role": "user", "content": "hi"}])
    time.sleep(1.05)
    assert store.get("s1") == []


def test_store_is_capped_and_evicts_oldest_first():
    """Without a cap, a spray of unique session ids is a trivial way to
    exhaust process memory."""
    store = SessionStore(max_sessions=3, ttl_seconds=60)
    for i in range(5):
        store.set(f"s{i}", [{"role": "user", "content": str(i)}])
    assert len(store) == 3
    assert store.get("s0") == [], "oldest should have been evicted"
    assert store.get("s4") != [], "newest should survive"


def test_reading_a_session_marks_it_recently_used():
    store = SessionStore(max_sessions=2, ttl_seconds=60)
    store.set("old", [{"role": "user", "content": "a"}])
    store.set("new", [{"role": "user", "content": "b"}])
    store.get("old")                      # touch it
    store.set("newest", [{"role": "user", "content": "c"}])
    assert store.get("old") != [], "recently-read session should not be the eviction victim"


def test_returned_history_is_a_copy():
    """Callers mutate the history list while building a turn; that must not
    corrupt what's stored."""
    store = SessionStore()
    store.set("s1", [{"role": "user", "content": "hi"}])
    store.get("s1").append({"role": "user", "content": "injected"})
    assert len(store.get("s1")) == 1
