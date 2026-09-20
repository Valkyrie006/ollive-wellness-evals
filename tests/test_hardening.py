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


# --------------------------------------------------------------------------
# provider retry hints
# --------------------------------------------------------------------------

def test_reads_geminis_retry_hint():
    """Gemini answers a 429 with the exact wait. Guessing instead of reading
    it is what turned a 20-requests-per-minute limit into a run that lost
    most of its items."""
    from agents.core import retry_after_seconds
    err = Exception('429 You exceeded your current quota ... '
                    'Please retry in 33.136598459s.')
    assert retry_after_seconds(err) == 33.136598459


def test_reads_retry_delay_field():
    from agents.core import retry_after_seconds
    assert retry_after_seconds(Exception('{"retryDelay": "27s"}')) == 27.0


def test_returns_none_when_no_hint_is_given():
    """Falls back to exponential backoff rather than inventing a number."""
    from agents.core import retry_after_seconds
    assert retry_after_seconds(Exception("connection reset by peer")) is None


def test_honours_the_hint_instead_of_exponential_backoff(monkeypatch):
    import agents.core as core

    slept = []
    monkeypatch.setattr(core.time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def rate_limited_twice(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 quota exceeded. Please retry in 30s.")
        return "ok"

    assert core._completion_with_retry(rate_limited_twice) == "ok"
    # 31s each (hint + 1s buffer), NOT 20s then 40s
    assert slept == [31.0, 31.0], slept


def test_retry_sleep_is_bounded_so_a_throttled_provider_cannot_hang_a_request(monkeypatch):
    """A free tier that keeps answering "retry in 60s" must not turn one
    request into a multi-minute hang. Past the sleep budget the provider's
    own error is raised, so the caller gets a cause instead of silence."""
    import agents.core as core

    slept = []
    monkeypatch.setattr(core.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(core, "MAX_RETRY_SLEEP_TOTAL_S", 75.0)
    calls = {"n": 0}

    def always_rate_limited(**kwargs):
        calls["n"] += 1
        raise RuntimeError("429 quota exceeded. Please retry in 60s.")

    try:
        core._completion_with_retry(always_rate_limited)
    except RuntimeError as e:
        assert "429" in str(e), "the provider's own error must survive"
    else:
        raise AssertionError("expected the provider error to be raised")

    assert sum(slept) <= 75.0, f"slept {sum(slept)}s, over budget"
    # it stopped on the budget, not on running out of attempts
    assert calls["n"] < core.MAX_COMPLETION_RETRIES


def test_reads_groqs_retry_hint_including_a_multi_hour_daily_cap():
    """Groq words its 429 differently from Gemini. Matching only Gemini's
    phrasing meant a Groq limit fell back to a blind 20s/40s backoff that
    could never clear a per-day cap."""
    from agents.core import retry_after_seconds
    assert retry_after_seconds(
        Exception("Rate limit reached ... Please try again in 1m23.4s")) == 83.4
    assert retry_after_seconds(
        Exception("Rate limit reached ... Please try again in 2h14m30s")) == 8070.0
    assert retry_after_seconds(Exception("Please try again in 45.2s")) == 45.2


def test_a_multi_hour_cap_fails_fast_rather_than_sleeping_through_it(monkeypatch):
    """A per-day cap can't be waited out inside one request. The budget must
    turn it into an immediate, explanatory failure."""
    import agents.core as core

    slept = []
    monkeypatch.setattr(core.time, "sleep", lambda s: slept.append(s))

    def daily_cap(**kwargs):
        raise RuntimeError("429 Rate limit reached for model X on tokens per day "
                           "(TPD). Please try again in 2h14m30s")

    try:
        core._completion_with_retry(daily_cap)
    except RuntimeError as e:
        assert "per day" in str(e)
    else:
        raise AssertionError("expected the provider error to be raised")
    assert slept == [], "must not sleep at all when the hint exceeds the budget"
