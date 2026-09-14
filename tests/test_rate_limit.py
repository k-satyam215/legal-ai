"""Tests for backend.core.rate_limit — in-memory sliding-window limiter."""
from backend.core.rate_limit import is_allowed, reset, stats


class TestIsAllowed:
    def test_allows_requests_under_limit(self):
        reset("ip-under")
        for _ in range(4):
            allowed, retry_after = is_allowed("ip-under", max_requests=5, window_seconds=60)
            assert allowed is True
            assert retry_after == 0

    def test_blocks_requests_over_limit(self):
        reset("ip-over")
        for _ in range(5):
            is_allowed("ip-over", max_requests=5, window_seconds=60)
        allowed, retry_after = is_allowed("ip-over", max_requests=5, window_seconds=60)
        assert allowed is False
        assert retry_after > 0

    def test_different_keys_tracked_independently(self):
        reset()
        for _ in range(5):
            is_allowed("ip-a", max_requests=5, window_seconds=60)
        allowed_a, _ = is_allowed("ip-a", max_requests=5, window_seconds=60)
        allowed_b, _ = is_allowed("ip-b", max_requests=5, window_seconds=60)
        assert allowed_a is False
        assert allowed_b is True

    def test_window_of_zero_always_blocks_after_first(self):
        # Degenerate but should not crash: a 0-second window means every
        # subsequent call in the same instant still counts against max_requests.
        reset("ip-zero")
        allowed1, _ = is_allowed("ip-zero", max_requests=1, window_seconds=60)
        allowed2, _ = is_allowed("ip-zero", max_requests=1, window_seconds=60)
        assert allowed1 is True
        assert allowed2 is False


class TestReset:
    def test_reset_clears_specific_key(self):
        reset("ip-c")
        for _ in range(5):
            is_allowed("ip-c", max_requests=5, window_seconds=60)
        reset("ip-c")
        allowed, _ = is_allowed("ip-c", max_requests=5, window_seconds=60)
        assert allowed is True

    def test_reset_none_clears_everything(self):
        is_allowed("ip-x", max_requests=1, window_seconds=60)
        is_allowed("ip-y", max_requests=1, window_seconds=60)
        reset()
        s = stats()
        assert s["tracked_keys"] == 0


class TestStats:
    def test_stats_reports_tracked_keys(self):
        reset()
        is_allowed("ip-1", max_requests=5, window_seconds=60)
        is_allowed("ip-2", max_requests=5, window_seconds=60)
        s = stats()
        assert s["tracked_keys"] == 2
        assert "window_seconds" in s
        assert "max_requests" in s
