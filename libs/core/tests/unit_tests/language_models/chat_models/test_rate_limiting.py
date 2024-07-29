import time

from langchain_core.caches import InMemoryCache
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter


def test_rate_limit_invoke() -> None:
    """Add rate limiter."""

    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02

    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # The second time we call the model, we should have 1 extra token
    # to proceed immediately.
    assert toc - tic < 0.005

    # The third time we call the model, we need to wait again for a token
    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02


async def test_rate_limit_ainvoke() -> None:
    """Add rate limiter."""

    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=20, check_every_n_seconds=0.1, max_bucket_size=10
        ),
    )
    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.1 < toc - tic < 0.2

    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # The second time we call the model, we should have 1 extra token
    # to proceed immediately.
    assert toc - tic < 0.01

    # The third time we call the model, we need to wait again for a token
    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.1 < toc - tic < 0.2


def test_rate_limit_batch() -> None:
    """Test that batch and stream calls work with rate limiters."""
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Need 2 tokens to proceed
    time_to_fill = 2 / 200.0
    tic = time.time()
    model.batch(["foo", "foo"])
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert time_to_fill < toc - tic < time_to_fill + 0.01


async def test_rate_limit_abatch() -> None:
    """Test that batch and stream calls work with rate limiters."""
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Need 2 tokens to proceed
    time_to_fill = 2 / 200.0
    tic = time.time()
    await model.abatch(["foo", "foo"])
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert time_to_fill < toc - tic < time_to_fill + 0.01


def test_rate_limit_stream() -> None:
    """Test rate limit by stream."""
    model = GenericFakeChatModel(
        messages=iter(["hello world", "hello world", "hello world"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Check astream
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds

    # Second time around we should have 1 token left
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert toc - tic < 0.005  # Slightly smaller than check every n seconds

    # Third time around we should have 0 tokens left
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds


async def test_rate_limit_astream() -> None:
    """Test rate limiting astream."""
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=20, check_every_n_seconds=0.1, max_bucket_size=10
    )
    model = GenericFakeChatModel(
        messages=iter(["hello world", "hello world", "hello world"]),
        rate_limiter=rate_limiter,
    )
    # Check astream
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    assert 0.1 < toc - tic < 0.2

    # Second time around we should have 1 token left
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert toc - tic < 0.01  # Slightly smaller than check every n seconds

    # Third time around we should have 0 tokens left
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    assert 0.1 < toc - tic < 0.2


def test_rate_limit_skips_cache() -> None:
    """Test that rate limiting does not rate limit cache look ups."""
    cache = InMemoryCache()
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=1
        ),
        cache=cache,
    )

    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02

    for _ in range(2):
        # Cache hits
        tic = time.time()
        model.invoke("foo")
        toc = time.time()
        # Should be larger than check every n seconds since the token bucket starts
        # with 0 tokens.
        assert toc - tic < 0.005

    # Test verifies that there's only a single key
    # Test also verifies that rate_limiter information is not part of the
    # cache key
    assert list(cache._cache) == [
        (
            '[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", '
            '"messages", '
            '"HumanMessage"], "kwargs": {"content": "foo", "type": "human"}}]',
            "[('_type', 'generic-fake-chat-model'), ('stop', None)]",
        )
    ]


class SerializableModel(GenericFakeChatModel):
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


def test_serialization_with_rate_limiter() -> None:
    """Test model serialization with rate limiter."""
    from langchain_core.load import dumps

    model = SerializableModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=1
        ),
    )
    serialized_model = dumps(model)
    assert InMemoryRateLimiter.__name__ not in serialized_model


async def test_rate_limit_skips_cache_async() -> None:
    """Test that rate limiting does not rate limit cache look ups."""
    cache = InMemoryCache()
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryRateLimiter(
            requests_per_second=100, check_every_n_seconds=0.01, max_bucket_size=1
        ),
        cache=cache,
    )

    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02

    for _ in range(2):
        # Cache hits
        tic = time.time()
        await model.ainvoke("foo")
        toc = time.time()
        # Should be larger than check every n seconds since the token bucket starts
        # with 0 tokens.
        assert toc - tic < 0.005
