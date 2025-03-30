import os
import time
import json
import asyncio
import tempfile
import sys
from functools import wraps

import neo


def test_decorator(func):
    """Simple test decorator to run and report test functions."""
    @wraps(func)
    def wrapper():
        print(f"Running test: {func.__name__}")
        try:
            func()
            print(f"✅ {func.__name__} PASSED")
            return True
        except AssertionError as e:
            print(f"❌ {func.__name__} FAILED: {e}")
            return False
        except Exception as e:
            print(f"❌ {func.__name__} ERROR: {e}")
            return False
    return wrapper


class MockCall:
    """Mock function for testing."""
    def __init__(self, side_effects=None):
        self.side_effects = side_effects or []
        self.calls = 0
        self.args_history = []
        self.kwargs_history = []
    
    def __call__(self, *args, **kwargs):
        self.args_history.append(args)
        self.kwargs_history.append(kwargs)
        self.calls += 1
        
        if self.calls <= len(self.side_effects):
            result = self.side_effects[self.calls - 1]
            if isinstance(result, Exception):
                raise result
            return result
        return None


@test_decorator
def test_basic_retry():
    """Test that retry decorator retries the specified number of times."""
    calls = [ValueError(), ValueError(), "success"]
    mock = MockCall(calls)
    
    @neo.retry(max_retries=3, delay=0.01)
    def flaky_function():
        return mock()
    
    result = flaky_function()
    assert result == "success", f"Expected 'success', got {result}"
    assert mock.calls == 3, f"Expected 3 calls, got {mock.calls}"


@test_decorator
def test_retry_exhaustion():
    """Test that retry raises after max retries."""
    calls = [ValueError("Error 1"), ValueError("Error 2"), ValueError("Error 3")]
    mock = MockCall(calls)
    
    @neo.retry(max_retries=3, delay=0.01)
    def always_fails():
        return mock()
    
    try:
        always_fails()
        assert False, "Function should have raised ValueError"
    except ValueError:
        assert mock.calls == 3, f"Expected 3 calls, got {mock.calls}"


@test_decorator
def test_retry_with_backoff():
    """Test retry with exponential backoff."""
    start_times = []
    
    @neo.retry(max_retries=3, delay=0.1, backoff_factor=2)
    def record_times():
        start_times.append(time.time())
        if len(start_times) < 3:
            raise ValueError("Failing to test backoff")
        return "success"
    
    result = record_times()
    assert result == "success", f"Expected 'success', got {result}"
    
    assert len(start_times) == 3, f"Expected 3 timestamps, got {len(start_times)}"
    first_interval = start_times[1] - start_times[0]
    second_interval = start_times[2] - start_times[1]
    assert second_interval > first_interval * 1.5, "Backoff not working correctly"


@test_decorator
def test_retry_specific_exceptions():
    """Test that retry only catches specified exceptions."""
    
    @neo.retry(max_retries=3, delay=0.01, exceptions=(ValueError,))
    def raise_type_error():
        raise TypeError("Wrong type")
    
    try:
        raise_type_error()
        assert False, "Function should have raised TypeError immediately"
    except TypeError:
        pass


@test_decorator
def test_async_retry():
    """Test that async retry works correctly."""
    calls = [ValueError(), ValueError(), "success"]
    mock = MockCall(calls)
    
    @neo.retry_async(max_retries=3, delay=0.01)
    async def flaky_async_function():
        return mock()
    
    result = asyncio.run(flaky_async_function())
    assert result == "success", f"Expected 'success', got {result}"
    assert mock.calls == 3, f"Expected 3 calls, got {mock.calls}"


@test_decorator
def test_memoization():
    """Test that memoize caches return values."""
    call_count = 0
    
    @neo.memoize
    def expensive_calculation(x, y):
        nonlocal call_count
        call_count += 1
        return x + y
    
    result1 = expensive_calculation(2, 3)
    assert result1 == 5, f"Expected 5, got {result1}"
    assert call_count == 1, f"Expected 1 call, got {call_count}"
    result2 = expensive_calculation(2, 3)
    assert result2 == 5, f"Expected 5, got {result2}"
    assert call_count == 1, f"Expected still 1 call, got {call_count}"
    result3 = expensive_calculation(3, 4)
    assert result3 == 7, f"Expected 7, got {result3}"
    assert call_count == 2, f"Expected 2 calls, got {call_count}"


@test_decorator
def test_memoize_with_kwargs():
    """Test memoize with keyword arguments."""
    call_count = 0
    
    @neo.memoize
    def with_kwargs(x, y=10):
        nonlocal call_count
        call_count += 1
        return x + y
    
    result1 = with_kwargs(5)
    assert result1 == 15, f"Expected 15, got {result1}"
    assert call_count == 1, f"Expected 1 call, got {call_count}"
    
    result2 = with_kwargs(5)
    assert result2 == 15, f"Expected 15, got {result2}"
    assert call_count == 1, f"Expected still 1 call, got {call_count}"
    
    result3 = with_kwargs(5, y=20)
    assert result3 == 25, f"Expected 25, got {result3}"
    assert call_count == 2, f"Expected 2 calls, got {call_count}"


@test_decorator
def test_timed_decorator():
    """Test that timed decorator works correctly."""
    original_stdout = sys.stdout
    sys.stdout = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    
    try:
        @neo.timed
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        result = slow_function()
        assert result == "done", f"Expected 'done', got {result}"
        sys.stdout.flush()
        sys.stdout.seek(0)
        output = sys.stdout.read()
        assert "slow_function" in output, "Function name not in output"
        assert "executed in" in output, "Timing info not in output"
    finally:
        temp_path = sys.stdout.name
        sys.stdout.close()
        sys.stdout = original_stdout
        if os.path.exists(temp_path):
            os.remove(temp_path)


@test_decorator
def test_rate_limiting():
    """Test that rate limiter enforces call limits."""
    
    limiter = neo.RateLimiter(max_calls=2, period=0.5)
    
    @limiter
    def limited_function():
        return "called"

    result1 = limited_function()
    assert result1 == "called", f"Expected 'called', got {result1}"
    result2 = limited_function()
    assert result2 == "called", f"Expected 'called', got {result2}"
    try:
        limited_function()
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Rate limit exceeded" in str(e), f"Unexpected exception: {e}"

    time.sleep(0.5)
    result3 = limited_function()
    assert result3 == "called", f"Expected 'called', got {result3}"


@test_decorator
def test_validate_args():
    """Test argument validation."""
    
    @neo.validate_args(
        age=lambda x: isinstance(x, int) and x >= 0,
        name=lambda x: isinstance(x, str) and len(x) > 0
    )
    def register_person(name, age):
        return f"{name} is {age} years old"
    
    result = register_person("Alice", 30)
    assert result == "Alice is 30 years old", f"Expected 'Alice is 30 years old', got {result}"

    try:
        register_person("", 30)
        assert False, "Should have raised ValueError for empty name"
    except ValueError:
        pass

    try:
        register_person("Bob", -5)
        assert False, "Should have raised ValueError for negative age"
    except ValueError:
        pass

    try:
        register_person("Charlie", "thirty")
        assert False, "Should have raised ValueError for non-int age"
    except ValueError:
        pass


@test_decorator
def test_json_serialization():
    """Test JSON serialization and deserialization."""
    data = {"name": "Test", "values": [1, 2, 3]}
    
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name
    
    try:
        neo.serialize_to_json(data, temp_path)
        with open(temp_path, 'r') as f:
            content = json.load(f)
            assert content == data, f"Expected {data}, got {content}"

        loaded = neo.deserialize_from_json(temp_path)
        assert loaded == data, f"Expected {data}, got {loaded}"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@test_decorator
def test_run_in_thread():
    """Test running a function in a thread."""
    results = []
    
    @neo.run_in_thread
    def background_task(value):
        time.sleep(0.1)
        results.append(value)

    thread = background_task("completed")
    assert results == [], f"Expected empty list, got {results}"
    thread.join()
    assert results == ["completed"], f"Expected ['completed'], got {results}"


@test_decorator
def test_chunked():
    """Test chunked iterator."""
    data = list(range(10))
    
    chunks = list(neo.chunked(data, 3))
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunks == expected, f"Expected {expected}, got {chunks}"
    chunks = list(neo.chunked(list(range(6)), 2))
    expected = [[0, 1], [2, 3], [4, 5]]
    assert chunks == expected, f"Expected {expected}, got {chunks}"


@test_decorator
def test_password_hashing():
    """Test password hashing and verification."""
    password = "secret_password"
    hashed, salt = neo.hash_password(password)
    assert neo.verify_password(password, hashed, salt), "Password verification failed"
    assert not neo.verify_password("wrong_password", hashed, salt), "Incorrect password verified"


@test_decorator
def test_hash_with_provided_salt():
    """Test hashing with a provided salt."""
    password = "secret_password"
    salt = "fixed_salt"
    hashed1, returned_salt = neo.hash_password(password, salt)
    
    assert returned_salt == salt, f"Expected salt {salt}, got {returned_salt}"
    hashed2, _ = neo.hash_password(password, salt)
    assert hashed1 == hashed2, "Hashing with same salt produced different results"


def run_all_tests():
    """Run all test functions and report results."""
    test_functions = [
        test_basic_retry,
        test_retry_exhaustion,
        test_retry_with_backoff,
        test_retry_specific_exceptions,
        test_async_retry,
        test_memoization,
        test_memoize_with_kwargs,
        test_timed_decorator,
        test_rate_limiting,
        test_validate_args,
        test_json_serialization,
        test_run_in_thread,
        test_chunked,
        test_password_hashing,
        test_hash_with_provided_salt
    ]
    
    total = len(test_functions)
    passed = 0
    
    print(f"\n{'=' * 50}")
    print(f"Running {total} tests for neo.py")
    print(f"{'=' * 50}\n")
    
    for test_func in test_functions:
        if test_func():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'=' * 50}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)