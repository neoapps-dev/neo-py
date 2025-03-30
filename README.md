# neo.py Utils

A powerful Python utility library that helps you write more robust and efficient code.

## Installation

```bash
pip install neopy-utils
```

## Features

- **Retry Mechanisms**: Easily add retry logic to any function with configurable backoff and jitter
- **Async Support**: Async-compatible retry for use with asyncio
- **Performance Tools**: Memoization and timing decorators
- **Rate Limiting**: Control how often functions can be called
- **Validation**: Validate function arguments with custom validators
- **Serialization**: Convenient JSON serialization helpers
- **Concurrency**: Thread execution and chunked iterators
- **Security**: Password hashing utilities

## Usage Examples

### Retry Decorator

```python
from neopy-utils import retry

@retry(max_retries=5, delay=1, backoff_factor=2, jitter=True)
def unstable_network_call():
    # code that might fail due to network issues
    pass
```

### Async Retry

```python
from neopy-utils import retry_async

@retry_async(max_retries=3, delay=0.5)
async def unstable_api_call():
    # async code that might fail
    pass
```

### Memoization

```python
from neopy-utils import memoize

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### Performance Timing

```python
from neopy-utils import timed

@timed
def expensive_operation():
    # code to time
    pass
```

### Rate Limiting

```python
from neopy-utils import RateLimiter

rate_limiter = RateLimiter(max_calls=100, period=60)

@rate_limiter
def api_call():
    # API call limited to 100 calls per minute
    pass
```

### Argument Validation

```python
from neopy-utils import validate_args

@validate_args(
    user_id=lambda x: isinstance(x, int) and x > 0,
    email=lambda x: isinstance(x, str) and '@' in x
)
def register_user(user_id, email, name=None):
    # Will only execute if validators pass
    pass
```

### Running Code in a Thread

```python
from neopy-utils import run_in_thread

@run_in_thread
def background_task():
    # code that runs in background
    pass
    
thread = background_task()
thread.join()  # Wait for completion if needed
```

### Processing Data in Chunks

```python
from neopy-utils import chunked

for chunk in chunked(large_list, 1000):
    process_chunk(chunk)
```

### Password Security

```python
from neopy-utils import hash_password, verify_password

# Hash a password
hashed_password, salt = hash_password("my_secure_password")

# Later, verify it
is_valid = verify_password("my_secure_password", hashed_password, salt)
```

## License

[MIT License](LICENSE)