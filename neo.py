"""
A collection of utilities to simplify common programming tasks
and make your code more robust and efficient.
"""

import time
import random
import functools
import threading
import asyncio
import inspect
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Generator

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def retry(max_retries: int = 3, delay: float = 1, backoff_factor: float = 1.0, 
          jitter: bool = False, exceptions: Tuple = (Exception,)) -> Callable:
    """
    Decorator that retries a function or method if it raises specified exceptions.
    
    Args:
        max_retries: Maximum number of retries before giving up
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier applied to delay between retries
        jitter: Whether to add random jitter to delay
        exceptions: Tuple of exceptions that trigger a retry
    
    Returns:
        Decorated function
    
    Example:
        @retry(max_retries=5, delay=2, backoff_factor=2)
        def unstable_network_call():
            # code that might fail due to network issues
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    
                    if jitter:
                        sleep_time = current_delay * (0.5 + random.random())
                    else:
                        sleep_time = current_delay
                        
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor
                    
        return wrapper
    return decorator


def retry_async(max_retries: int = 3, delay: float = 1, backoff_factor: float = 1.0,
                jitter: bool = False, exceptions: Tuple = (Exception,)) -> Callable:
    """
    Async version of the retry decorator for use with async functions.
    
    Args:
        max_retries: Maximum number of retries before giving up
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier applied to delay between retries
        jitter: Whether to add random jitter to delay
        exceptions: Tuple of exceptions that trigger a retry
    
    Returns:
        Decorated async function
    
    Example:
        @retry_async(max_retries=5, delay=2, backoff_factor=2)
        async def unstable_api_call():
            # async code that might fail
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    
                    if jitter:
                        sleep_time = current_delay * (0.5 + random.random())
                    else:
                        sleep_time = current_delay
                        
                    await asyncio.sleep(sleep_time)
                    current_delay *= backoff_factor
                    
        return wrapper
    return decorator


def memoize(func: F) -> F:
    """
    Decorator that caches function results based on input arguments.
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    
    Example:
        @memoize
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
        
    return wrapper


def timed(func: F) -> F:
    """
    Decorator that measures and prints execution time of a function.
    
    Args:
        func: Function to time
    
    Returns:
        Timed function
    
    Example:
        @timed
        def slow_operation():
            time.sleep(2)
            return "Done"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class RateLimiter:
    """
    Rate limiter that limits function calls to a maximum number per time period.
    
    Example:
        rate_limiter = RateLimiter(max_calls=100, period=60)
        
        @rate_limiter
        def api_call():
            # API call limited to 100 calls per minute
            pass
    """
    
    def __init__(self, max_calls: int, period: float):
        """
        Initialize a rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
        
    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time <= self.period]
                
                if len(self.calls) >= self.max_calls:
                    raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.period} seconds")
                
                self.calls.append(now)
                
            return func(*args, **kwargs)
        return wrapper


def validate_args(**validators):
    """
    Decorator that validates function arguments against provided validators.
    
    Args:
        validators: Mapping of argument names to validator functions
    
    Returns:
        Decorated function with argument validation
    
    Example:
        @validate_args(
            user_id=lambda x: isinstance(x, int) and x > 0,
            email=lambda x: isinstance(x, str) and '@' in x
        )
        def register_user(user_id, email, name=None):
            # Will only execute if validators pass
            pass
    """
    def decorator(func):
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for arg_name, validator in validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(f"Argument '{arg_name}' with value {value} failed validation")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def serialize_to_json(obj: Any, file_path: str, indent: int = 2) -> None:
    """
    Serialize an object to a JSON file.
    
    Args:
        obj: Object to serialize
        file_path: Path to output file
        indent: JSON indentation level
        
    Example:
        data = {"users": [{"id": 1, "name": "John"}]}
        serialize_to_json(data, "users.json")
    """
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=indent)


def deserialize_from_json(file_path: str) -> Any:
    """
    Deserialize an object from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Deserialized object
        
    Example:
        data = deserialize_from_json("users.json")
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def run_in_thread(func: F) -> Callable:
    """
    Decorator that runs a function in a separate thread.
    
    Args:
        func: Function to run in a thread
    
    Returns:
        Function that returns a thread object
    
    Example:
        @run_in_thread
        def background_task():
            # code that runs in background
            pass
            
        thread = background_task()
        thread.join()  # Wait for completion if needed
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


def chunked(iterable: List[T], size: int) -> Generator[List[T], None, None]:
    """
    Split an iterable into chunks of specified size.
    
    Args:
        iterable: Iterable to split
        size: Size of each chunk
    
    Yields:
        Chunks of the iterable
        
    Example:
        for chunk in chunked(range(10), 3):
            process_chunk(chunk)
    """
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using SHA-256 with random salt.
    
    Args:
        password: Password to hash
        salt: Optional salt (generates random salt if None)
        
    Returns:
        Tuple of (hashed_password, salt)
        
    Example:
        hashed_pw, salt = hash_password("my_secure_password")
        # Store both hashed_pw and salt
    """
    if salt is None:
        salt = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()
    
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against a hash and salt.
    
    Args:
        password: Password to verify
        hashed_password: Previously hashed password
        salt: Salt used in original hashing
        
    Returns:
        True if password matches, False otherwise
        
    Example:
        is_valid = verify_password("my_password", stored_hash, stored_salt)
    """
    return hash_password(password, salt)[0] == hashed_password


__version__ = "1.0.0"