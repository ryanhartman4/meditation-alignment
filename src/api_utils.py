"""
Utilities for API calls with proper error handling and retry logic.
"""

import time
import random
from typing import Callable, Any, Optional, Dict
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

def exponential_backoff_retry(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: The function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    
    Returns:
        The result of the function call
        
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (RateLimitError, APITimeoutError) as e:
            last_exception = e
            
            if attempt == max_retries:
                print(f"Max retries ({max_retries}) exceeded. Final error: {e}")
                raise
            
            # Calculate delay with exponential backoff
            delay = min(initial_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())
            
            print(f"API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            
        except APIError as e:
            # For other API errors, check if they're retryable
            if hasattr(e, 'status_code') and e.status_code >= 500:
                # Server errors are retryable
                last_exception = e
                if attempt < max_retries:
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    print(f"Server error {e.status_code} (attempt {attempt + 1}/{max_retries + 1})")
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
            # Non-retryable errors
            raise
        except Exception as e:
            # Non-API errors should not be retried
            raise
    
    if last_exception:
        raise last_exception


def make_api_call_with_retry(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    response_format: Optional[Dict] = None,
    **kwargs
) -> Any:
    """
    Make an OpenAI API call with automatic retry logic.
    
    Args:
        client: OpenAI client instance
        model: Model to use
        messages: Messages for the chat completion
        temperature: Temperature parameter
        max_tokens: Maximum tokens in response
        response_format: Optional response format specification
        **kwargs: Additional arguments for the API call
        
    Returns:
        The API response
    """
    def _make_call():
        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if response_format is not None:
            call_kwargs["response_format"] = response_format
            
        return client.chat.completions.create(**call_kwargs)
    
    return exponential_backoff_retry(_make_call)