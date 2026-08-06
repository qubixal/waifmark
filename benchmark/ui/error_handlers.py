"""Error handling utilities for Streamlit UI components."""

from __future__ import annotations

import logging
import traceback
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import streamlit as st

F = TypeVar("F", bound=Callable[..., Any])
LOGGER = logging.getLogger(__name__)


def handle_ui_errors(error_message: str = "An unexpected error occurred", show_trace: bool = False) -> Callable[[F], F]:
    """Decorator to wrap UI operations with error handling and user feedback.

    Args:
        error_message: User-facing error message
        show_trace: Whether to show full traceback (debug mode)

    Example:
        @handle_ui_errors("Failed to start server")
        def start_server():
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                LOGGER.exception(f"Error in {func.__name__}: {exc}")
                st.error(f"{error_message}: {str(exc)}")
                if show_trace:
                    st.code(traceback.format_exc(), language="python")
                return None

        return wrapper  # type: ignore

    return decorator


def try_catch_block(func: Callable[..., Any], *args: Any, error_message: str = "", **kwargs: Any) -> Optional[Any]:
    """Execute a function with error handling, returning None on failure.

    Args:
        func: Function to execute
        *args: Positional arguments
        error_message: Optional custom error message
        **kwargs: Keyword arguments

    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        LOGGER.exception(f"Error executing {func.__name__}: {exc}")
        if error_message:
            st.error(error_message)
        else:
            st.error(f"Operation failed: {str(exc)}")
        return None


def safe_json_load(path: Any, default: Any = None) -> Any:
    """Safely load a JSON file with error handling.

    Args:
        path: Path to JSON file
        default: Default value if load fails

    Returns:
        Loaded JSON or default value
    """
    import json
    from pathlib import Path

    try:
        if isinstance(path, str):
            path = Path(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning(f"Failed to load JSON from {path}: {exc}")
        return default


def safe_yaml_load(path: Any, default: Any = None) -> Any:
    """Safely load a YAML file with error handling.

    Args:
        path: Path to YAML file
        default: Default value if load fails

    Returns:
        Loaded YAML or default value
    """
    import yaml
    from pathlib import Path

    try:
        if isinstance(path, str):
            path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        LOGGER.warning(f"Failed to load YAML from {path}: {exc}")
        return default
