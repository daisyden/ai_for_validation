"""
My Pytest Plugin
"""

__version__ = "1.0.0"
__author__ = "daisyden"

# Import hook implementations
from .call_tracer import (
    trace_calls,
)