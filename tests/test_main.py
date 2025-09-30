"""
Tests for the main module.
"""

import pytest
from src.main import main


def test_main():
    """Test the main function."""
    # This is a simple test that verifies main() can be called
    # without raising an exception
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised {type(e).__name__}: {e}")
