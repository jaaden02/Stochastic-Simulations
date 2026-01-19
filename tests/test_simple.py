"""Simple test without imports."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic():
    """Basic test that always passes."""
    assert 1 + 1 == 2


def test_import():
    """Test that we can import the library."""
    try:
        import stochlib
        assert True
    except ImportError as e:
        assert False, f"Failed to import stochlib: {e}"
