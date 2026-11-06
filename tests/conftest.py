import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")    # frax and the tests need float64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
