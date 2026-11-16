"""Import entry point for the shared company-data preprocessing implementation."""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.Preprocessing.company_data import (
    DEFAULT_SOURCE,
    DEFAULT_SPLIT_CSV,
    SAMPLE_RATE,
    read_wav,
    select_window_starts,
    build_dataset,
    ChannelNormalizer,
)


__all__ = [
    "DEFAULT_SOURCE", "DEFAULT_SPLIT_CSV", "SAMPLE_RATE", "read_wav",
    "select_window_starts", "build_dataset", "ChannelNormalizer",
]
