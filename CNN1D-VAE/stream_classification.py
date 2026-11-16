"""Replay PCM WAV chunks through one loaded CNN classification model."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier import StateClassifier
from model import ModelConfig
from classification_streaming import run_streaming


if __name__ == '__main__':
    raise SystemExit(run_streaming(StateClassifier, ModelConfig, 'CNN1D-VAE'))
