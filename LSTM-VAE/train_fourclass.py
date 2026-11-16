"""Train raw-waveform classification with LSTM-VAE and labels.json classes."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier import StateClassifier
from model import ModelConfig
from fourclass_training import run_training


def main(argv=None):
    return run_training(StateClassifier, ModelConfig, 'LSTM-VAE', argv)


if __name__ == '__main__':
    raise SystemExit(main())
