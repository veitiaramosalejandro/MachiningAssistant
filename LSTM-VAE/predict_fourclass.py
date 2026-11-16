"""Classify one complete five-second WAV with a saved LSTM classification head."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fourclass_prediction import run_prediction
from classifier import StateClassifier
from model import ModelConfig


if __name__ == '__main__':
    raise SystemExit(run_prediction(StateClassifier, ModelConfig, 'LSTM-VAE'))
