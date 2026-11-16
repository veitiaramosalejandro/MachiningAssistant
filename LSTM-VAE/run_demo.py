"""Run shared reconstruction training for LSTM-VAE."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction_training import evaluate as _evaluate, run_training
from model import ModelConfig, VAE, loss_function
from runtime import MODEL_FAMILY, StreamingPredictor, VAEEngine


def evaluate(model, x, batch_size, beta):
    return _evaluate(model, x, batch_size, beta, loss_function)


def main(argv=None):
    return run_training(model_class=VAE, config_class=ModelConfig,
                        loss_function=loss_function, engine_class=VAEEngine,
                        streaming_class=StreamingPredictor, model_family=MODEL_FAMILY,
                        argv=argv)


if __name__ == '__main__':
    raise SystemExit(main())
