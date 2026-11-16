"""Train and evaluate the CNN1D-VAE classification branch on prepared MSM data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'msm_classification'))
from experiment import run_training
from classifier import StateClassifier, classification_loss
from model import ModelConfig

if __name__=='__main__':
    run_training(StateClassifier,ModelConfig,classification_loss,'CNN1D-VAE')
