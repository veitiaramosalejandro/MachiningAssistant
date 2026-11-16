"""Summarize reconstructed WAV windows for JSON output and the console."""
from math import fsum


def summarize_prediction(result):
    windows = result['windows']
    duration = sum(row['end_frame'] - row['start_frame'] for row in windows) / result['sample_rate_hz']
    if not windows:
        return {
            'evaluated_duration_s': duration,
            'mean_reconstruction_mse_standardized': None,
            'mean_reconstruction_mse_by_channel_standardized': [None, None],
        }
    count = len(windows)
    return {
        'evaluated_duration_s': duration,
        'mean_reconstruction_mse_standardized': fsum(row['reconstruction_mse_standardized'] for row in windows) / count,
        'mean_reconstruction_mse_by_channel_standardized': [
            fsum(row['reconstruction_mse_by_channel_standardized'][channel] for row in windows) / count
            for channel in range(2)
        ],
    }


def print_prediction_summary(summary):
    print('-' * 64, flush=True)
    print('Prediction results (standardized reconstruction MSE):', flush=True)
    print(f"Evaluated duration: {summary['evaluated_duration_s']:.3f} s", flush=True)
    mean = summary['mean_reconstruction_mse_standardized']
    if mean is None:
        print('Reconstruction MSE: unavailable (input shorter than one model window).', flush=True)
        print('-' * 64, flush=True)
        return
    left, right = summary['mean_reconstruction_mse_by_channel_standardized']
    print(f'Mean MSE: {mean:.6f}', flush=True)
    print(f'Left channel MSE: {left:.6f}', flush=True)
    print(f'Right channel MSE: {right:.6f}', flush=True)
    print('-' * 64, flush=True)
