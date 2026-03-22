# wavelet_nn_junkbox
Junkbox for wavelets as loss function experiments and audio source separation.

## Overview
This repository contains a multiscale wavelet-based audio source separation pipeline. It integrates Convolutional Dictionary Autoencoders (ConvDictAE), UNet architectures, and a Fractal-Bayesian refinement post-processing stage.

### Key Components
- **`multiscale_wavelet_instrument_pipeline.py`**: Main pipeline for training separators using multiscale wavelet loss.
- **`fractal_bayes.py`**: Fractal-Bayesian refinement module for post-processing separation results.
- **`epi_sep.py`**: Epistemic-aware separation with uncertainty estimation and attribution.
- **`setup_mock_data.py`**: Tool for generating synthetic polyphonic audio data from MIDI files for development and testing.

## Installation
Ensure you have Python 3.10+ installed. Install dependencies via pip:

```bash
pip install torch torchaudio numpy scipy soundfile librosa PyWavelets pretty_midi mido captum tensorboard pytest
```

## Getting Started

### 1. Generate Mock Data
Generate synthetic instrument and mixture audio for testing:

```bash
python3 setup_mock_data.py
```

### 2. Run the Demo Pipeline
Execute a full demonstration of training and refinement:

```bash
PYTHONPATH=. python3 multiscale_wavelet_instrument_pipeline.py
```

## Testing
The project includes a comprehensive test suite using `pytest`.

Run all tests:
```bash
PYTHONPATH=. pytest tests/
```

### Test Coverage
- **`tests/test_components.py`**: Unit tests for neural network modules (GainNorm, UNet, etc.).
- **`tests/test_fractal_bayes.py`**: Validation of wavelet analysis, fractal estimation, and Bayesian shrinkage.
- **`tests/test_pipelines.py`**: Integration tests for the full separation flow and epistemic agent.

## License
MIT
