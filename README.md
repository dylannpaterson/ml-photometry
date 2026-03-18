# Castor: Roman Point Source ML Pipeline

A specialized machine learning pipeline designed for fast, direct point-source detection, photometry, and completeness estimation on Level 2 slope images from the Nancy Grace Roman Space Telescope.

## Overview

Traditional point-spread function (PSF) fitting algorithms can be computationally prohibitive in extremely crowded fields, such as those targeted by the Roman Bulge Time Domain Survey (upwards of millions of stars per sensor chip assembly). This project frames point-source photometry as a **Dense Grid Prediction** problem (inspired by architectures like YOLO and CenterNet), allowing the network to simultaneously detect sources, measure their flux, estimate local background, and predict a recoverability/completeness score in a single forward pass.

### Key Capabilities
- **Simultaneous Detection & Photometry:** Predicts source probabilities ($p$) and log-transformed flux ($m$) directly.
- **Sub-pixel Localization:** Predicts fine spatial offsets ($dx, dy$) within the grid.
- **Generative PSF Recovery:** Learns a canonical, normalized 9x9 PSF profile ($S$) for each detection, enabling full generative image reconstruction and residual analysis.
- **Background Modeling:** Predicts a smoothly varying 2D background surface ($b$) alongside the star catalog.
- **Completeness Estimation:** Outputs a calibrated recoverability score ($c$) for each star, eliminating the need for massive post-hoc injection/recovery simulations to characterize catalog depth.

## Architecture

The model processes $256 \times 256$ image chunks through a ResNet-34 backbone, outputting a $128 \times 128$ spatial grid. Each cell represents a $2 \times 2$ pixel area and is capable of predicting up to $K=3$ overlapping point sources.

Each predicted star consists of **86 parameters**:
- $p$: Probability (Objectness score)
- $dx, dy$: Sub-pixel offsets
- $m$: Magnitude ($\log_{10}(\text{Flux})$)
- $c$: Completeness / Recoverability score
- $S$: 81 values representing a $9 \times 9$ normalized PSF shape

Additionally, each $2 \times 2$ cell predicts a shared local background value ($b$).

## Data Storage Strategy

To handle the massive dimensionality of predicting 81 shape parameters for millions of stars without causing disk I/O bottlenecks, the pipeline uses a **Sparse-on-Disk, Dense-in-RAM** approach:
- Target grids are stored sparsely (saving only active star shapes and their indices).
- The PyTorch `Dataset` automatically re-densifies these targets Just-In-Time (JIT) during training, maintaining a footprint of < 100 GB for 25,000 dense chunks.

## Usage

The pipeline is structured around curriculum learning stages, starting with synthetic Gaussian profiles (Stage 0) before advancing to realistic `romanisim` data.

### 1. Data Pre-generation
Generate synthetic training and validation chunks:
```bash
python scripts/pregenerate_data.py 0 --config config/config.yaml
```

### 2. Training
Start or resume training a specific stage:
```bash
python scripts/run_stage.py 0 train --config config/config.yaml
```

### 3. Evaluation
Evaluate the model against strict Acceptance Criteria (Recall, Precision, Flux Ratio, Completeness MAE):
```bash
python scripts/run_stage.py 0 eval --config config/config.yaml --checkpoint checkpoints/stage0_epoch_20.pth
```

### 4. Inference & Visualization
Run inference on a generated chunk to produce a comprehensive diagnostic visualization (`inference_comparison.png`), including generative reconstruction and residual maps:
```bash
python scripts/run_stage.py 0 infer --config config/config.yaml --checkpoint checkpoints/stage0_epoch_20.pth
```

## Configuration
Hyperparameters, curriculum stages, and data dimensions are controlled via YAML configurations located in the `config/` directory. For local debugging, use `config/debug_config.yaml` or `config/medium_test.yaml`.
