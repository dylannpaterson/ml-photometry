# Castor: Roman Point Source ML Pipeline

**Castor** is the research and training engine for a specialized machine learning pipeline designed for fast, direct point-source detection, photometry, and recoverability estimation on Level 2 images from the Nancy Grace Roman Space Telescope.

## The Castor & Pollux Ecosystem

This project is part of a dual-pipeline architecture:

*   **Castor (This Repository):** The **Training & Research** suite. It focuses on curriculum learning, PSF shape modeling, and refining the Dense Grid Prediction network on $256 \times 256$ image chunks.
*   **Pollux (Sister Project):** The **Production Photometry Pipeline**. Pollux utilizes the models trained by Castor to perform automated, wide-field photometry on full-sized ($4088 \times 4088$) Roman L2 images by implementing intelligent tiling, hardware-accelerated inference, and global catalog stitching.

## Overview

Traditional point-spread function (PSF) fitting algorithms can be computationally prohibitive in extremely crowded fields, such as those targeted by the Roman Bulge Time Domain Survey (upwards of millions of stars per sensor chip assembly). This project frames point-source photometry as a **Dense Grid Prediction** problem (inspired by architectures like YOLO and CenterNet), allowing the network to simultaneously detect sources, measure their flux, estimate local background, and predict a recoverability score via SNR-based soft labels in a single forward pass.

### Key Capabilities
- **Simultaneous Detection & Photometry:** Predicts source probabilities ($p$) and log-transformed flux ($m$) directly.
- **Sub-pixel Localization:** Predicts fine spatial offsets ($dx, dy$) within the grid.
- **Generative PSF Recovery:** Learns eigen-PSF weights ($S$) for each detection, enabling full generative image reconstruction and residual analysis.
- **Background Modeling:** Predicts a smoothly varying 2D background surface ($b$) alongside the star catalog.
- **Continuous Objectness:** Uses SNR-based soft labels ($0.0 \to 1.0$) to represent physical recoverability, eliminating the need for binary detection thresholds during training and providing a physical confidence score.

## Architecture

The model processes $256 \times 256$ image chunks through a ResNet-34 backbone with an FPN neck, outputting a $64 \times 64$ spatial grid. Each cell represents a $4 \times 4$ pixel area and is capable of predicting up to $K=3$ overlapping point sources.

Each predicted star consists of parameters (e.g., **24 parameters** for $N_{PCA}=20$):
- $p$: SNR-Based Objectness score ($0.0 \to 1.0$)
- $dx, dy$: Sub-pixel offsets
- $m$: Natural Log Flux ($\ln(\text{Flux})$)
- $S$: $N_{PCA}$ values representing Eigen-PSF weights

Additionally, each $4 \times 4$ cell predicts a shared local background value ($b$).

## Data Storage Strategy

To maintain a virtually negligible disk footprint while preserving extremely high I/O throughput, the pipeline uses a **"Cached Physics, Live Noise"** dual-mmap architecture:

*   **Cached Physics, Live Noise:** Base optical physics (clean images) and catalogs are pre-rendered and saved to disk. During training, the PyTorch `Dataset` uses fast memory-mapping (`mmap`) to load random crops.
*   **JIT Densification:** Target grids are constructed Just-In-Time (JIT) in RAM with SNR-based soft labels.
*   **On-the-fly Noise:** Sky background, Poisson noise, and Gaussian read noise are injected dynamically upon loading, meaning the network never sees the exact same noise realization twice.

## Usage

The pipeline is structured around curriculum learning stages, starting with synthetic Gaussian profiles (Stage 0) before advancing to realistic multi-telescope and `romanisim` data.

### 1. Data Pre-generation
Generate synthetic training and validation chunks. For Stage 0, this utilizes a dynamic Galactic Bulge Luminosity Function and leverages JAX-accelerated GPU rendering to rapidly generate extremely crowded fields (up to 8 million stars).
```bash
python scripts/pregenerate_data.py 0 --config config/config.yaml
```

### 2. Training
Start or resume training a specific stage:
```bash
python scripts/run_stage.py 0 train --config config/config.yaml
```

### 3. Evaluation
Evaluate the model against strict Acceptance Criteria (Recall, Precision, Flux Ratio, Positional RMSE):
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
