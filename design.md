# Architecture Design: Castor (Roman Point Source ML Pipeline)

## 1. Objective
To develop a machine learning pipeline capable of performing fast, direct point-source detection, photometry, and completeness estimation on Level 2 images from the Nancy Grace Roman Space Telescope. By framing the problem as a Dense Grid Prediction (inspired by YOLO/CenterNet), this architecture scales gracefully to handle extreme stellar densities while outputting a physical recoverability score for every source.

## 2. Input & Output Specifications

### Input (Edge-to-Edge Prediction)
*   **Format:** 2D Image Tensor
*   **Dimensions:** $256 \times 256 \times 1$ (Single-band image chunk).
*   **Preprocessing:** Background subtraction is performed during training via a dedicated model head.

### Output (The Spatial Grid)
*   **Format:** 4D Tensor
*   **Dimensions:** $128 \times 128 \times (K \times 86 + 1)$ (where $K=3$ is optimized for Bulge densities).
*   **Structure:** The output is a $128 \times 128$ spatial grid. Each cell predicts star parameters for $K$ slots plus one shared local background value.
*   **Slot Values (86 per slot):**
    1.  **p:** Probability (Objectness score, $0.0 \to 1.0$)
    2.  **dx, dy:** Sub-pixel offset from the cell's top-left corner ($0.0 \to 2.0$)
    3.  **m:** Log-transformed Flux ($\log_{10}(\text{Flux})$).
    4.  **c:** Completeness / Recoverability Score ($0.0 \to 1.0$)
    5.  **S (Shape):** 9x9 Point Source Profile (81 values). Represents the isolated, centered PSF shape.
*   **Background Value (1 per cell):**
    1.  **b:** Local Background Level. Represents the smoothly varying sky/unresolved light surface.

## 3. Neural Network Architecture

### Stage 1: The Backbone
*   **Backbone:** Full ResNet-34 (all 4 stages).
*   **Input:** $256 \times 256 \times 1$.
*   **Multi-scale Features:** Extracts features at $1/4, 1/8, 1/16, 1/32$ resolutions.

### Stage 2: The FPN Neck
A **Feature Pyramid Network (FPN)** merges deep semantic context from the lower resolutions back into the high-resolution prediction grid.
*   **Top-down path:** Upsamples deep features and merges them with lateral high-res connections.
*   **Final Feature Map:** $64 \times 64 \times 128$ tensor (Stride 4 relative to input).

### Stage 3: The Prediction Head
*   **Spatial Awareness:** Uses **CoordConv** (normalized x,y coordinate channels) to help the model learn geometric dependencies within the grid cells.
*   **Output Layer:** $K \times (5 + S^2) + 1$ channels (259 channels total for $K=3, S=9$).
*   **Activations:**
    *   **p, c:** Sigmoid.
    *   **dx, dy:** Sigmoid $\times \text{cell\_size}$.
    *   **m:** Linear.
    *   **S:** Softmax over the $S^2$ values per slot.
    *   **b:** ReLU (ensures positive background levels).

## 4. The Loss Function
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Pos} + \lambda_3 \mathcal{L}_{Flux} + \lambda_4 \mathcal{L}_{Shape} + \lambda_5 \mathcal{L}_{BG} + \lambda_6 \mathcal{L}_{TV}$
*   **$\mathcal{L}_{Prob}$:** Focal Loss with inverse-flux importance weighting to boost the detection of faint sources.
*   **$\mathcal{L}_{Pos}$:** Masked MSE for $dx, dy$ sub-pixel offsets, heavily weighted ($\lambda_2 \approx 5.0$) to force geometric precision.
*   **$\mathcal{L}_{Flux}$:** Masked MSE for log-flux ($m$) and completeness ($c$).
*   **$\mathcal{L}_{Shape}$:** Masked MSE for the 9x9 PSF.
*   **$\mathcal{L}_{BG}$:** Global MSE for the background map.
*   **$\mathcal{L}_{TV}$:** Total Variation regularization on the predicted background map to enforce smoothness and prevent the model from "absorbing" star light into the background head.

## 5. Success Metrics (Acceptance Criteria)
| Metric | Target | Description |
| :--- | :--- | :--- |
| **Recall (SNR > 10)** | $> 95\%$ | Successful detection of clear sources. |
| **Precision** | $> 98\%$ | Minimal false positives. |
| **Positional RMSE** | $< 0.15$ px | Sub-pixel coordinate accuracy. |
| **Flux Ratio (Mean)** | $1.00 \pm 0.05$ | Accuracy in magnitude recovery. |
| **Flux Scatter (StdDev)**| $< 0.10$ | Precision in magnitude recovery. |
| **Completeness MAE** | $< 0.10$ | Reliability of predicted recoverability score. |
| **Shape Loss ($S$)** | $< 0.0001$ | PSF profile fidelity. |

## 6. Implementation Strategy: Sparse Storage
To maintain a < 100 GB disk footprint:
*   **Storage:** Samples save the image, the 5-channel star grid, the 1-channel background map, and a compressed list of 81-pixel shapes.
*   **JIT Re-densification:** The `Dataset` class re-inflates these into the full $128 \times 128 \times 259$ tensor during training.

## 7. Training Curriculum
The pipeline uses a multi-stage curriculum to build a robust foundation model for space-based point source recovery.

### Stage 0: Gaussian Pre-training (The "Simple Physics" Phase)
*   **Objective:** Teach the model the grid-based prediction format using simple 2D Gaussian PSFs.
*   **Data:** Vectorized synthetic images with physically accurate Roman background levels and noise.
*   **Goal:** Reach basic competency in detection and flux recovery on smooth, well-defined sources.

### Stage 1: Multi-Telescope Foundation Training (The "Universal Photometrist" Phase)
*   **Objective:** Build instrument-agnostic features by training on diverse space-based instruments.
*   **Data:** High-fidelity simulations using **GalSim** for:
    1. **Roman** (Wide-field, 0.11"/px)
    2. **Hubble (WFC3/IR)** (Diffraction-limited, 0.13"/px)
    3. **JWST (NIRCam)** (Hexagonal diffraction, 0.031"/px)
    4. **Euclid (VIS)** (Unique stable PSF, 0.1"/px)
    5. **Gaia-like** (Extremely sparse, super-resolution)
*   **Goal:** Learn to decouple the intrinsic stellar signal from the varied instrumental PSF and noise profiles.

### Stage 2: Roman-specific High-Fidelity Fine-tuning (The "Mission" Phase)
*   **Objective:** Master the specific artifacts and complex PSF of the Roman Space Telescope.
*   **Data:** Real mission-simulated data from **Romanisim** including geometric distortion, inter-pixel capacitance (IPC), and time-varying PSFs.
*   **Goal:** Exceed Mission Acceptance Criteria for the Galactic Bulge Time Domain Survey.
