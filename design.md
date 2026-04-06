# Architecture Design: Castor (Roman Point Source ML Pipeline)

## 1. Objective
To develop a machine learning pipeline capable of performing fast, direct point-source detection, photometry, and calibrated uncertainty estimation on Level 2 images from the Nancy Grace Roman Space Telescope. By framing the problem as a Dense Grid Prediction (inspired by YOLO/CenterNet), this architecture scales gracefully to handle extreme stellar densities while outputting a physical recoverability score and per-source measurement covariances via Aleatoric Uncertainty Estimation.

## 2. Input & Output Specifications

### Input (Edge-to-Edge Prediction)
*   **Format:** 2D Image Tensor
*   **Dimensions:** $256 \times 256 \times 1$ (Single-band image chunk).
*   **Preprocessing:** **"Dynamic Input, Residual Target" Strategy**. Each chunk is median-subtracted and normalized using a global Arcsinh stretch: $I_{norm} = \text{asinh}((I_{raw} - \text{median}(I_{raw})) / \text{scale})$. This ensures the noise floor always rides on the linear portion of the Arcsinh curve.

### Output (The Spatial Grid)
*   **Format:** 3D Tensor (Flattened Channels)
*   **Dimensions:** $64 \times 64 \times (K \times 7 + 1)$ (where $K=3$ star slots per cell).
*   **Structure:** The output is a $64 \times 64$ spatial grid (stride 4 relative to input). Each cell predicts star parameters for $K$ slots plus one shared local background value. **Canonical Slot Sorting:** Stars within each cell are sorted by flux (brightest to faintest) before being assigned to the $K$ slots. This provides a stable and consistent learning target.
*   **Slot Values (7 per slot):**
    1.  **p:** **SNR-Based Objectness ($0.0 \to 1.0$)**. Represents the physical detectability of the source.
        *   **Logarithmic Soft Labels:** Instead of binary 1/0 targets, the objectness target is mapped smoothly based on physical SNR using a Base-10 Logarithmic scale:
            *   $\text{SNR} \ge 5.0 \implies p = 1.0$
            *   $1.0 < \text{SNR} < 5.0 \implies p = \log_{10}(\text{SNR}) / \log_{10}(5)$
            *   $\text{SNR} \le 1.0 \implies p = 0.0$
        *   **Logit Bypass:** For training stability, the model outputs raw logits for $p$. Sigmoid is only applied during inference.
        *   **Sub-Pixel Confusion SNR:** The local SNR natively accounts for crowding. It is calculated by sampling the total simulated starlight at the sub-pixel coordinates, subtracting the star's own peak flux, and adding this residual "confusion light" to the sky background and read noise variance.
    2.  **dx, dy:** Sub-pixel offset from the cell's top-left corner ($0.0 \to \text{cell\_size}$).
    3.  **flux:** Physical Flux. Predicted via a **Log-Space Bypass**: The network predicts raw log-flux, which is clamped ($[-10.0, 22.0]$) and exponentiated to output physical flux.
    4.  **log_var_x, log_var_y:** Natural log of the astrometric variance ($\sigma_x^2, \sigma_y^2$).
    5.  **log_var_m:** Natural log of the photometric variance ($\sigma_m^2$ in log-flux space).
*   **Background Value (1 per cell):**
    1.  **b:** Residual Background Level. Represents local deviations from the chunk's median sky in stretched space.

## 3. Neural Network Architecture

### Stage 0: Trainable Physics Prior
Before entering the backbone, the raw input image passes through a **DiffractionAwareFilter (LoG)**.
*   **Filter Type:** Laplacian of Gaussian (Mexican Hat) wavelet.
*   **Kernel Size:** $21 \times 21$.
*   **Purpose:** Provides a mathematical prior optimized for blob detection and edge suppression. By concatenating the original image with this filter response, the network is immediately alerted to point-source structures vs. diffraction spikes or background gradients.
*   **Trainability:** The filter weights are initialized using the LoG formula (or the survey-average PSF mean) but remain trainable, allowing the model to "warp" the prior to perfectly match the unique diffraction profile of the telescope.

### Stage 1: The Backbone
*   **Backbone:** Full ResNet-34 (all 4 stages).
*   **Input:** $256 \times 256 \times 2$ (Original Stretched Image + Physics Prior Response).
*   **Multi-scale Features:** Extracts features at $1/4, 1/8, 1/16, 1/32$ resolutions.

### Stage 2: The FPN Neck
A **Feature Pyramid Network (FPN)** merges deep semantic context from the lower resolutions back into the high-resolution prediction grid.
*   **Top-down path:** Upsamples deep features and merges them with lateral high-res connections.
*   **Final Feature Map:** $64 \times 64 \times 128$ tensor (Stride 4 relative to input).

### Stage 3: The Prediction Head
*   **Spatial Awareness:** Uses **CoordConv** (normalized x,y coordinate channels) to help the model learn geometric dependencies within the grid cells.
*   **Output Layer:** $K \times 7 + 1$ channels.
*   **Activations:**
    *   **p:** Linear (Raw logits during training).
    *   **dx, dy:** Sigmoid $\times \text{cell\_size}$.
    *   **flux:** Bounded Exponential (Log-Flux Bypass).
    *   **log_vars:** Linear (Natural log space allows linear linear outputs to span $[-\infty, \infty]$ for stability).

## 4. The Loss Function: Aleatoric Uncertainty NLL
Instead of simple regression, the model minimizes the **Gaussian Negative Log-Likelihood (NLL)** for all measurement parameters.
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Pos\_NLL} + \lambda_3 \mathcal{L}_{Flux\_NLL} + \lambda_4 \mathcal{L}_{BG} + \lambda_5 \mathcal{L}_{DReg}$
*   **$\mathcal{L}_{Prob}$:** `BCEWithLogitsLoss` with Logarithmic Soft SNR Targets, combined with manual Focal Loss.
*   **NLL Regression:** For $dx, dy$ and $flux$, the network learns to balance two terms:
    1. **The Residual Term:** $\exp(-\ln \sigma^2) (\text{pred} - \text{target})^2$. If the network's prediction is far from the target, it can reduce this massive penalty by increasing the predicted variance.
    2. **The Regularization Term:** $\ln \sigma^2$. This prevents the network from just predicting infinite variance everywhere to zero out the residual term.
*   **$\mathcal{L}_{BG}$:** Global MSE for the background residuals.
*   **$\mathcal{L}_{DReg}$:** L2 regularization for the Diffraction-Aware Filter to prevent it from drifting too far from the initialization prior.

## 5. Success Metrics (Acceptance Criteria)
| Metric | Target | Description |
| :--- | :--- | :--- |
| **Recall (SNR > 10)** | $> 95\%$ | Successful detection of clear sources. |
| **Precision** | $> 98\%$ | Minimal false positives. |
| **Positional RMSE** | $< 0.15$ px | Sub-pixel coordinate accuracy. |
| **Uncertainty Calibration**| $\text{Error} / \sigma \approx 1.0$ | Statistical validity of predicted error bars. |
| **Flux Ratio (Mean)** | $1.00 \pm 0.05$ | Accuracy in magnitude recovery. |
| **Flux Scatter (StdDev)**| $< 0.10$ | Precision in magnitude recovery. |

## 6. Implementation Strategy: The Data Pipeline
To achieved maximum throughput and minimum disk footprint, the pipeline uses a hybrid rendering and standardization strategy.

*   **JAX-Accelerated Generation (Two-Tier Speed Hack):** To generate the massive Stage 0 mosaics (up to 8 million stars), the pipeline can use a fused JAX GPU operation (`lax.conv_general_dilated`). This allows for ultra-fast, sub-pixel accurate phase rendering of the point sources directly on the GPU.
*   **NumPy-Optimized Rendering:** For CPU-heavy environments or pre-training, generation is also driven by highly optimized NumPy and SciPy operations (e.g., `fftconvolve`, `map_coordinates`), using **Numba (@njit)** for fast sub-pixel grid painting.
*   **HDF5 Conversion with Compression:** Raw mosaics are converted into HDF5 files using **LZF compression** and **float32 precision** for targets. An incremental "sample-and-delete" strategy is used during conversion to minimize the peak disk footprint.
*   **JIT Live Noise:** During training, the PyTorch `Dataset` dynamically injects sky background, Poisson noise, and Gaussian read noise on the GPU, ensuring infinite noise realizations and preventing overfitting.

## 7. Training Curriculum
The pipeline uses a multi-stage curriculum to build a robust foundation model for space-based point source recovery.

### Stage 0: Gaussian Pre-training (The "Physics & Crowding Prior" Phase)
*   **Objective:** Teach the model the Dense Grid prediction format, sub-pixel localization, and crowding recoverability using Gaussian PSFs with global jitter simulation.
*   **Global Jitter Augmentation:** Introduces a dynamic "Global Jitter" pass applying asymmetric Gaussian convolution (parameterized by $s_{jit}$, $q_{jit}$, and $\theta_{jit}$) to simulate varying optical smear.
*   **Data:** Vectorized synthetic images generated via a NumPy/Numba or JAX renderer. Star fluxes and counts are drawn from a realistic Dynamic Bulge Luminosity Function.
*   **Goal:** Reach competency in detection and flux recovery in hyper-dense fields before introducing complex optical diffraction artifacts.

### Stage 1: Multi-Telescope Foundation Training (The "Universal Photometrist" Phase)
* **Objective:** Build instrument-agnostic features by training the FPN to handle diverse space-based and ground-based optical physics without overfitting to a single telescope's noise profile or diffraction geometry.
* **Data Generation:** An offline multiprocessing script uses **GalSim** to render a bank of massive clean "physics mosaics" representing four optical archetypes (Roman-like, Hubble-like, Ideal Space, and Ground-based).
* **Goal:** Learn to decouple the intrinsic stellar signal from varied instrumental PSFs, smoothly mapping core structures and naturally suppressing diffraction spikes before encountering Romanisim data.

### Stage 2: Roman-specific High-Fidelity Fine-tuning (The "Mission" Phase)
*   **Objective:** Master the specific artifacts and complex PSF of the Roman Space Telescope.
*   **Data:** Real mission-simulated data from **Romanisim** including geometric distortion, inter-pixel capacitance (IPC), and time-varying PSFs.
*   **Goal:** Exceed Mission Acceptance Criteria for the Galactic Bulge Time Domain Survey.
