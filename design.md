# Architecture Design: Castor (Roman Point Source ML Pipeline)

## 1. Objective
To develop a machine learning pipeline capable of performing fast, direct point-source detection, photometry, and completeness estimation on Level 2 images from the Nancy Grace Roman Space Telescope. By framing the problem as a Dense Grid Prediction (inspired by YOLO/CenterNet), this architecture scales gracefully to handle extreme stellar densities while outputting a physical recoverability score for every source.

## 2. Input & Output Specifications

### Input (Edge-to-Edge Prediction)
*   **Format:** 2D Image Tensor
*   **Dimensions:** $256 \times 256 \times 1$ (Single-band image chunk).
*   **Preprocessing:** **"Dynamic Input, Residual Target" Strategy**. Each chunk is median-subtracted and normalized using a global Arcsinh stretch: $I_{norm} = \text{asinh}((I_{raw} - \text{median}(I_{raw})) / \text{scale})$. This ensures the noise floor always rides on the linear portion of the Arcsinh curve.

### Output (The Spatial Grid)
*   **Format:** 3D Tensor (Flattened Channels)
*   **Dimensions:** $128 \times 128 \times (K \times 86 + 1)$ (where $K=3$ is optimized for Bulge densities).
*   **Structure:** The output is a $128 \times 128$ spatial grid. Each cell predicts star parameters for $K$ slots plus one shared local background value. **Canonical Slot Sorting:** Stars within each cell are sorted by flux (brightest to faintest) before being assigned to the $K$ slots. This provides a stable and consistent learning target.
*   **Slot Values (86 per slot):**
    1.  **p:** Probability (Objectness score, $0.0 \to 1.0$)
    2.  **dx, dy:** Sub-pixel offset from the cell's top-left corner ($0.0 \to 2.0$)
    3.  **m:** Stretched Flux ($\text{asinh}(\text{Flux} / \text{scale})$). Matches the input feature space for stable regression.
    4.  **c:** **Crowding-Aware Completeness Score ($0.0 \to 1.0$)**. Represents physical recoverability. 
        *   **Base:** Sigmoid-scaled SNR ($0.5$ at SNR 5.0).
        *   **Penalty:** Proximity-based suppression from brighter neighbors: $P = \prod (1 - \text{clip}(0.2 \cdot \frac{F_{bright}}{F_{target}} \cdot e^{-d/2}, 0, 0.8))$.
        *   **Purpose:** Honest labeling—tells the network which stars are physically "drowned out" to prevent unstable gradients from impossible-to-recover sources.
    5.  **S (Shape):** 9x9 Point Source Profile (81 values). Represents the isolated, centered PSF shape.
*   **Background Value (1 per cell):**
    1.  **b:** Residual Background Level ($\text{asinh}((\text{BG}_{raw} - \text{median}(I_{raw})) / \text{scale})$). Represents local deviations from the chunk's median sky.

## 3. Neural Network Architecture

### Stage 0: Trainable Physics Prior
Before entering the backbone, the raw input image passes through a **DiffractionAwareFilter (LoG)**.
*   **Filter Type:** Laplacian of Gaussian (Mexican Hat) wavelet.
*   **Kernel Size:** $21 \times 21$.
*   **Purpose:** Provides a mathematical prior optimized for blob detection and edge suppression. By concatenating the original image with this filter response, the network is immediately alerted to point-source structures vs. diffraction spikes or background gradients.
*   **Trainability:** The filter weights are initialized using the LoG formula but remain trainable, allowing the model to "warp" the prior to perfectly match the unique diffraction profile of the Roman PSF.

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
*   **Output Layer:** $K \times (5 + S^2) + 1$ channels (259 channels total for $K=3, S=9$).
*   **Activations:**
    *   **p, c:** Sigmoid.
    *   **dx, dy:** Sigmoid $\times \text{cell\_size}$.
    *   **m, b:** Linear (predicting in stretched space).
    *   **S:** Softmax over the $S^2$ values per slot.

## 4. The Loss Function
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Pos} + \lambda_3 \mathcal{L}_{Flux} + \lambda_4 \mathcal{L}_{Comp} + \lambda_5 \mathcal{L}_{Shape} + \lambda_6 \mathcal{L}_{BG}$
*   **$\mathcal{L}_{Prob}$:** Focal Loss with inverse-flux importance weighting to boost the detection of faint sources.
*   **$\mathcal{L}_{Pos}$:** Masked MSE for $dx, dy$ sub-pixel offsets, heavily weighted ($\lambda_2 \approx 50.0$) to force geometric precision.
*   **$\mathcal{L}_{Flux}$:** Masked MSE for stretched flux ($m$). Weighted ($\lambda_3 \approx 5.0$) to compensate for Arcsinh compression.
*   **$\mathcal{L}_{Comp}$:** Masked MSE for completeness ($c$).
*   **$\mathcal{L}_{Shape}$:** Masked MSE for the 9x9 PSF.
*   **$\mathcal{L}_{BG}$:** Global MSE for the Arcsinh-stretched background map.

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

## 6. Implementation Strategy: The Macro-Sparse Pipeline
To maintain a virtually negligible disk footprint (< 2 GB for 30 million simulated stars) while preserving extremely high I/O throughput, the pipeline uses a **"Cached Physics, Live Noise"** dual-mmap architecture combined with Just-In-Time (JIT) grid densification.

* **Macro-Sparse Storage:** Instead of saving massive 259-channel target tensors, the offline generator only saves:
    1.  **The Base Image:** A flat, clean float32 array containing the simulated optical physics (e.g., $4088 \times 4088$).
    2.  **The Target Catalog:** A lightweight Parquet or HDF5 table containing the ground truth for each star. 
* **Tabular Shape Integration ($S$):** The exact $9 \times 9$ optical PSF shape profile for every star is flattened into an 81-value array and saved directly as a column in the Target Catalog alongside its $x, y$, and `flux` coordinates.
* **JIT Densification & Live Noise:** During training, the PyTorch `Dataset`:
    1. Memory-maps the clean image and slices a random $256 \times 256$ crop.
    2. Dynamically injects sky background, Poisson noise, and Gaussian read noise on the fly, ensuring infinite noise realizations.
    3. Queries the catalog for stars within the crop bounds and instantly "paints" their $x, y, m$, and $S$ values into a dense $128 \times 128 \times 259$ target tensor in RAM.
* **Dynamic Completeness Calculation ($c$):** Because noise is injected live, the completeness/recoverability score cannot be pre-calculated. The dataloader calculates $c$ analytically on the fly for each star by computing its penalized Signal-to-Noise Ratio (factoring in the newly injected sky/read noise and local crowding from neighbor fluxes) and passing it through a logistic sigmoid centered on the detection threshold.

## 7. Training Curriculum
The pipeline uses a multi-stage curriculum to build a robust foundation model for space-based point source recovery.

### Stage 0: Gaussian Pre-training (The "Simple Physics" Phase)
*   **Objective:** Teach the model the grid-based prediction format using simple 2D Gaussian PSFs.
*   **Data:** Vectorized synthetic images with physically accurate Roman background levels and noise.
*   **Goal:** Reach basic competency in detection and flux recovery on smooth, well-defined sources.

### Stage 1: Multi-Telescope Foundation Training (The "Universal Photometrist" Phase)
* **Objective:** Build instrument-agnostic features by training the FPN to handle diverse space-based and ground-based optical physics without overfitting to a single telescope's noise profile or diffraction geometry.
* **Data Generation:** An offline multiprocessing script uses **GalSim** to render a bank of 20 massive $4088 \times 4088$ clean "physics mosaics" representing four optical archetypes:
    1. **Roman-like:** 6-strut, heavy diffraction.
    2. **Hubble-like:** 4-strut perpendicular diffraction.
    3. **Ideal Space:** Unobscured, pure Airy disks with varying aberrations (coma, astigmatism).
    4. **Ground-based:** Seeing-limited Moffat profiles simulating atmospheric blur.
* **Astrophysical Priors:** Star counts and fluxes (~150,000 per mosaic) are drawn from an empirical Galactic Bulge luminosity function (e.g., VVV or Besançon) to ensure authentic density ratios and severe crowding.
* **Training Mechanics:** Utilizing the Macro-Sparse Pipeline, the dataloader slices crops from the 20-mosaic bank, applies $D_4$ symmetry augmentations, and injects live noise. This provides ~50,000 unique, dynamically noisy training chunks for a 100-epoch curriculum.
* **Goal:** Learn to decouple the intrinsic stellar signal from varied instrumental PSFs, smoothly mapping core structures and naturally suppressing diffraction spikes before encountering Romanisim data.

### Stage 2: Roman-specific High-Fidelity Fine-tuning (The "Mission" Phase)
*   **Objective:** Master the specific artifacts and complex PSF of the Roman Space Telescope.
*   **Data:** Real mission-simulated data from **Romanisim** including geometric distortion, inter-pixel capacitance (IPC), and time-varying PSFs.
*   **Goal:** Exceed Mission Acceptance Criteria for the Galactic Bulge Time Domain Survey.
