# Architecture Design: Castor (Roman Point Source ML Pipeline)

## 1. Objective
To develop a machine learning pipeline capable of performing fast, direct point-source detection, photometry, and recoverability estimation on Level 2 images from the Nancy Grace Roman Space Telescope. By framing the problem as a Dense Grid Prediction (inspired by YOLO/CenterNet), this architecture scales gracefully to handle extreme stellar densities while outputting a physical recoverability score for every source via continuous objectness labels.

## 2. Input & Output Specifications

### Input (Edge-to-Edge Prediction)
*   **Format:** 2D Image Tensor
*   **Dimensions:** $256 \times 256 \times 1$ (Single-band image chunk).
*   **Preprocessing:** **"Dynamic Input, Residual Target" Strategy**. Each chunk is median-subtracted and normalized using a global Arcsinh stretch: $I_{norm} = \text{asinh}((I_{raw} - \text{median}(I_{raw})) / \text{scale})$. This ensures the noise floor always rides on the linear portion of the Arcsinh curve.

### Output (The Spatial Grid)
*   **Format:** 3D Tensor (Flattened Channels)
*   **Dimensions:** $64 \times 64 \times (K \times (4 + N_{PCA}) + 1)$ (where $K=3$ and $N_{PCA}=20$).
*   **Structure:** The output is a $64 \times 64$ spatial grid (stride 4 relative to input). Each cell predicts star parameters for $K$ slots plus one shared local background value. **Canonical Slot Sorting:** Stars within each cell are sorted by flux (brightest to faintest) before being assigned to the $K$ slots. This provides a stable and consistent learning target.
*   **Slot Values (24 per slot for $N_{PCA}=20$):**
    1.  **p:** **SNR-Based Objectness ($0.0 \to 1.0$)**. Represents the physical detectability of the source.
        *   **Continuous Soft Labels:** Instead of binary 1/0 targets, the objectness target is mapped smoothly based on physical SNR:
            *   $\text{SNR} \ge 5.0 \implies p = 1.0$
            *   $\text{SNR} = 3.0 \implies p = 0.5$
            *   $\text{SNR} \le 1.0 \implies p = 0.0$
        *   **Logit Bypass:** For training stability, the model outputs raw logits for $p$. Sigmoid is only applied during inference.
        *   **Sub-Pixel Confusion SNR:** The local SNR natively accounts for crowding. It is calculated by sampling the total simulated starlight at the exact sub-pixel coordinates, subtracting the star's own peak flux, and adding this residual "confusion light" to the sky background and read noise variance.
    2.  **dx, dy:** Sub-pixel offset from the cell's top-left corner ($0.0 \to \text{cell\_size}$).
    3.  **m:** Natural Log Flux ($\ln(\text{Flux} + 1e^{-6})$). Predicts bounded log-flux (clamped between -10.0 and 22.0) for numerical stability.
    4.  **S (Shape):** Eigen-PSF PCA Weights ($N_{PCA}$ values). Continuous weights that are combined with a predefined global PCA basis and mean PSF to reconstruct high-fidelity $31 \times 31$ point source profiles.
*   **Background Value (1 per cell):**
    1.  **b:** Residual Background Level. Represents local deviations from the chunk's median sky in stretched space.

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
*   **Output Layer:** $K \times (4 + N_{PCA}) + 1$ channels (73 channels total for $K=3, N_{PCA}=20$).
*   **Activations:**
    *   **p:** Linear (Raw logits during training, Sigmoid during inference).
    *   **dx, dy:** Sigmoid $\times \text{cell\_size}$.
    *   **m:** Linear (Bounded Log-Flux).
    *   **S:** Linear (Continuous PCA Weights).

## 4. The Loss Function
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Pos} + \lambda_3 \mathcal{L}_{Flux} + \lambda_4 \mathcal{L}_{Shape} + \lambda_5 \mathcal{L}_{BG} + \lambda_6 \mathcal{L}_{DReg}$
*   **$\mathcal{L}_{Prob}$:** `BCEWithLogitsLoss` with Soft SNR Targets, combined with manual Focal Loss and inverse-flux importance weighting to boost the detection of faint sources.
*   **$\mathcal{L}_{Pos}$:** Masked Smooth L1 Loss for $dx, dy$ sub-pixel offsets, heavily weighted ($\lambda_2 \approx 50.0$) to force geometric precision.
*   **$\mathcal{L}_{Flux}$:** Masked Smooth L1 Loss evaluated in natural log space ($\ln(\text{Flux} + 1e^{-6})$).
*   **$\mathcal{L}_{Shape}$:** **Eigen-PSF Reconstruction Loss**. Predicted weights are multiplied by the PCA basis and added to the mean PSF to reconstruct the $31 \times 31$ profile. Masked Smooth L1 is then evaluated in the reconstructed pixel space to guarantee photometric consistency.
*   **$\mathcal{L}_{BG}$:** Global MSE for the background residuals.
*   **$\mathcal{L}_{DReg}$:** L2 regularization for the Diffraction-Aware Filter to prevent it from drifting too far from the LoG kernel prior.

## 5. Success Metrics (Acceptance Criteria)
| Metric | Target | Description |
| :--- | :--- | :--- |
| **Recall (SNR > 10)** | $> 95\%$ | Successful detection of clear sources. |
| **Precision** | $> 98\%$ | Minimal false positives. |
| **Positional RMSE** | $< 0.15$ px | Sub-pixel coordinate accuracy. |
| **Flux Ratio (Mean)** | $1.00 \pm 0.05$ | Accuracy in magnitude recovery. |
| **Flux Scatter (StdDev)**| $< 0.10$ | Precision in magnitude recovery. |
| **Shape Loss ($S$)** | $< 0.0001$ | PSF profile fidelity (Pixel MSE). |

## 6. Implementation Strategy: The Macro-Sparse Pipeline
To maintain a virtually negligible disk footprint while preserving extremely high I/O throughput, the pipeline uses a **"Cached Physics, Live Noise"** architecture combined with compressed HDF5 storage.

*   **JAX-Accelerated Generation (Two-Tier Speed Hack):** To generate the massive Stage 0 mosaics (up to 8 million stars), the pipeline uses a fused JAX GPU operation (`lax.conv_general_dilated`). This allows for ultra-fast, sub-pixel accurate phase rendering of the point sources directly on the GPU.
*   **Eigen-PSF Storage:** Instead of saving raw pixel grids for every star, the offline generator performs a native PyTorch PCA on the PSF library once per mosaic. It then saves:
    1.  **The Base Image:** A flat, clean float32 array containing the simulated optical physics.
    2.  **PCA Basis & Mean:** The 20 principal components and mean PSF (as small $20 \times 961$ and $1 \times 961$ arrays).
    3.  **Target Catalog:** Lightweight table containing ground truth $x, y, \ln(\text{Flux})$, and the 20 continuous PCA weights per star. 
*   **HDF5 Conversion with Compression:** Raw mosaics are converted into HDF5 files using **LZF compression** and **float32 precision** for targets. An incremental "sample-and-delete" strategy is used during conversion to minimize the peak disk footprint.
*   **JIT Live Noise:** During training, the PyTorch `Dataset` dynamically injects sky background, Poisson noise, and Gaussian read noise on the GPU, ensuring infinite noise realizations and preventing overfitting to specific noise patterns.

## 7. Training Curriculum
The pipeline uses a multi-stage curriculum to build a robust foundation model for space-based point source recovery.

### Stage 0: Gaussian Pre-training (The "Physics & Crowding Prior" Phase)
*   **Objective:** Teach the model the Dense Grid prediction format, sub-pixel localization, and crowding recoverability using clean Gaussian PSFs.
*   **Data:** Vectorized synthetic images generated via a JAX-accelerated renderer. Star fluxes and counts are drawn from a realistic Dynamic Bulge Luminosity Function (featuring a continuous exponential Main Sequence/RGB with a density-anchored Red Clump prior) to simulate extreme mission-level crowding (1M to 8M stars per full mosaic).
*   **Goal:** Reach competency in detection and flux recovery in hyper-dense fields before introducing complex optical diffraction artifacts.

### Stage 1: Multi-Telescope Foundation Training (The "Universal Photometrist" Phase)
* **Objective:** Build instrument-agnostic features by training the FPN to handle diverse space-based and ground-based optical physics without overfitting to a single telescope's noise profile or diffraction geometry.
* **Data Generation:** An offline multiprocessing script uses **GalSim** to render a bank of massive clean "physics mosaics" representing four optical archetypes:
    1. **Roman-like:** 6-strut, heavy diffraction.
    2. **Hubble-like:** 4-strut perpendicular diffraction.
    3. **Ideal Space:** Unobscured, pure Airy disks with varying aberrations (coma, astigmatism).
    4. **Ground-based:** Seeing-limited Moffat profiles simulating atmospheric blur.
* **Astrophysical Priors:** Star counts and fluxes drawn from an empirical Galactic Bulge luminosity function.
* **Training Mechanics:** Slices crops from the mosaic bank, applies $D_4$ symmetry augmentations, and injects live noise. 
* **Goal:** Learn to decouple the intrinsic stellar signal from varied instrumental PSFs, smoothly mapping core structures and naturally suppressing diffraction spikes before encountering Romanisim data.

### Stage 2: Roman-specific High-Fidelity Fine-tuning (The "Mission" Phase)
*   **Objective:** Master the specific artifacts and complex PSF of the Roman Space Telescope.
*   **Data:** Real mission-simulated data from **Romanisim** including geometric distortion, inter-pixel capacitance (IPC), and time-varying PSFs.
*   **Goal:** Exceed Mission Acceptance Criteria for the Galactic Bulge Time Domain Survey.
