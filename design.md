# Architecture Design: Roman Point Source ML Pipeline (Dense Grid Prediction)

## 1. Objective
To develop a machine learning pipeline capable of performing fast, direct point-source detection, photometry, and completeness estimation on Level 2 images from the Nancy Grace Roman Space Telescope. By framing the problem as a Dense Grid Prediction (inspired by YOLO/CenterNet), this architecture scales gracefully to handle extreme stellar densities while outputting a physical recoverability score for every source.

## 2. Input & Output Specifications

### Input (Edge-to-Edge Prediction)
*   **Format:** 2D Image Tensor
*   **Dimensions:** $256 \times 256 \times 1$ (Single-band image chunk).
*   **Strategy:** The network performs point source detection across the entire input area. To handle detector-wide catalogs, large images are tiled into $256 \times 256$ chunks with a small overlap (e.g., 16 pixels) to ensure stars on the boundaries are captured correctly.
*   **Preprocessing:** Minimal to none. Ingest standard Level 2 slope images natively.

### Output (The Spatial Grid)
*   **Format:** 4D Tensor
*   **Dimensions:** $128 \times 128 \times K \times 54$ (where $K=3$ is the optimized capacity for Bulge densities).
*   **Structure:** The output is a $128 \times 128$ spatial grid mapping directly to the $256 \times 256$ input. Each cell is responsible for finding stars within a specific $2 \times 2$ pixel region.
*   **Slot Values (54):**
    1.  **p:** Probability (Objectness score, $0.0 \to 1.0$)
    2.  **dx:** Sub-pixel offset from the cell's top-left corner ($0.0 \to 2.0$)
    3.  **dy:** Sub-pixel offset from the cell's top-left corner ($0.0 \to 2.0$)
    4.  **m:** Instrumental magnitude or normalized flux
    5.  **c:** Completeness / Recoverability Score ($0.0 \to 1.0$)
    6.  **S (Shape):** 7x7 Point Source Profile (49 values). Represents the isolated, field-dependent PSF shape, exactly centered within the 7x7 window to decouple shape estimation from sub-pixel localization.

## 3. Neural Network Architecture

### Stage 1: The Backbone (Feature Extractor)
A Fully Convolutional Neural Network (CNN) processes the $256 \times 256$ input.
*   **Recommendation:** A standard ResNet-34 or ConvNeXt backbone.
*   **Output:** A feature map geometrically downsampled by a factor of 2. The output dimensions are exactly $128 \times 128 \times C$.

### Stage 2: The Grid Prediction Head
The $128 \times 128$ feature map is passed through $1 \times 1$ convolutional layers to map the channel depth directly to the required outputs.
*   **Output Layer:** A $1 \times 1$ Conv layer outputting $K \times 54$ channels (e.g., 162 channels for $K=3$). The tensor is then reshaped to $128 \times 128 \times 3 \times 54$.
*   **Activations:**
    *   **p (Probability):** Sigmoid (0 to 1).
    *   **dx, dy (Offsets):** Sigmoid multiplied by 2 (constrains predictions to the local cell).
    *   **m (Magnitude):** Linear or ReLU.
    *   **c (Completeness):** Sigmoid (0 to 1).
    *   **S (Shape):** Softmax over the 49 values per slot. This ensures the predicted profile is normalized (sums to 1.0), forcing the `m` parameter to handle all flux scaling.

## 4. The Loss Function (Grid Assignment)

### Step A: Ground Truth Grid Assignment
1.  Create a target tensor of zeros: $128 \times 128 \times 3 \times 54$.
2.  For each real star in the $256 \times 256$ chunk, calculate which $2 \times 2$ grid cell it falls into.
3.  Calculate its local dx, dy offset within that cell.
4.  Assign this star to the first available $K$-slot in that specific grid cell in the target tensor.

### Step B: The Masked Loss
*   **Probability Loss (Every cell, every slot):** Focal Loss (FL) between predicted p and target p. Focal Loss is used to address the extreme class imbalance between star-filled and empty grid cells, and to force the model to focus on "hard" examples (faint, low-SNR stars).
    $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
*   **Regression Loss (Masked):** Mean Squared Error (MSE) for dx, dy, m, and c.
*   **Shape Loss (Masked):** Mean Squared Error (MSE) between the predicted 7x7 normalized profile and the ground-truth PSF cutout.
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Reg} + \lambda_3 \mathcal{L}_{Shape}$

## 5. Training Data Strategy
*   **Generation:** Synthetic Roman images generated via `romanisim` utilizing spatially varying PSFs.
*   **Extraction:** Render a $384 \times 384$ image chunk. Extract the true catalog only for the stars located in the inner $256 \times 256$ core.
*   **Completeness Ground Truth (The Local SNR Calculation):** To generate the target c (Completeness) value for each injected star, use `romanisim`'s intermediate "idealized counts" image (which contains the exact pre-noise photon counts per pixel):
    1.  For a given star, extract a $3 \times 3$ pixel patch centered on its $(x, y)$ coordinate from the idealized image to find the true local background + overlapping star flux ($B$).
    2.  Use the standard astronomical CCD equation:
        $$\text{SNR} = \frac{S}{\sqrt{S + B + D + N_R^2}}$$
        (Where $S$ is the star's pure signal, $D$ is the Roman dark current, and $N_R$ is the Roman read noise).
    3.  Map the calculated SNR to a $0.0 \to 1.0$ completeness target score (e.g., clipping at $\text{SNR} < 3 = 0.0$ and saturating at $\text{SNR} > 10 = 1.0$).
    4.  Place this value into the 5th column (c) of the target tensor.
*   **Strict Augmentation Rules:** Adding synthetic noise and sub-pixel shifting is allowed. No rotations or flips, as this violates the fixed orientation of Roman's diffraction spikes.

## 6. Current Implementation: Cloud Lifecycle
The current pipeline is optimized for high-throughput training on GCP (T4/L4 GPU) using:
*   **Pregenerated Datasets:** To eliminate the CPU bottleneck of on-the-fly synthetic data generation, use `scripts/pregenerate_data.py`. 
*   **Sparse Target Storage:** To handle the increased dimensionality of the 7x7 shape while staying within a **100 GB disk limit**, target tensors are stored sparsely on disk. Each sample consists of the input image, the 5-channel base grid, and a compressed list of the 49-value shapes for active stars only.
*   **JIT Re-densification:** The `Dataset` class re-densifies the sparse shape data into the full $128 \times 128 \times 3 \times 54$ tensor during the `__getitem__` call. This move minimizes disk space (~65GB total for 25,000 samples) while ensuring minimal CPU overhead during training.
*   **Cloud Lifecycle Management:** `scripts/deploy_cloud.sh` manages the full VM lifecycle:
    1.  **Sync & Setup:** Pulls latest code and installs dependencies.
    2.  **Auto-Pregen:** Automatically detects if the data directory is empty and runs pre-generation.
    3.  **Auto-Shutdown:** Launches training and triggers `sudo poweroff` upon completion.
    4.  **Result Retrieval:** Use `./scripts/deploy_cloud.sh get-results` to automatically start the VM, download models/logs, and shut down.

## 7. Scaling to Realistic Data (Romanisim Integration)

As the pipeline matures, the synthetic Gaussian sources will be replaced with high-fidelity simulations from `romanisim` to capture the telescope's complex PSF variations, detector effects, and varied backgrounds.

### 7.1 Data Pregeneration Strategy
To maximize training efficiency, `romanisim` data must be pregenerated and stored on disk. Generating these complex simulations on-the-fly during training is computationally prohibitive.
*   **Format:** Save chunks as `.npy` or HDF5 files to bypass the overhead of FITS headers during the training loop.
*   **Target Volume:**
    *   **Pilot Phase:** 1,000 chunks (for pipeline debugging).
    *   **Base Phase:** 5,000 chunks (strong baseline for CPU/GPU training).
    *   **Production Phase:** 15,000 - 20,000 chunks (~15-20 million individual stars). This ensures complete coverage of the PSF variation across all 18 SCAs and the full field of view.

### 7.2 Training Curriculum
A "Curriculum Learning" approach is recommended to ensure the model converges efficiently:
1.  **Stage 0 (Gaussian Pre-training):** Train the model on high-density Gaussian sources (the current stage). This allows the network to learn the fundamental geometry of grid assignment and sub-pixel localization on "clean" data.
2.  **Stage 1 (Ideal PSF Fine-tuning):** Load the Gaussian weights and fine-tune on pregenerated `romanisim` chunks with static PSFs and uniform backgrounds using a reduced learning rate.
3.  **Stage 2 (Realistic Variation):** Introduce spatially varying PSFs (WFI-wide) and realistic Bulge survey backgrounds.
4.  **Stage 3 (Detector Effects):** Final fine-tuning on data including cosmic rays, persistence, and non-linearity.

### 7.3 Storage and Memory
At $384 \times 384$ resolution (float32), a 20,000-chunk dataset requires approximately **8-10 GB** of storage. This footprint is small enough to be cached entirely in System RAM on modern compute nodes, enabling extremely high-throughput data loading.
