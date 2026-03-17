# Architecture Design: Roman Point Source ML Pipeline (Dense Grid Prediction)

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
*   **Recommendation:** ResNet-34.
*   **Output:** $128 \times 128 \times 64$ feature map.

### Stage 2: The Grid Prediction Head
The feature map is passed through a $3 \times 3$ convolutional layer followed by a $1 \times 1$ projection layer.
*   **Output Layer:** $K \times 86 + 1$ channels (259 channels total for $K=3$).
*   **Activations:**
    *   **p, c:** Sigmoid.
    *   **dx, dy:** Sigmoid $\times 2.0$.
    *   **m:** Linear.
    *   **S:** Softmax over the 81 values per slot.
    *   **b:** ReLU (ensures positive background levels).

## 4. The Loss Function
*   **Total Loss:** $\mathcal{L}_{Total} = \lambda_1 \mathcal{L}_{Prob} + \lambda_2 \mathcal{L}_{Reg} + \lambda_3 \mathcal{L}_{Shape} + \lambda_4 \mathcal{L}_{BG}$
*   **$\mathcal{L}_{BG}$:** Global MSE between predicted background map and ground truth background.

## 5. Success Metrics (Acceptance Criteria)
| Metric | Target | Description |
| :--- | :--- | :--- |
| **Recall (SNR > 10)** | $> 95\%$ | Successful detection of clear sources. |
| **Precision** | $> 98\%$ | Minimal false positives. |
| **Flux Ratio (Mean)** | $1.00 \pm 0.05$ | Accuracy in magnitude recovery. |
| **Flux Scatter (StdDev)**| $< 0.10$ | Precision in magnitude recovery. |
| **Shape Loss ($S$)** | $< 0.0001$ | PSF profile fidelity. |

## 6. Implementation Strategy: Sparse Storage
To maintain a < 100 GB disk footprint:
*   **Storage:** Samples save the image, the 5-channel star grid, the 1-channel background map, and a compressed list of 81-pixel shapes.
*   **JIT Re-densification:** The `Dataset` class re-inflates these into the full $128 \times 128 \times 259$ tensor during training.
