import torch
import numpy as np
import os
from scipy.spatial import cKDTree
from castor.data.transforms import AstroSpaceTransform
from castor.constants import MAX_CAPACITY_PER_CELL, GLOBAL_STRETCH_SCALE

def match_stars(true_stars, pred_stars, distance_threshold=1.0, flux_threshold_dex=0.5):
    """
    Matches predicted stars to true stars using a Bright-to-Faint priority strategy.
    
    The cost function balances positional error and magnitude difference:
    Cost = (dist / dist_thresh)**2 + (abs(log_flux_diff) / flux_thresh_dex)**2

    Parameters
    ----------
    true_stars : list of tuple
        Ground truth stars as (x, y, flux, ...).
    pred_stars : list of tuple
        Predicted stars as (x, y, flux, ...).
    distance_threshold : float, optional
        Maximum allowed distance for a match in pixels, by default 1.0.
    flux_threshold_dex : float, optional
        Normalization factor for flux difference in dex (log10), by default 0.5.

    Returns
    -------
    tuple
        A tuple (matches, unmatched_true_indices, unmatched_pred_indices). 
        `matches` is a list of (true_idx, pred_idx, cost).
    """
    if not pred_stars:
        return [], list(range(len(true_stars))), []
    if not true_stars:
        return [], [], list(range(len(pred_stars)))

    # Convert to arrays for efficiency
    true_coords = np.array([(s[0], s[1]) for s in true_stars])
    true_fluxes = np.array([s[2] for s in true_stars])
    
    pred_coords = np.array([(s[0], s[1]) for s in pred_stars])
    pred_fluxes = np.array([s[2] for s in pred_stars])

    # Pre-calculate log fluxes
    log_true = np.log10(np.maximum(1e-9, true_fluxes))
    log_pred = np.log10(np.maximum(1e-9, pred_fluxes))

    # Search tree for predictions
    tree = cKDTree(pred_coords)
    
    # 1. Sort True Stars by Flux (Brightest First)
    true_indices_sorted = np.argsort(true_fluxes)[::-1]
    
    matches = []
    matched_true = set()
    matched_pred = set()

    for t_idx in true_indices_sorted:
        # Find all predictions within the search radius
        p_indices = tree.query_ball_point(true_coords[t_idx], r=distance_threshold)
        
        best_p_idx = -1
        min_cost = float('inf') 

        for p_idx in p_indices:
            if p_idx in matched_pred:
                continue
                
            dist = np.sqrt(np.sum((true_coords[t_idx] - pred_coords[p_idx])**2))
            flux_diff = abs(log_true[t_idx] - log_pred[p_idx])
            
            # Hybrid Cost Function
            # Each component is normalized by its respective threshold
            cost = (dist / distance_threshold)**2 + (flux_diff / flux_threshold_dex)**2
            
            if cost < min_cost:
                min_cost = cost
                best_p_idx = p_idx
        
        if best_p_idx != -1:
            matches.append((t_idx, best_p_idx, min_cost))
            matched_true.add(t_idx)
            matched_pred.add(best_p_idx)

    unmatched_true = [i for i in range(len(true_stars)) if i not in matched_true]
    unmatched_pred = [i for i in range(len(pred_stars)) if i not in matched_pred]

    return matches, unmatched_true, unmatched_pred

class Evaluator:
    """
    Handles model evaluation and metric reporting.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model.
    device : torch.device
        The device to run evaluation on.
    config : dict
        Configuration dictionary.
    stage_idx : int
        The training stage index.
    K : int
        Maximum number of stars per cell.
    stretch_scale : float
        The scale used for arcsinh flux stretching.
    transform : AstroSpaceTransform
        The transform used for image preprocessing.
    """

    def __init__(self, model, device, config, stage_idx=0):
        """
        Initialize the Evaluator.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model.
        device : torch.device
            The device to use.
        config : dict
            The configuration dictionary.
        stage_idx : int, optional
            The stage index, by default 0.
        """
        self.model = model
        self.device = device
        self.config = config
        self.stage_idx = stage_idx
        self.K = config["data_params"].get("max_capacity_per_cell", MAX_CAPACITY_PER_CELL)
        self.stretch_scale = config["data_params"].get("GLOBAL_STRETCH_SCALE", GLOBAL_STRETCH_SCALE)
        self.transform = AstroSpaceTransform(stretch_scale=self.stretch_scale)

    def run_evaluation(self, num_chunks=100, threshold=0.5):
        """
        Runs the evaluation suite on a set number of chunks.

        Parameters
        ----------
        num_chunks : int, optional
            Number of chunks to evaluate, by default 100.
        threshold : float, optional
            Detection threshold, by default 0.5.
        """
        print(f"Evaluating model on {num_chunks} chunks...")
        self.model.eval()
        
        all_tp, all_fp, all_fn = 0, 0, 0
        pos_errors, ratios = [], []
        
        # Stage-specific data generation
        if self.stage_idx == 0:
            from castor.data.stage0_gaussian import HDF5ChunkDataset
            data_cfg = self.config["data_params"]
            stage_cfg = self.config["curriculum"]["stage0"]
            val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
            
            if not os.path.exists(val_h5):
                print(f"❌ Error: Validation HDF5 not found at {val_h5}. Run 'train' first.")
                return
                
            dataset = HDF5ChunkDataset(val_h5)
            num_to_eval = min(num_chunks, len(dataset))
            
            for i in range(num_to_eval):
                sample = dataset[i]
                image_tensor = sample["image"]
                target_grid = sample["target"]
                
                # --- Apply Live Noise and Stretch (Match Training) ---
                img_pos = torch.clamp(image_tensor, min=0.0)
                img_noisy = torch.poisson(img_pos)
                img_noisy += torch.randn_like(img_noisy) * 5.0  # Read noise
                
                # USE STABLE MEDIAN (Match Trainer)
                stable_median = sample["chunk_median"]
                
                # Apply the Arcsinh stretch (Network Space)
                img_stretched = torch.arcsinh((img_noisy - stable_median) / self.stretch_scale)
                # -----------------------------------------------------

                # Predict
                with torch.no_grad():
                    input_tensor = img_stretched.unsqueeze(0).to(self.device)
                    # Ensure channel dimension [B, 1, H, W]
                    if input_tensor.dim() == 3:
                        input_tensor = input_tensor.unsqueeze(1)
                        
                    # FIX: Match Training Mixed Precision Context
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        prediction_dict = self.model(input_tensor)
                    
                    # FIX: Cast back to numpy via CPU and float32
                    prediction = prediction_dict["stars"].squeeze(0).float().cpu().numpy()
                
                # Extract True Stars
                true_stars = []
                grid_h, grid_w = target_grid.shape[:2]
                K = self.K
                
                # Target in HDF5 is (grid_size, grid_size, K * (4 + N_PCA) + 1)
                target_reshaped = target_grid[..., :-1].view(grid_h, grid_w, K, -1).numpy()
                
                cell_size = dataset.cell_size
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K):
                            slot = target_reshaped[y, x, k]
                            tp, tdx, tdy, raw_flux_target = slot[:4]
                            # NEW: For evaluation, we only consider stars with high SNR labels as targets (SNR >= 2.0)
                            if tp >= 0.43:
                                star_info = ((x * cell_size) + tdx, (y * cell_size) + tdy, float(raw_flux_target))
                                true_stars.append(star_info)
                
                # Extract Predicted Stars (p > threshold)
                pred_stars = []
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K):
                            p, dx, dy, physical_flux_pred = prediction[y, x, k, :4]
                            if p > threshold:
                                pred_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux_pred), p))
                
                matches, unmatched_true, unmatched_pred = match_stars(true_stars, pred_stars, distance_threshold=1.0)
                
                all_tp += len(matches)
                all_fp += len(unmatched_pred)
                all_fn += len(unmatched_true)

                for t_idx, p_idx, cost in matches:
                    dist = np.sqrt(np.sum((np.array(true_stars[t_idx][:2]) - np.array(pred_stars[p_idx][:2]))**2))
                    pos_errors.append(dist)
                    
                    t_flux = true_stars[t_idx][2]
                    p_flux = pred_stars[p_idx][2]
                    
                    ratios.append(p_flux / (t_flux + 1e-9))
                    
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        rmse = np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else 1.0
        flux_accuracy = np.median(ratios) if ratios else 0
        flux_scatter = np.std(ratios) if ratios else 0

        print("\n=============================================")
        print(" STAGE 0 ACCEPTANCE CRITERIA CHECK")
        print("=============================================")
        self._print_metric("Recall (Visible Stars)", recall, 0.95)
        self._print_metric("Precision", precision, 0.98)
        self._print_metric("Positional RMSE", rmse, 0.15, reverse=True)
        self._print_metric("Flux Ratio Accuracy", flux_accuracy, 0.95)
        self._print_metric("Flux Scatter (StdDev)", flux_scatter, 0.10, reverse=True)
        print("---------------------------------------------")
        
        if recall > 0.9 and precision > 0.9:
            print("\n✅ MODEL IS LOOKING GOOD!")
        else:
            print("\n⚠️ MODEL NEEDS MORE TRAINING OR CALIBRATION.")
        print("=============================================\n")

    def _print_metric(self, name, value, target, reverse=False):
        """Prints a single metric with a pass/fail indicator."""
        status = "✅"
        if reverse:
            if value > target: status = "❌"
        else:
            if value < target: status = "❌"
        print(f"{status} {name:<23}:   {value:.4f} (Target: {target:>8.4f})")
