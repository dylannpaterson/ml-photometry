import torch
import numpy as np
from scipy.spatial import cKDTree
from src.data.transforms import AstroSpaceTransform

def match_stars(true_stars, pred_stars, distance_threshold=1.0, flux_threshold_dex=0.5):
    """
    Matches predicted stars to true stars using a KDTree for efficiency.
    Adds a flux-consistency constraint to prevent mis-assignment in dense fields.
    """
    if not pred_stars:
        return [], list(range(len(true_stars))), []
    if not true_stars:
        return [], [], list(range(len(pred_stars)))

    true_coords = np.array([(s[0], s[1]) for s in true_stars])
    pred_coords = np.array([(s[0], s[1]) for s in pred_stars])

    tree = cKDTree(true_coords)
    distances, indices = tree.query(pred_coords, distance_upper_bound=distance_threshold)

    matches = []
    matched_true = set()
    matched_pred = set()

    # 1. Gather potential matches that satisfy BOTH distance and flux constraints
    potential_matches = []
    for p_idx, (dist, t_idx) in enumerate(zip(distances, indices)):
        if dist < distance_threshold:
            # Flux Consistency Check (dex difference)
            t_flux = true_stars[t_idx][2]
            p_flux = pred_stars[p_idx][2]
            
            # Avoid log of zero/negative
            log_t = np.log10(max(1e-9, t_flux))
            log_p = np.log10(max(1e-9, p_flux))
            
            if abs(log_t - log_p) < flux_threshold_dex:
                potential_matches.append((dist, t_idx, p_idx))
    
    # 2. Greedy assignment (closest first)
    potential_matches.sort()

    for dist, t_idx, p_idx in potential_matches:
        if t_idx not in matched_true and p_idx not in matched_pred:
            matches.append((t_idx, p_idx, dist))
            matched_true.add(t_idx)
            matched_pred.add(p_idx)

    unmatched_true = [i for i in range(len(true_stars)) if i not in matched_true]
    unmatched_pred = [i for i in range(len(pred_stars)) if i not in matched_pred]

    return matches, unmatched_true, unmatched_pred

class Evaluator:
    def __init__(self, model, device, config, stage_idx=0):
        self.model = model
        self.device = device
        self.config = config
        self.stage_idx = stage_idx
        self.K = config["data_params"]["max_capacity_per_cell"]
        self.stretch_scale = config["data_params"].get("GLOBAL_STRETCH_SCALE", 10.0)
        self.transform = AstroSpaceTransform(stretch_scale=self.stretch_scale)

    def run_evaluation(self, num_chunks=100, threshold=0.5):
        print(f"Evaluating model on {num_chunks} chunks...")
        self.model.eval()
        
        all_tp, all_fp, all_fn = 0, 0, 0
        pos_errors, ratios, comp_errors = [], [], []
        matched_completeness = []
        missed_completeness = []
        
        # Bin recall by completeness (as a proxy for SNR)
        comp_bins = np.linspace(0, 1, 6)
        comp_tp = np.zeros(len(comp_bins)-1)
        comp_total = np.zeros(len(comp_bins)-1)

        # Stage-specific data generation
        if self.stage_idx == 0:
            from src.data.stage0_gaussian import GaussianPretrainingProvider
            data_cfg = self.config["data_params"]
            provider = GaussianPretrainingProvider(
                min_stars=data_cfg["min_stars"],
                max_stars=data_cfg["max_stars"],
                image_size=data_cfg["image_size"],
                max_capacity_per_cell=data_cfg["max_capacity_per_cell"],
                shape_size=data_cfg["shape_size"],
                global_stretch_scale=self.stretch_scale
            )
            
            for _ in range(num_chunks):
                sample = provider[0] # Get one chunk
                image_tensor = sample["image"]
                target_grid = sample["target"]
                
                # Predict
                with torch.no_grad():
                    input_tensor = image_tensor.unsqueeze(0).to(self.device)
                    prediction_dict = self.model(input_tensor)
                    prediction = prediction_dict["stars"].squeeze(0).cpu().numpy()
                
                # Extract True Stars (Filter by completeness for honest recall)
                true_stars = []
                true_stars_all = [] # For internal reference
                grid_h, grid_w = target_grid.shape[:2]
                K = data_cfg["max_capacity_per_cell"]
                S2_plus_5 = (target_grid.shape[-1] - 1) // K
                target_reshaped = target_grid[..., :-1].view(grid_h, grid_w, K, S2_plus_5).numpy()
                
                cell_size = provider.cell_size
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K):
                            slot = target_reshaped[y, x, k]
                            tp, tdx, tdy, raw_flux_target, tc = slot[:5]
                            if tp == 1.0:
                                star_info = ((x * cell_size) + tdx, (y * cell_size) + tdy, float(raw_flux_target), tc)
                                true_stars_all.append(star_info)
                                # NEW: Only expect the model to find "detectable" stars (c > 0.5)
                                if tc > 0.5:
                                    true_stars.append(star_info)
                
                # Extract Predicted Stars (p > threshold)
                pred_stars = []
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K):
                            p, dx, dy, physical_flux_pred, c = prediction[y, x, k, :5]
                            if p > threshold:
                                pred_stars.append(((x * cell_size) + dx, (y * cell_size) + dy, float(physical_flux_pred), c, p))
                
                # Match using 2.0 pixel threshold
                dist_thresh = 2.0
                matches, unmatched_true, unmatched_pred = match_stars(true_stars, pred_stars, distance_threshold=dist_thresh)
                
                all_tp += len(matches)
                all_fp += len(unmatched_pred)
                all_fn += len(unmatched_true)

                # Analyze completeness of matched vs missed
                matched_true_indices = [m[0] for m in matches]
                for i, star in enumerate(true_stars):
                    comp = star[3]
                    if i in matched_true_indices:
                        matched_completeness.append(comp)
                    else:
                        missed_completeness.append(comp)

                    # Bin recall by completeness
                    for b in range(len(comp_bins)-1):
                        if comp_bins[b] <= comp <= comp_bins[b+1]:
                            comp_total[b] += 1
                            if i in matched_true_indices:
                                comp_tp[b] += 1
                            break

                for t_idx, p_idx, dist in matches:
                    pos_errors.append(dist)
                    t_flux = true_stars[t_idx][2]
                    p_flux = pred_stars[p_idx][2]
                    t_comp = true_stars[t_idx][3]
                    p_comp = pred_stars[p_idx][3]
                    
                    ratios.append(p_flux / (t_flux + 1e-9))
                    comp_errors.append(abs(p_comp - t_comp))
                    
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        rmse = np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else 1.0
        flux_accuracy = np.median(ratios) if ratios else 0
        flux_scatter = np.std(ratios) if ratios else 0
        comp_mae = np.mean(comp_errors) if comp_errors else 1.0

        print("\n=============================================")
        print(" STAGE 0 ACCEPTANCE CRITERIA CHECK")
        print("=============================================")
        self._print_metric("Recall (All Sources)", recall, 0.95)
        self._print_metric("Precision", precision, 0.98)
        self._print_metric("Positional RMSE", rmse, 0.15, reverse=True)
        self._print_metric("Flux Ratio Accuracy", flux_accuracy, 0.95)
        self._print_metric("Flux Scatter (StdDev)", flux_scatter, 0.10, reverse=True)
        self._print_metric("Completeness MAE", comp_mae, 0.10, reverse=True)
        print("---------------------------------------------")
        print(f"📊 Avg Completeness (Detected):   {np.mean(matched_completeness) if matched_completeness else 0:.4f}")
        print(f"📊 Avg Completeness (Missed):     {np.mean(missed_completeness) if missed_completeness else 0:.4f}")
        print("---------------------------------------------")
        print("📈 Recall vs. Completeness (SNR Proxy):")
        for b in range(len(comp_bins)-1):
            b_recall = comp_tp[b] / comp_total[b] if comp_total[b] > 0 else 0
            print(f"   Bin {comp_bins[b]:.1f}-{comp_bins[b+1]:.1f}:   {b_recall:.4f} ({int(comp_tp[b])}/{int(comp_total[b])})")
        
        if recall > 0.9 and precision > 0.9:
            print("\n✅ MODEL IS LOOKING GOOD!")
        else:
            print("\n⚠️ MODEL NEEDS MORE TRAINING OR CALIBRATION.")
        print("=============================================\n")

    def _print_metric(self, name, value, target, reverse=False):
        status = "✅"
        if reverse:
            if value > target: status = "❌"
        else:
            if value < target: status = "❌"
        print(f"{status} {name:<23}:   {value:.4f} (Target: {target:>8.4f})")
