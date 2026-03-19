import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.data.stage0_gaussian import GaussianPretrainingProvider

def match_stars(true_stars, pred_stars, dist_threshold=1.0):
    if not true_stars or not pred_stars:
        return [], list(range(len(true_stars))), list(range(len(pred_stars)))

    true_coords = np.array([(s[0], s[1]) for s in true_stars])
    pred_coords = np.array([(s[0], s[1]) for s in pred_stars])
    
    dists = np.sqrt(((true_coords[:, np.newaxis, :] - pred_coords[np.newaxis, :, :])**2).sum(axis=2))
    true_idx, pred_idx = linear_sum_assignment(dists)
    
    matches = []
    matched_true = set()
    matched_pred = set()
    
    for t, p in zip(true_idx, pred_idx):
        if dists[t, p] < dist_threshold:
            matches.append((t, p, dists[t, p]))
            matched_true.add(t)
            matched_pred.add(p)
            
    unmatched_true = [i for i in range(len(true_stars)) if i not in matched_true]
    unmatched_pred = [i for i in range(len(pred_stars)) if i not in matched_pred]
    
    return matches, unmatched_true, unmatched_pred

class Evaluator:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.K = config["data_params"]["max_capacity_per_cell"]
        self.S = config["data_params"]["shape_size"]
        self.dataset = GaussianPretrainingProvider(
            min_stars=config["data_params"]["min_stars"],
            max_stars=config["data_params"]["max_stars"],
            image_size=config["data_params"]["image_size"],
            max_capacity_per_cell=self.K,
            shape_size=self.S
        )

    def run_evaluation(self, num_chunks=100, threshold=0.5):
        self.model.eval()
        
        all_tp, all_fp, all_fn = 0, 0, 0
        pos_errors, ratios, comp_errors = [], [], []
        matched_completeness, missed_completeness = [], []

        # Completeness Bins for Recall Analysis
        comp_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        comp_total = [0] * (len(comp_bins) - 1)
        comp_tp = [0] * (len(comp_bins) - 1)

        print(f"Evaluating model on {num_chunks} chunks...")

        for _ in range(num_chunks):
            sparse_sample = self.dataset.generate_chunk()
            image_tensor = sparse_sample["image"]
            
            with torch.no_grad():
                input_tensor = image_tensor.to(self.device)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                prediction_dict = self.model(input_tensor)
                prediction = prediction_dict["stars"].squeeze(0).cpu().numpy()
                
            pred_stars = []
            cell_size, grid_size = self.dataset.cell_size, self.dataset.grid_size
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(self.K):
                        p, dx, dy, log_m, c = prediction[y, x, k, :5]
                        if p > threshold:
                            gx = (x * cell_size) + dx
                            gy = (y * cell_size) + dy
                            pred_stars.append((gx, gy, 10**log_m, c, p))
            
            true_stars = []
            target_grid = sparse_sample["base_grid"].numpy()
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(self.K):
                        slot = target_grid[y, x, k]
                        tp, tdx, tdy, tm, tc = slot
                        if tp == 1.0:
                            tgx = (x * cell_size) + tdx
                            tgy = (y * cell_size) + tdy
                            true_stars.append((tgx, tgy, 10**tm, tc))
                            
            matches, unmatched_true, unmatched_pred = match_stars(true_stars, pred_stars)
            
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
                
                ratios.append(p_flux / t_flux)
                comp_errors.append(abs(p_comp - t_comp))
                
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        
        mean_ratio = np.mean(ratios) if ratios else 0
        std_ratio = np.std(ratios) if ratios else 0
        comp_mae = np.mean(comp_errors) if comp_errors else 1.0
        pos_rmse = np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else 1.0
        
        avg_matched_comp = np.mean(matched_completeness) if matched_completeness else 0
        avg_missed_comp = np.mean(missed_completeness) if missed_completeness else 0

        print("\n" + "="*45)
        print(" STAGE 0 ACCEPTANCE CRITERIA CHECK")
        print("="*45)
        
        def check(val, target, label, higher_is_better=True):
            status = "✅" if (val >= target if higher_is_better else val <= target) else "❌"
            print(f"{status} {label:22}: {val:8.4f} (Target: {target:8.4f})")
            return val >= target if higher_is_better else val <= target

        results_ok = [
            check(recall, 0.95, "Recall (All Sources)"),
            check(precision, 0.98, "Precision"),
            check(pos_rmse, 0.15, "Positional RMSE", higher_is_better=False),
            check(1.0 - abs(1.0 - mean_ratio), 0.95, "Flux Ratio Accuracy"),
            check(std_ratio, 0.10, "Flux Scatter (StdDev)", higher_is_better=False),
            check(comp_mae, 0.10, "Completeness MAE", higher_is_better=False)
        ]
        
        print("-" * 45)
        print(f"📊 Avg Completeness (Detected): {avg_matched_comp:8.4f}")
        print(f"📊 Avg Completeness (Missed):   {avg_missed_comp:8.4f}")
        print("-" * 45)
        print("📈 Recall vs. Completeness (SNR Proxy):")
        for b in range(len(comp_bins)-1):
            bin_recall = comp_tp[b] / comp_total[b] if comp_total[b] > 0 else 0
            label = f"{comp_bins[b]:.1f}-{comp_bins[b+1]:.1f}"
            print(f"   Bin {label}: {bin_recall:8.4f} ({comp_tp[b]}/{comp_total[b]})")

        if all(results_ok):
            print("\n🎉 MODEL IS READY FOR STAGE 1 (REAL PSF)!")
        else:
            print("\n⚠️ MODEL NEEDS MORE TRAINING OR CALIBRATION.")
        print("="*45)
            
        return {
            "precision": precision,
            "recall": recall,
            "mean_ratio": mean_ratio,
            "std_ratio": std_ratio,
            "comp_mae": comp_mae,
            "pos_rmse": pos_rmse
        }
