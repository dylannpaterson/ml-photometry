import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.data.stage0_gaussian import GaussianPretrainingProvider

def match_stars(true_stars, pred_stars, dist_threshold=2.0):
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
        self.dataset = GaussianPretrainingProvider(
            min_stars=config["data_params"]["min_stars"],
            max_stars=config["data_params"]["max_stars"],
            image_size=config["data_params"]["image_size"]
        )

    def run_evaluation(self, num_chunks=100, threshold=0.5):
        self.model.eval()
        
        all_tp, all_fp, all_fn = 0, 0, 0
        pos_errors, flux_errors = [], []

        # Analysis vs Flux
        flux_bins = [0, 50, 100, 200, 500]
        flux_tp = [0] * (len(flux_bins) - 1)
        flux_total = [0] * (len(flux_bins) - 1)

        print(f"Evaluating model on {num_chunks} chunks...")

        for _ in range(num_chunks):
            image_tensor, _, true_catalogue = self.dataset.generate_chunk()
            
            with torch.no_grad():
                input_tensor = image_tensor.unsqueeze(0).to(self.device)
                prediction = self.model(input_tensor).squeeze(0).cpu().numpy()
                
            pred_stars = []
            cell_size, grid_size = self.dataset.cell_size, self.dataset.grid_size
            
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(5):
                        slot = prediction[y, x, k]
                        p, dx, dy, m, c = slot
                        if p > threshold:
                            gx = (x * cell_size) + dx
                            gy = (y * cell_size) + dy
                            pred_stars.append((gx, gy, m, c, p))
                            
            matches, unmatched_true, unmatched_pred = match_stars(true_catalogue, pred_stars)
            
            all_tp += len(matches)
            all_fp += len(unmatched_pred)
            all_fn += len(unmatched_true)

            matched_true_indices = [m[0] for m in matches]
            for i, star in enumerate(true_catalogue):
                f = star[2]
                for b in range(len(flux_bins)-1):
                    if flux_bins[b] <= f < flux_bins[b+1]:
                        flux_total[b] += 1
                        if i in matched_true_indices:
                            flux_tp[b] += 1
                        break

            for t_idx, p_idx, dist in matches:
                pos_errors.append(dist)
                true_flux = true_catalogue[t_idx][2]
                pred_flux = pred_stars[p_idx][2]
                flux_errors.append((pred_flux - true_flux) / true_flux)
                
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pos_rmse": np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else 0,
            "flux_mae": np.mean(np.abs(flux_errors)) if flux_errors else 0,
            "recall_vs_flux": []
        }
        
        print("\n--- Evaluation Results ---")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if pos_errors:
            print(f"Positional RMSE:  {results['pos_rmse']:.4f} pixels")
            print(f"Flux Relative MAE: {results['flux_mae']*100:.2f}%")
        
        print("\n--- Recall vs Flux ---")
        for b in range(len(flux_bins)-1):
            r = flux_tp[b] / flux_total[b] if flux_total[b] > 0 else 0
            results["recall_vs_flux"].append((flux_bins[b], flux_bins[b+1], r))
            print(f"Flux {flux_bins[b]:3d}-{flux_bins[b+1]:3d}: {r:.4f} ({flux_tp[b]:5d}/{flux_total[b]:5d})")
            
        return results
