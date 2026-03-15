import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.generate_simple_synthetic_data import GaussianStarDataset
from models.dense_grid_model import DenseGridModel

def match_stars(true_stars, pred_stars, dist_threshold=2.0):
    """
    Matches true stars to predicted stars using the Hungarian algorithm.
    true_stars: list of (x, y, flux, completeness)
    pred_stars: list of (x, y, flux, completeness, prob)
    """
    if not true_stars or not pred_stars:
        return [], list(range(len(true_stars))), list(range(len(pred_stars)))

    # Create cost matrix (Euclidean distance)
    true_coords = np.array([(s[0], s[1]) for s in true_stars])
    pred_coords = np.array([(s[0], s[1]) for s in pred_stars])
    
    # [num_true, num_pred]
    dists = np.sqrt(((true_coords[:, np.newaxis, :] - pred_coords[np.newaxis, :, :])**2).sum(axis=2))
    
    # Hungarian matching
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

def evaluate(model_path="checkpoints/dense_grid_final.pth", num_chunks=500, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseGridModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = GaussianStarDataset(max_stars_per_chunk=150)

    all_tp = 0
    all_fp = 0
    all_fn = 0

    pos_errors = []
    flux_errors = []
    comp_errors = []

    # Analysis vs Flux
    flux_bins = [0, 100, 200, 300, 400, 500]
    flux_tp = [0] * (len(flux_bins) - 1)
    flux_total = [0] * (len(flux_bins) - 1)

    print(f"Evaluating model on {num_chunks} chunks...")

    
    for _ in range(num_chunks):
        image_tensor, _, true_catalogue = dataset.generate_chunk()
        
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).cpu().numpy()
            
        pred_stars = []
        pad = dataset.pad
        cell_size = dataset.cell_size
        
        for y in range(64):
            for x in range(64):
                for k in range(5):
                    slot = prediction[y, x, k]
                    p, dx, dy, m, c = slot
                    if p > threshold:
                        gx = pad + (x * cell_size) + dx
                        gy = pad + (y * cell_size) + dy
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
            
            # Flux error: (pred - true) / true
            true_flux = true_catalogue[t_idx][2]
            pred_flux = pred_stars[p_idx][2]
            flux_err = (pred_flux - true_flux) / true_flux
            flux_errors.append(flux_err)
            
            # Completeness error (MAE)
            true_comp = true_catalogue[t_idx][3]
            pred_comp = pred_stars[p_idx][3]
            comp_errors.append(abs(pred_comp - true_comp))
            
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_pos_rmse = np.sqrt(np.mean(np.array(pos_errors)**2)) if pos_errors else 0
    avg_flux_mae = np.mean(np.abs(flux_errors)) if flux_errors else 0
    avg_comp_mae = np.mean(comp_errors) if comp_errors else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Total True Stars: {all_tp + all_fn}")
    print(f"Total Predicted:  {all_tp + all_fp}")
    print(f"True Positives:   {all_tp}")
    print(f"False Positives:  {all_fp}")
    print(f"False Negatives:  {all_fn}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"Positional RMSE:  {avg_pos_rmse:.4f} pixels")
    print(f"Flux Relative MAE: {avg_flux_mae*100:.2f}%")
    print(f"Completeness MAE: {avg_comp_mae:.4f} (0-1 scale)")
    
    print("\n--- Recall vs Flux ---")
    for b in range(len(flux_bins)-1):
        r = flux_tp[b] / flux_total[b] if flux_total[b] > 0 else 0
        print(f"Flux {flux_bins[b]}-{flux_bins[b+1]}: {r:.4f} ({flux_tp[b]}/{flux_total[b]})")

if __name__ == "__main__":
    evaluate()
