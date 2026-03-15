import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.generate_simple_synthetic_data import GaussianStarDataset
from models.dense_grid_model import DenseGridModel

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

def evaluate(model_path="checkpoints/dense_grid_final.pth", num_chunks=100, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseGridModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = GaussianStarDataset(min_stars=500, max_stars=1500)

    all_tp, all_fp, all_fn = 0, 0, 0
    pos_errors, flux_errors = [], []

    print(f"Evaluating model on {num_chunks} chunks (Edge-to-Edge)...")

    for _ in range(num_chunks):
        image_tensor, _, true_catalogue = dataset.generate_chunk()
        
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).cpu().numpy()
            
        pred_stars = []
        cell_size, grid_size = dataset.cell_size, dataset.grid_size
        
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

        for t_idx, p_idx, dist in matches:
            pos_errors.append(dist)
            true_flux = true_catalogue[t_idx][2]
            pred_flux = pred_stars[p_idx][2]
            flux_errors.append((pred_flux - true_flux) / true_flux)
            
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Evaluation Results (Edge-to-Edge) ---")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if pos_errors:
        print(f"Positional RMSE: {np.sqrt(np.mean(np.array(pos_errors)**2)):.4f} pixels")
        print(f"Flux Relative MAE: {np.mean(np.abs(flux_errors))*100:.2f}%")

if __name__ == "__main__":
    evaluate()
