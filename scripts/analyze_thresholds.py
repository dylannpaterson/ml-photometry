import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.generate_simple_synthetic_data import GaussianStarDataset
from models.dense_grid_model import DenseGridModel
from scripts.evaluate import match_stars

def analyze_thresholds(model_path="checkpoints/bulge_survey_final.pth", num_chunks=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseGridModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = GaussianStarDataset(min_stars=500, max_stars=1500)
    
    thresholds = np.linspace(0.01, 0.99, 50)
    results = []

    print(f"Analyzing {num_chunks} chunks over {len(thresholds)} thresholds...")

    # Pre-collect predictions to avoid redundant forward passes
    all_true_catalogues = []
    all_raw_predictions = []
    
    for _ in range(num_chunks):
        image_tensor, _, true_catalogue = dataset.generate_chunk()
        with torch.no_grad():
            input_tensor = image_tensor.unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).cpu().numpy()
        
        all_true_catalogues.append(true_catalogue)
        all_raw_predictions.append(prediction)

    cell_size = dataset.cell_size
    grid_size = dataset.grid_size

    for thresh in thresholds:
        tp, fp, fn = 0, 0, 0
        
        for true_catalogue, prediction in zip(all_true_catalogues, all_raw_predictions):
            pred_stars = []
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(5):
                        p, dx, dy, m, c = prediction[y, x, k]
                        if p > thresh:
                            gx = (x * cell_size) + dx
                            gy = (y * cell_size) + dy
                            pred_stars.append((gx, gy, m, c, p))
            
            matches, unmatched_true, unmatched_pred = match_stars(true_catalogue, pred_stars)
            tp += len(matches)
            fp += len(unmatched_pred)
            fn += len(unmatched_true)
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        results.append((thresh, precision, recall, fp, fn))
        print(f"Thresh: {thresh:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | FP: {fp} | FN: {fn}")

    # Plotting
    results = np.array(results)
    t = results[:, 0]
    prec = results[:, 1]
    rec = results[:, 2]
    fp_count = results[:, 3]
    fn_count = results[:, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Precision-Recall Curve
    ax1.plot(rec, prec, 'b-o', markersize=4)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # Plot 2: FP and FN vs Threshold
    ax2.plot(t, fp_count, 'r-', label='False Positives (FP)')
    ax2.set_xlabel('Probability Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('FP / FN vs Threshold')
    
    ax3 = ax2.twinx()
    ax3.plot(t, fn_count, 'g-', label='False Negatives (FN)')
    ax3.set_ylabel('False Negative Count')
    
    # Combined legend
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')
    
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Threshold Sensitivity Analysis (Edge-to-Edge Model)\nData: {num_chunks} chunks, 500-1500 stars/chunk", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("threshold_analysis.png")
    print("\nAnalysis complete. Results saved to threshold_analysis.png")

if __name__ == "__main__":
    analyze_thresholds()
