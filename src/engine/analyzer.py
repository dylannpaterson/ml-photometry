import torch
import numpy as np
import matplotlib.pyplot as plt
from src.engine.evaluator import match_stars

class ThresholdAnalyzer:
    def __init__(self, model, device, dataset):
        self.model = model
        self.device = device
        self.dataset = dataset

    def run_analysis(self, num_chunks=50, output_path="threshold_analysis.png"):
        self.model.eval()
        
        thresholds = np.linspace(0.01, 0.99, 50)
        results = []

        print(f"Analyzing {num_chunks} chunks over {len(thresholds)} thresholds...")

        all_true_catalogues = []
        all_raw_predictions = []
        
        for _ in range(num_chunks):
            image_tensor, _, true_catalogue = self.dataset.generate_chunk()
            with torch.no_grad():
                input_tensor = image_tensor.unsqueeze(0).to(self.device)
                prediction = self.model(input_tensor).squeeze(0).cpu().numpy()
            
            all_true_catalogues.append(true_catalogue)
            all_raw_predictions.append(prediction)

        cell_size = 2 # Fixed by current architecture
        grid_h, grid_w, K, _ = all_raw_predictions[0].shape

        for thresh in thresholds:
            tp, fp, fn = 0, 0, 0
            
            for true_catalogue, prediction in zip(all_true_catalogues, all_raw_predictions):
                pred_stars = []
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K):
                            p, dx, dy, m, c = prediction[y, x, k]
                            if p > thresh:
                                gx = (x * cell_size) + dx
                                gy = (y * cell_size) + dy
                                pred_stars.append((gx, gy, m, c, p))
                
                matches, _, unmatched_pred, unmatched_true = match_stars(true_catalogue, pred_stars)
                tp += len(matches)
                fp += len(unmatched_pred)
                fn += len(unmatched_true)
                
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            results.append((thresh, precision, recall, fp, fn))

        results = np.array(results)
        self._plot_results(results, num_chunks, output_path)
        return results

    def _plot_results(self, results, num_chunks, output_path):
        t = results[:, 0]
        prec = results[:, 1]
        rec = results[:, 2]
        fp_count = results[:, 3]
        fn_count = results[:, 4]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(rec, prec, 'b-o', markersize=4)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])

        ax2.plot(t, fp_count, 'r-', label='False Positives (FP)')
        ax2.set_xlabel('Probability Threshold')
        ax2.set_ylabel('Count')
        ax2.set_title('FP / FN vs Threshold')
        
        ax3 = ax2.twinx()
        ax3.plot(t, fn_count, 'g-', label='False Negatives (FN)')
        ax3.set_ylabel('False Negative Count')
        
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper center')
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Threshold Sensitivity Analysis\nData: {num_chunks} chunks", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        print(f"Analysis complete. Results saved to {output_path}")
