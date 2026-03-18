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
        
        cell_size = self.dataset.cell_size
        grid_size = self.dataset.grid_size
        K = self.dataset.K

        for _ in range(num_chunks):
            sparse_target = self.dataset.generate_chunk()
            image_tensor = sparse_target["image"]
            target_grid = sparse_target["base_grid"].numpy()

            with torch.no_grad():
                input_tensor = image_tensor.to(self.device)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                prediction_dict = self.model(input_tensor)
                prediction = prediction_dict["stars"].squeeze(0).cpu().numpy()
            
            # Extract true stars from grid
            true_stars = []
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(K):
                        slot = target_grid[y, x, k]
                        tp, tdx, tdy, tm, tc = slot
                        if tp == 1.0:
                            tgx = (x * cell_size) + tdx
                            tgy = (y * cell_size) + tdy
                            true_stars.append((tgx, tgy, 10**tm, tc))

            all_true_catalogues.append(true_stars)
            all_raw_predictions.append(prediction)

        for thresh in thresholds:
            tp, fp, fn = 0, 0, 0
            
            for true_catalogue, prediction in zip(all_true_catalogues, all_raw_predictions):
                pred_stars = []
                for y in range(grid_size):
                    for x in range(grid_size):
                        for k in range(K):
                            p, dx, dy, log_m, c = prediction[y, x, k, :5]
                            if p > thresh:
                                gx = (x * cell_size) + dx
                                gy = (y * cell_size) + dy
                                pred_stars.append((gx, gy, 10**log_m, c, p))
                
                matches, unmatched_true, unmatched_pred = match_stars(true_catalogue, pred_stars)
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
