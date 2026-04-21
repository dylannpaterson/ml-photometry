import torch
import numpy as np
import matplotlib.pyplot as plt
from castor.engine.evaluator import match_stars
from castor.data.transforms import AstroSpaceTransform

class ThresholdAnalyzer:
    """
    Analyzes the sensitivity of the model to different probability thresholds.

    This tool helps in selecting the optimal threshold for star detection 
    by evaluating precision and recall across a range of values.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model.
    device : torch.device
        The device to run inference on.
    dataset : Dataset
        The dataset used for analysis.
    stretch_scale : float
        The scale used for arcsinh flux stretching.
    transform : AstroSpaceTransform
        The transform used for image preprocessing.
    """

    def __init__(self, model, device, dataset):
        """
        Initialize the ThresholdAnalyzer.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model.
        device : torch.device
            The device to use.
        dataset : Dataset
            The dataset to sample chunks from.
        """
        self.model = model
        self.device = device
        self.dataset = dataset
        self.stretch_scale = dataset.transform.stretch_scale
        self.transform = dataset.transform

    def run_analysis(self, num_chunks=20, output_path="threshold_analysis.png"):
        """
        Runs the threshold sensitivity analysis.

        Parameters
        ----------
        num_chunks : int, optional
            Number of image chunks to evaluate, by default 20.
        output_path : str, optional
            Path to save the resulting plots, by default "threshold_analysis.png".

        Returns
-------
        numpy.ndarray
            A 2D array of results: [threshold, precision, recall, fp_count, fn_count].
        """
        self.model.eval()
        
        thresholds = np.linspace(0.01, 0.99, 50)
        results = []

        print(f"Analyzing {num_chunks} chunks over {len(thresholds)} thresholds...")

        all_true_catalogues = []
        all_raw_predictions = []
        
        obj_p_scores = []
        bg_p_scores = []
        
        cell_size = self.dataset.cell_size
        grid_size = self.dataset.grid_size
        K = self.dataset.K

        for _ in range(num_chunks):
            sparse_sample = self.dataset.generate_chunk()
            image_tensor = sparse_sample["image"]
            # target_grid shape is (grid_size, grid_size, K, -1)
            target_grid = sparse_sample["base_grid"].numpy()

            # --- Apply Live Noise and Stretch ---
            img_pos = torch.clamp(image_tensor, min=0.0)
            img_noisy = torch.poisson(img_pos)
            img_noisy += torch.randn_like(img_noisy) * 5.0

            # Robust Background Estimation: Use 10th percentile to avoid bright star contamination
            robust_median = float(torch.quantile(img_noisy.view(-1), 0.10))
            img_stretched = torch.arcsinh((img_noisy - robust_median) / self.stretch_scale)
            with torch.no_grad():
                input_tensor = img_stretched.to(self.device)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                prediction_dict = self.model(input_tensor)
                prediction = prediction_dict["stars"].squeeze(0).cpu().numpy()
            
            true_stars = []
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(K):
                        p_pred = prediction[y, x, k, 0]
                        slot = target_grid[y, x, k]
                        tp, tdx, tdy, raw_flux_target, tc = slot[:5]
                        if tp == 1.0:
                            tgx = (x * cell_size) + tdx
                            tgy = (y * cell_size) + tdy
                            true_stars.append((tgx, tgy, float(raw_flux_target), tc))
                            obj_p_scores.append(p_pred)
                        else:
                            bg_p_scores.append(p_pred)

            all_true_catalogues.append(true_stars)
            all_raw_predictions.append(prediction)

        self._print_p_summary(obj_p_scores, bg_p_scores)

        for thresh in thresholds:
            tp, fp, fn = 0, 0, 0
            for true_catalogue, prediction in zip(all_true_catalogues, all_raw_predictions):
                pred_stars = []
                grid_h, grid_w, K_pred, _ = prediction.shape
                for y in range(grid_h):
                    for x in range(grid_w):
                        for k in range(K_pred):
                            p, dx, dy, physical_flux_pred, c = prediction[y, x, k, :5]
                            if p > thresh:
                                gx = (x * cell_size) + dx
                                gy = (y * cell_size) + dy
                                pred_stars.append((gx, gy, float(physical_flux_pred), c, p))
                
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

    def _print_p_summary(self, obj_p, bg_p):
        """Prints a distribution summary of the probability scores."""
        obj_p = np.array(obj_p)
        bg_p = np.array(bg_p)
        
        print("\n--- Probability Score (p) Distribution Summary ---")
        print(f"{'Metric':<15} | {'Object Slots':<15} | {'Background Slots':<15}")
        print("-" * 50)
        print(f"{'Mean':<15} | {np.mean(obj_p):<15.4f} | {np.mean(bg_p):<15.4f}")
        print(f"{'Max':<15} | {np.max(obj_p):<15.4f} | {np.max(bg_p):<15.4f}")
        print(f"{'Min':<15} | {np.min(obj_p):<15.4f} | {np.min(bg_p):<15.4f}")
        print(f"{'Median':<15} | {np.median(obj_p):<15.4f} | {np.median(bg_p):<15.4f}")
        print(f"{'90th Percentile':<15} | {np.percentile(obj_p, 90):<15.4f} | {np.percentile(bg_p, 90):<15.4f}")
        
        print("\n--- Potential Star Recall (by Threshold) ---")
        for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
            recall_at_t = np.sum(obj_p > t) / len(obj_p) if len(obj_p) > 0 else 0
            fp_at_t = np.sum(bg_p > t)
            print(f"Thresh {t:.2f}: Recall={recall_at_t:.4f}, Est. FPs per chunk={fp_at_t/20.0:.1f}")
        print("-" * 50)

    def _plot_results(self, results, num_chunks, output_path):
        """Generates diagnostic plots for the threshold analysis."""
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
