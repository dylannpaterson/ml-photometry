import torch
import numpy as np

class AstroSpaceTransform:
    """
    Handles bi-directional mapping between Physical Space (linear photons) 
    and Network Space (stretched, zero-centered features).
    """
    def __init__(self, stretch_scale=10.0):
        self.stretch_scale = stretch_scale

    # ==========================================
    # 1. PHYSICAL -> NETWORK (Pre-processing)
    # ==========================================
    def image_to_network(self, linear_image, chunk_median):
        """Prepares an input image for the neural network."""
        residual = linear_image - chunk_median
        if isinstance(linear_image, torch.Tensor):
            return torch.arcsinh(residual / self.stretch_scale)
        return np.arcsinh(residual / self.stretch_scale)

    def target_flux_to_network(self, linear_flux):
        """
        Converts true linear flux to the target value the network should predict.
        """
        if isinstance(linear_flux, torch.Tensor):
            return torch.arcsinh(linear_flux / self.stretch_scale)
        return np.arcsinh(linear_flux / self.stretch_scale)
        
    def target_bg_to_network(self, linear_bg_residual):
        """Stretches the background target to match the image space."""
        if isinstance(linear_bg_residual, torch.Tensor):
            return torch.arcsinh(linear_bg_residual / self.stretch_scale)
        return np.arcsinh(linear_bg_residual / self.stretch_scale)

    # ==========================================
    # 2. NETWORK -> PHYSICAL (Post-processing)
    # ==========================================
    def network_to_image(self, stretched_image, chunk_median):
        """Reconstructs the absolute linear image."""
        if isinstance(stretched_image, torch.Tensor):
            linear_residual = torch.sinh(stretched_image) * self.stretch_scale
        else:
            linear_residual = np.sinh(stretched_image) * self.stretch_scale
        return linear_residual + chunk_median

    def network_to_flux(self, predicted_m):
        """Converts the network's flux prediction back to linear photons."""
        if isinstance(predicted_m, torch.Tensor):
            return torch.sinh(predicted_m) * self.stretch_scale
        return np.sinh(predicted_m) * self.stretch_scale
        
    def network_to_bg(self, predicted_bg):
        """Converts the network's background prediction back to linear residuals."""
        if isinstance(predicted_bg, torch.Tensor):
            return torch.sinh(predicted_bg) * self.stretch_scale
        return np.sinh(predicted_bg) * self.stretch_scale
