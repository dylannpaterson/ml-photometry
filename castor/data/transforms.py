import torch
import numpy as np

class AstroSpaceTransform:
    """
    Handles bi-directional mapping between Physical Space and Network Space.

    Physical Space consists of linear photons, while Network Space consists 
    of stretched, zero-centered features suitable for neural network input.

    Attributes
    ----------
    stretch_scale : float
        The scale used for arcsinh stretching.
    """
    def __init__(self, stretch_scale=10.0):
        """
        Initialize the AstroSpaceTransform.

        Parameters
        ----------
        stretch_scale : float, optional
            Flux stretch scale, by default 10.0.
        """
        self.stretch_scale = stretch_scale

    # ==========================================
    # 1. PHYSICAL -> NETWORK (Pre-processing)
    # ==========================================
    def image_to_network(self, linear_image, chunk_median):
        """
        Prepares an input image for the neural network.

        Parameters
        ----------
        linear_image : torch.Tensor or numpy.ndarray
            The raw linear image data.
        chunk_median : float
            The median value of the image chunk for subtraction.

        Returns
-------
        torch.Tensor or numpy.ndarray
            The stretched and median-subtracted image.
        """
        residual = linear_image - chunk_median
        if isinstance(linear_image, torch.Tensor):
            return torch.arcsinh(residual / self.stretch_scale)
        return np.arcsinh(residual / self.stretch_scale)

    def target_flux_to_network(self, linear_flux):
        """
        Converts true linear flux to the target value the network should predict.

        Parameters
        ----------
        linear_flux : torch.Tensor or numpy.ndarray
            The raw linear flux.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            The stretched target flux.
        """
        if isinstance(linear_flux, torch.Tensor):
            return torch.arcsinh(linear_flux / self.stretch_scale)
        return np.arcsinh(linear_flux / self.stretch_scale)
        
    def target_bg_to_network(self, linear_bg_residual):
        """
        Stretches the background target to match the image space.

        Parameters
        ----------
        linear_bg_residual : torch.Tensor or numpy.ndarray
            The raw background linear residual.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            The stretched background target.
        """
        if isinstance(linear_bg_residual, torch.Tensor):
            return torch.arcsinh(linear_bg_residual / self.stretch_scale)
        return np.arcsinh(linear_bg_residual / self.stretch_scale)

    # ==========================================
    # 2. NETWORK -> PHYSICAL (Post-processing)
    # ==========================================
    def network_to_image(self, stretched_image, chunk_median):
        """
        Reconstructs the absolute linear image from network output.

        Parameters
        ----------
        stretched_image : torch.Tensor or numpy.ndarray
            The stretched image from the network.
        chunk_median : float
            The median value to add back.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            The reconstructed absolute linear image.
        """
        if isinstance(stretched_image, torch.Tensor):
            linear_residual = torch.sinh(stretched_image) * self.stretch_scale
        else:
            linear_residual = np.sinh(stretched_image) * self.stretch_scale
        return linear_residual + chunk_median

    def network_to_flux(self, predicted_m):
        """
        Converts the network's flux prediction back to linear photons.

        Parameters
        ----------
        predicted_m : torch.Tensor or numpy.ndarray
            The stretched flux predicted by the network.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            The reconstructed linear flux.
        """
        if isinstance(predicted_m, torch.Tensor):
            return torch.sinh(predicted_m) * self.stretch_scale
        return np.sinh(predicted_m) * self.stretch_scale
        
    def network_to_bg(self, predicted_bg):
        """
        Converts the network's background prediction back to linear residuals.

        Parameters
        ----------
        predicted_bg : torch.Tensor or numpy.ndarray
            The stretched background predicted by the network.

        Returns
        -------
        torch.Tensor or numpy.ndarray
            The reconstructed linear background residual.
        """
        if isinstance(predicted_bg, torch.Tensor):
            return torch.sinh(predicted_bg) * self.stretch_scale
        return np.sinh(predicted_bg) * self.stretch_scale
