import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Loads and validates the pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration YAML file, by default "config/config.yaml".

    Returns
    -------
    dict
        The loaded and validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty or missing required sections/fields.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Basic Validation
    required_top_sections = ["run_config", "curriculum", "data_params"]
    for section in required_top_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: '{section}'")

    # Validate data_params (needed for generation)
    data_req = ["num_train_samples", "num_val_samples", "min_stars", "max_stars", "image_size"]
    for field in data_req:
        if field not in config["data_params"]:
            raise ValueError(f"Missing field in data_params: '{field}'")

    return config
