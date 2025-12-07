"""Utility to load Hydra configuration with optional overrides."""

from hydra import compose, initialize
from omegaconf import OmegaConf


def load_config(overrides=None, config_path="config"):
    """Load Hydra configuration with optional overrides.

    Args:
        overrides (list): List of configuration overrides (e.g., ["data=csv", "llm=openai"])
        config_path (str): Path to the configuration directory.

    Returns:
        DictConfig: Loaded configuration object.
    """
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name="base", overrides=overrides or [])
    print(OmegaConf.to_yaml(cfg))
    return cfg
