"""Configuration loader utility."""
import yaml
from pathlib import Path


def find_project_root():
    """Find project root directory by looking for configs folder.

    Returns:
        Path: Project root directory
    """
    current = Path.cwd()

    # Try current directory first
    if (current / "configs" / "config.yaml").exists():
        return current

    # Try going up to find project root
    for parent in current.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent

    # Last resort: assume we're in project root
    return Path(__file__).parent.parent.parent


def load_config(config_path=None):
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file (optional)

    Returns:
        dict: Configuration dictionary

    Example:
        >>> config = load_config()
        >>> data_path = config['data']['raw_path']
    """
    if config_path is None:
        # Find project root and construct path
        project_root = find_project_root()
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# Global config instance
_config = None


def get_config():
    """Get global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config