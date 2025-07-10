from pathlib import Path


def get_config_dir() -> str:
    """Get the path of the config directory"""
    script_dir = Path(__file__).parent
    return str(script_dir / "config")
