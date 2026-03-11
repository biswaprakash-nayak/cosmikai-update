from pathlib import Path
import yaml

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, "r") as f:
        return yaml.safe_load(f)
