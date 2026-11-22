from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
def load_config(path=None):
    p=Path(path) if path else ROOT/"config.yaml"
    with p.open(encoding="utf-8") as f: cfg=yaml.safe_load(f)
    cfg["_root"]=str(ROOT)
    return cfg
def project_path(value,cfg=None):
    p=Path(value)
    return p if p.is_absolute() else Path((cfg or {}).get("_root",ROOT))/p
