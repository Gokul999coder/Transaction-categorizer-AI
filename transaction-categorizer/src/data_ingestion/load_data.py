import pandas as pd
from pathlib import Path
from ..utils.helper import ensure_dir

def load_csv(path: str, required_cols=None):
   
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if required_cols:
        mis = [c for c in required_cols if c not in df.columns]
        if mis:
            raise ValueError(f"Missing columns in {path}: {mis}")
    return df

def save_dataframe(df, out_path):
    ensure_dir(Path(out_path).parent)
    df.to_csv(out_path, index=False)
