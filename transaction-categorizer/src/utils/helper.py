from pathlib import Path
import json
import os

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def write_csv(path, rows, header):
    ensure_dir(Path(path).parent)
    import csv
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
