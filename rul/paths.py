"""Where the data lives. Override with environment variables; defaults are inside the repo (data/, git-ignored)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get('RUL_RAW', os.path.join(ROOT, 'data', 'raw'))                 # downloaded datasets, see data/README.md
FEATURES = os.environ.get('RUL_FEATURES', os.path.join(ROOT, 'data', 'features'))  # per-cycle feature files written by features/
os.makedirs(FEATURES, exist_ok=True)
