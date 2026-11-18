#!/usr/bin/env python3
"""Emit reproducible synthetic demand CSV to stdout (no external dependencies)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from dataset import dataset_csv, generate_rows

if __name__ == '__main__':
    sys.stdout.write(dataset_csv(generate_rows()))
