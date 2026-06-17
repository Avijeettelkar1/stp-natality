"""Inspect raw file structure to understand the actual schema."""

from pathlib import Path
import pandas as pd

for fname in ["newborn_1986", "newborn_2010"]:
    p = Path("newborn_data") / fname
    # Read first 5 rows as-is, no names assigned
    df = pd.read_csv(
        p, header=None, nrows=5, low_memory=False, sep=",", on_bad_lines="warn"
    )
    print(f"=== {fname} (no names, first 5 rows) ===")
    print(df.to_string())
    print(f"Shape: {df.shape}")
    print()

    # Also try reading with header=0
    df2 = pd.read_csv(
        p, header=0, nrows=3, low_memory=False, sep=",", on_bad_lines="warn"
    )
    print(f"=== {fname} (header=0, first 3 rows) ===")
    print(df2.columns.tolist()[:10])
    print(df2.head(3).iloc[:, :8].to_string())
    print()
