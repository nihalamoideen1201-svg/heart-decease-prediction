import argparse
from pathlib import Path

import pandas as pd


def inspect_dataset(data_path: Path, rows: int) -> None:
    df = pd.read_csv(data_path)
    print(f"Dataset: {data_path.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a heart disease dataset CSV.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--rows", type=int, default=5, help="Number of preview rows to print.")
    args = parser.parse_args()

    inspect_dataset(Path(args.data), args.rows)


if __name__ == "__main__":
    main()
