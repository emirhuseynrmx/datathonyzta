import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round float columns in a submission CSV to 2 decimals."
    )
    parser.add_argument("--input", type=str, default="submission.csv")
    parser.add_argument("--output", type=str, default="submission_2dp.csv")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)

    float_cols = [
        col for col in df.columns if pd.api.types.is_float_dtype(df[col])
    ]
    if not float_cols:
        raise ValueError(
            f"No float columns detected in {input_path}. "
            "If your score column was read as string, convert it to float first."
        )

    df[float_cols] = df[float_cols].round(args.decimals)
    float_format = f"%.{args.decimals}f"
    df.to_csv(output_path, index=False, float_format=float_format)

    print(f"Wrote: {output_path} (rounded {float_cols} to {args.decimals}dp)")


if __name__ == "__main__":
    main()
