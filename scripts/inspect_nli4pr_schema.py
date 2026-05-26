from datasets import load_dataset
import pandas as pd

ds = load_dataset("Mathilde/NLI4PR")

print("Splits:", list(ds.keys()))
for split in ds.keys():
    print("\n" + "=" * 80)
    print("SPLIT:", split)
    df = ds[split].to_pandas()
    print("Columns:", list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    for col in df.columns:
        vals = df[col].astype(str).head(5).tolist()
        print(f"\nColumn: {col}")
        for i, v in enumerate(vals):
            v_short = v.replace("\n", " ")[:300]
            print(f"  [{i}] {v_short}")
