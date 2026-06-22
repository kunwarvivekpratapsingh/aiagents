"""
IMMUTABLE — do not modify.
Loads auth and fraud tables, exposes schema info for Claude prompts.
"""

import pandas as pd
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data() -> tuple:
    auth_path = os.path.join(DATA_DIR, "auth.csv")
    fraud_path = os.path.join(DATA_DIR, "fraud.csv")
    if not os.path.exists(auth_path) or not os.path.exists(fraud_path):
        raise FileNotFoundError(
            f"Missing data files. Run: python generate_sample.py\n"
            f"Expected: {auth_path} and {fraud_path}"
        )
    auth_df = pd.read_csv(auth_path)
    fraud_df = pd.read_csv(fraud_path)
    return auth_df, fraud_df


def load_problem_statement() -> str:
    path = os.path.join(FE_DIR, "problem.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "Detect fraudulent payment card transactions."


def get_schema_info(auth_df: pd.DataFrame, fraud_df: pd.DataFrame) -> dict:
    def describe_df(df: pd.DataFrame) -> dict:
        info = {}
        for col in df.columns:
            series = df[col].dropna()
            info[col] = {
                "dtype": str(df[col].dtype),
                "null_pct": round(df[col].isna().mean() * 100, 1),
                "n_unique": int(df[col].nunique()),
                "sample_values": series.head(3).tolist(),
            }
        return info

    return {
        "auth": {
            "shape": list(auth_df.shape),
            "columns": describe_df(auth_df),
        },
        "fraud": {
            "shape": list(fraud_df.shape),
            "columns": describe_df(fraud_df),
            "note": "fraud table arrives 60-120 days AFTER the transaction (reporting delay). NEVER use in feature engineering.",
        },
    }


def schema_to_prompt_text(schema: dict) -> str:
    """Format schema as readable text for Claude prompts."""
    lines = ["AUTH TABLE SCHEMA (these are the ONLY columns available at inference time):"]
    for col, info in schema["auth"]["columns"].items():
        sample = ", ".join([repr(v) for v in info["sample_values"]])
        lines.append(
            f"  - {col}: {info['dtype']} | {info['n_unique']} unique | "
            f"null={info['null_pct']}% | sample=[{sample}]"
        )
    lines.append(f"\nTotal rows: {schema['auth']['shape'][0]:,}")
    lines.append(f"\nFRAUD TABLE (DO NOT USE IN FEATURES — only for context):")
    for col, info in schema["fraud"]["columns"].items():
        lines.append(f"  - {col}: {info['dtype']}")
    return "\n".join(lines)
