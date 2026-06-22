"""
IMMUTABLE — do not modify.
Three-layer validation for generated features.py:
  L1 — Static checks (syntax, column existence, leakage, temporal safety)
  L2 — PySpark dry-run on 10-row mini DataFrame (skipped if Spark unavailable)
  L3 — LLM-as-judge scoring (Claude, 0–10)
"""

import ast
import re
import os
import json
import textwrap
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

FEATURES_PATH = Path(__file__).parent / "features.py"
FE_DIR = Path(__file__).parent

# ── Column leakage patterns (must never appear in features.py) ─────────────────
LEAKAGE_PATTERNS = [
    "fraud_df", "is_fraud", "confirmed", "fraud_type",
    "reported_date", "fraud_amount", "isFraud", "Class",
]

# Positive rangeBetween upper bound = future data
FUTURE_WINDOW_RE = re.compile(r'rangeBetween\s*\(\s*-?\d+\s*,\s*(\d+)\s*\)')


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Static analysis
# ─────────────────────────────────────────────────────────────────────────────

def _strip_comments_and_strings(source: str) -> str:
    """Remove docstrings and comments so leakage checks don't fire on instructional text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines()
    # Collect line ranges covered by string literals (docstrings)
    string_lines: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                string_lines.add(ln)
    clean = []
    for i, line in enumerate(lines, start=1):
        if i in string_lines:
            continue
        # Strip inline comments
        stripped = re.sub(r'#.*', '', line)
        clean.append(stripped)
    return "\n".join(clean)


def layer1_static(features_content: str, auth_columns: list) -> dict:
    """Fast rule-based static checks. Returns {pass, reason}."""

    # 1. Python syntax
    try:
        ast.parse(features_content)
    except SyntaxError as e:
        return {"pass": False, "layer": "L1", "reason": f"SyntaxError at line {e.lineno}: {e.msg}"}

    # 2. engineer_features function exists
    if "def engineer_features" not in features_content:
        return {"pass": False, "layer": "L1", "reason": "Missing engineer_features() function"}

    # 3. Fraud label leakage — check only executable code, not docstrings/comments
    code_only = _strip_comments_and_strings(features_content)
    for pat in LEAKAGE_PATTERNS:
        if re.search(r'\b' + re.escape(pat) + r'\b', code_only):
            return {"pass": False, "layer": "L1", "reason": f"Fraud leakage: '{pat}' referenced in features.py"}

    # 4. Future data (positive rangeBetween upper bound) — check code only
    for match in FUTURE_WINDOW_RE.finditer(code_only):
        upper = int(match.group(1))
        if upper > 0:
            return {
                "pass": False, "layer": "L1",
                "reason": f"Future data leak: rangeBetween upper bound +{upper} uses future rows"
            }

    # 5. Column existence — extract F.col("name") patterns from code only
    col_refs = set(re.findall(r'[Ff]\.col\s*\(\s*["\'](\w+)["\']', code_only))
    unknown = col_refs - set(auth_columns)
    if unknown:
        return {
            "pass": False, "layer": "L1",
            "reason": f"Unknown columns referenced (not in auth schema): {sorted(unknown)}"
        }

    # 6. Must add at least one new column
    if "withColumn" not in features_content and ".select(" not in features_content:
        return {"pass": False, "layer": "L1", "reason": "No new columns added (no withColumn found)"}

    return {"pass": True, "layer": "L1", "reason": "All static checks passed"}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — PySpark dry-run
# ─────────────────────────────────────────────────────────────────────────────

def _make_mini_spark_df(spark, auth_df_pandas: pd.DataFrame, n: int = 10):
    """Create a tiny Spark DataFrame from a pandas sample."""
    sample = auth_df_pandas.head(n).copy()
    # Convert object columns with mixed nulls to string to avoid Spark type errors
    for col in sample.select_dtypes(include="object").columns:
        sample[col] = sample[col].astype(str).replace("nan", None)
    return spark.createDataFrame(sample)


def layer2_dryrun(features_content: str, auth_df_pandas: pd.DataFrame) -> dict:
    """Run engineer_features on a tiny local Spark DataFrame."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return {"pass": True, "layer": "L2", "reason": "L2 skipped — PySpark not installed", "skipped": True}

    spark = None
    try:
        spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("FEValidate")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        mini_df = _make_mini_spark_df(spark, auth_df_pandas)
        input_cols = set(mini_df.columns)

        # Execute features.py in isolated namespace
        ns = {}
        exec(features_content, ns)
        engineer_features = ns.get("engineer_features")
        if engineer_features is None:
            return {"pass": False, "layer": "L2", "reason": "engineer_features not found after exec"}

        result_df = engineer_features(mini_df)
        output_cols = set(result_df.columns)
        new_cols = output_cols - input_cols

        if not new_cols:
            return {"pass": False, "layer": "L2", "reason": "engineer_features returned no new columns"}

        # Force evaluation (action)
        result_df.count()

        return {
            "pass": True, "layer": "L2",
            "reason": f"Dry-run OK — added {len(new_cols)} columns: {sorted(new_cols)}",
            "new_columns": sorted(new_cols),
        }
    except Exception as e:
        return {
            "pass": False, "layer": "L2",
            "reason": f"Runtime error: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-600:],
        }
    finally:
        if spark:
            spark.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert fraud detection data scientist evaluating PySpark feature engineering code.
Score the submitted features.py on the rubric below. Return ONLY valid JSON, no markdown.
"""

JUDGE_RUBRIC = """\
SCORING RUBRIC (return JSON with integer or float scores):
{
  "hypothesis_alignment": 0-3,     // Does the hypothesis logically follow from the problem statement?
  "feature_operationalization": 0-2,// Do the features correctly test the stated sub-hypotheses?
  "temporal_leakage_safety": 0-2,  // Are all windows look-back only? No fraud labels referenced?
  "code_quality": 0-2,             // Is the PySpark correct, readable, and non-trivial?
  "novelty": 0-1,                  // Are features meaningfully different from existing catalog?
  "total": 0-10,                   // Sum of above
  "feedback": "one concise sentence of constructive feedback",
  "keep_recommendation": true/false // Would you approve this for the feature catalog?
}
"""


def layer3_llm_judge(
    features_content: str,
    problem: str,
    existing_catalog_summary: str,
    client,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """LLM-as-judge scoring of the generated features."""
    prompt = f"""PROBLEM STATEMENT:
{problem}

EXISTING FEATURE CATALOG (avoid redundancy with these):
{existing_catalog_summary if existing_catalog_summary else "Empty — this is the first iteration."}

SUBMITTED features.py:
```python
{features_content}
```

{JUDGE_RUBRIC}
"""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        scores = json.loads(raw)
        scores["layer"] = "L3"
        scores["pass"] = scores.get("total", 0) >= 7.0 and scores.get("keep_recommendation", False)
        return scores
    except json.JSONDecodeError as e:
        return {
            "pass": False, "layer": "L3",
            "reason": f"LLM judge returned invalid JSON: {e}",
            "total": 0.0,
            "feedback": "Parse error",
        }
    except Exception as e:
        return {
            "pass": False, "layer": "L3",
            "reason": f"LLM judge error: {e}",
            "total": 0.0,
            "feedback": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Combined entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    features_content: str,
    auth_df: pd.DataFrame,
    problem: str,
    existing_catalog_summary: str,
    client,
) -> dict:
    """Run all three layers. Returns consolidated result dict."""
    auth_cols = list(auth_df.columns)

    l1 = layer1_static(features_content, auth_cols)
    if not l1["pass"]:
        return {
            "keep": False,
            "failure_layer": "L1",
            "failure_reason": l1["reason"],
            "l1": l1, "l2": None, "l3": None,
            "score": 0.0,
            "feedback": l1["reason"],
        }

    l2 = layer2_dryrun(features_content, auth_df)
    if not l2["pass"]:
        return {
            "keep": False,
            "failure_layer": "L2",
            "failure_reason": l2["reason"],
            "l1": l1, "l2": l2, "l3": None,
            "score": 0.0,
            "feedback": l2["reason"],
        }

    l3 = layer3_llm_judge(features_content, problem, existing_catalog_summary, client)
    keep = l3.get("pass", False)

    return {
        "keep": keep,
        "failure_layer": None if keep else "L3",
        "failure_reason": None if keep else l3.get("feedback", "Score below threshold"),
        "l1": l1, "l2": l2, "l3": l3,
        "score": l3.get("total", 0.0),
        "feedback": l3.get("feedback", ""),
        "new_columns": l2.get("new_columns", []),
    }


if __name__ == "__main__":
    import anthropic
    from prepare import load_problem_statement

    auth_df = pd.read_csv(Path(__file__).parent / "data" / "auth.csv")
    content = FEATURES_PATH.read_text()
    client = anthropic.Anthropic()
    problem = load_problem_statement()

    print("Running validation on current features.py...")
    result = run_validation(content, auth_df, problem, "", client)
    print(json.dumps(result, indent=2, default=str))
