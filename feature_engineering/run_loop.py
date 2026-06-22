"""
Auto-Improving Feature Engineering Loop
========================================
Inspired by auto-improving-kernel (github.com/jyotilakra92/auto-improving-kernel).

Pattern:
  1. Generate hypothesis  →  2. Write features.py  →  3. Validate (L1/L2/L3)
  4. KEEP (git commit) or REVERT (git checkout)  →  5. Log  →  repeat

Usage:
  python run_loop.py                        # run with defaults (15 iterations)
  python run_loop.py --max-iter 20          # custom iteration count
  python run_loop.py --dataset ieee         # use IEEE-CIS from Kaggle
  python run_loop.py --dataset creditcard   # use ULB Credit Card from Kaggle
  python run_loop.py --generate-sample      # only generate synthetic data, then exit
"""

import os
import sys
import re
import json
import time
import shutil
import argparse
import textwrap
import subprocess
from pathlib import Path
from datetime import datetime

import anthropic
import pandas as pd

GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"

FE_DIR = Path(__file__).parent
FEATURES_PATH = FE_DIR / "features.py"
RESULTS_PATH = FE_DIR / "results.tsv"
CATALOG_PATH = FE_DIR / "feature_catalog.md"
REPO_ROOT = FE_DIR.parent  # aiagents/

# Models
MODEL_GENERATOR = "claude-sonnet-4-6"
MODEL_JUDGE = "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis generation
# ─────────────────────────────────────────────────────────────────────────────

HYPOTHESIS_SYSTEM = """\
You are an expert fraud detection data scientist specialising in payment systems (Visa/Mastercard).
Generate diverse, testable fraud hypotheses from the problem statement and auth table schema.
Return ONLY valid JSON — no markdown fences, no explanation.
"""

HYPOTHESIS_PROMPT = """\
PROBLEM STATEMENT:
{problem}

AUTH TABLE SCHEMA:
{schema_text}

Generate exactly 10 diverse, testable fraud hypotheses that can be validated purely from the
auth table columns listed above. Cover different fraud patterns:
velocity, geography, authentication weakness, amount anomalies, device/channel patterns, time patterns.

Return JSON array:
[
  {{
    "id": "H1",
    "hypothesis": "<one-sentence, specific and testable>",
    "rationale": "<why this pattern correlates with fraud>",
    "sub_hypotheses": ["SH1: ...", "SH2: ...", "SH3: ..."],
    "suggested_features": ["feature_name_1", "feature_name_2"],
    "temporal_windows": ["T+0", "T-10m", "T-1h"]
  }},
  ...
]
"""


def generate_hypotheses(client, problem: str, schema_text: str) -> list:
    print("  Generating initial hypothesis space...")
    response = client.messages.create(
        model=MODEL_GENERATOR,
        max_tokens=4096,
        system=HYPOTHESIS_SYSTEM,
        messages=[{
            "role": "user",
            "content": HYPOTHESIS_PROMPT.format(problem=problem, schema_text=schema_text),
        }],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    # Robust parse: if truncated, recover complete objects from partial JSON
    try:
        hypotheses = json.loads(raw)
    except json.JSONDecodeError:
        # Find the last complete object by scanning for complete {...} blocks
        objects = []
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objects.append(json.loads(raw[start:i+1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
        hypotheses = objects
        print(f"  (Recovered {len(hypotheses)} hypotheses from partial response)")

    if not hypotheses:
        raise ValueError("No hypotheses could be parsed from the LLM response")

    print(f"  Generated {len(hypotheses)} hypotheses")
    return hypotheses


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SYSTEM = """\
You are an expert PySpark feature engineer for payment fraud detection.
Write complete, executable PySpark code for the engineer_features() function.

STRICT RULES:
1. engineer_features(auth_df: DataFrame) -> DataFrame
   - Input: PySpark DataFrame with auth table columns ONLY
   - Output: Same DataFrame enriched with new feature columns
2. NEVER reference fraud_df, is_fraud, confirmed, fraud_type, reported_date, isFraud, or Class
3. All Window functions must use NEGATIVE rangeBetween (look-back only):
   - OK:  .rangeBetween(-600, 0)   ← 10-minute look-back
   - BAD: .rangeBetween(-600, 600) ← uses future rows
4. Only reference column names that exist in the auth schema provided
5. Output ONLY the complete features.py file — no markdown, no explanation

REQUIRED FILE FORMAT:
\"\"\"
HYPOTHESIS: <one-sentence>
RATIONALE:  <why it correlates with fraud>
SUB_HYPOTHESES:
  - SH1: <sub-hypothesis 1>
  - SH2: <sub-hypothesis 2>
  - SH3: <sub-hypothesis 3>
FEATURES: feature1, feature2, feature3
TEMPORAL_WINDOWS: T-10m, T-1h
\"\"\"

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pyspark.sql.types as T


def engineer_features(auth_df: DataFrame) -> DataFrame:
    # implementation here
    return auth_df
"""

FEATURE_PROMPT = """\
PROBLEM STATEMENT:
{problem}

AUTH TABLE SCHEMA (ONLY these columns are available):
{schema_text}

HYPOTHESIS TO IMPLEMENT:
{hypothesis}

SUB-HYPOTHESES TO COVER:
{sub_hypotheses}

SUGGESTED FEATURES:
{suggested_features}

EXISTING FEATURE CATALOG (avoid redundancy):
{catalog_summary}

HISTORY OF FAILED ATTEMPTS FOR THIS HYPOTHESIS:
{failure_history}

Write a complete features.py that implements 3-5 PySpark features testing this hypothesis.
Each feature must test one of the sub-hypotheses.
Use Window functions for velocity/aggregation features (with look-back windows only).
"""


def generate_features(
    client, problem: str, schema_text: str, hyp: dict,
    catalog_summary: str, failure_history: str
) -> str:
    sub_hyps = "\n".join(f"  {sh}" for sh in hyp.get("sub_hypotheses", []))
    suggested = ", ".join(hyp.get("suggested_features", []))

    response = client.messages.create(
        model=MODEL_GENERATOR,
        max_tokens=3500,
        system=FEATURE_SYSTEM,
        messages=[{
            "role": "user",
            "content": FEATURE_PROMPT.format(
                problem=problem,
                schema_text=schema_text,
                hypothesis=hyp["hypothesis"],
                sub_hypotheses=sub_hyps,
                suggested_features=suggested,
                catalog_summary=catalog_summary or "Empty — first iteration.",
                failure_history=failure_history or "None.",
            ),
        }],
    )
    content = response.content[0].text.strip()
    # Strip markdown fences if Claude wrapped the output
    content = re.sub(r"^```python\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────

def git(cmd: list, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True, text=True,
    )


def git_commit_features(iteration: int, score: float, hypothesis: str, new_cols: list):
    short_hyp = hypothesis[:70]
    new_cols_str = ", ".join(new_cols[:5])
    msg = (
        f"feat(fe): iter {iteration} | score={score:.1f} | {short_hyp}\n\n"
        f"New features: {new_cols_str}\n\n"
        f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    git(["add", "feature_engineering/features.py"])
    git(["add", "feature_engineering/feature_catalog.md"])
    result = git(["commit", "-m", msg])
    return result.returncode == 0


def git_revert_features():
    """Revert features.py to last committed version."""
    result = git(["checkout", "HEAD", "--", "feature_engineering/features.py"])
    if result.returncode != 0:
        # No prior commit — restore the baseline
        from features import engineer_features  # noqa: just check import works
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def init_results_tsv():
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "iter\tstatus\tscore\tfailure_layer\thypothesis\tnew_features\tfeedback\ttimestamp\n"
        )


def log_result(
    iteration: int, status: str, score: float,
    failure_layer: str, hypothesis: str, new_features: list, feedback: str,
):
    row = "\t".join([
        str(iteration), status, f"{score:.2f}",
        failure_layer or "-",
        hypothesis[:80],
        ", ".join(new_features[:6]),
        feedback[:100],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]) + "\n"
    with open(RESULTS_PATH, "a") as f:
        f.write(row)


# ─────────────────────────────────────────────────────────────────────────────
# Feature catalog
# ─────────────────────────────────────────────────────────────────────────────

def parse_docstring_meta(features_content: str) -> dict:
    """Extract hypothesis, sub_hypotheses, features from features.py docstring."""
    meta = {
        "hypothesis": "", "rationale": "",
        "sub_hypotheses": [], "features": [], "temporal_windows": [],
    }
    doc_match = re.search(r'"""(.*?)"""', features_content, re.DOTALL)
    if not doc_match:
        return meta

    doc = doc_match.group(1)

    for key, pattern in [
        ("hypothesis", r"HYPOTHESIS:\s*(.+)"),
        ("rationale", r"RATIONALE:\s*(.+)"),
        ("temporal_windows", r"TEMPORAL_WINDOWS:\s*(.+)"),
    ]:
        m = re.search(pattern, doc)
        if m:
            val = m.group(1).strip()
            meta[key] = val.split(", ") if key == "temporal_windows" else val

    sh_section = re.search(r"SUB_HYPOTHESES:(.*?)(?:FEATURES:|TEMPORAL_WINDOWS:|$)", doc, re.DOTALL)
    if sh_section:
        meta["sub_hypotheses"] = [
            line.strip().lstrip("- ") for line in sh_section.group(1).strip().splitlines()
            if line.strip() and line.strip().startswith("-")
        ]

    feat_match = re.search(r"FEATURES:\s*(.+)", doc)
    if feat_match:
        meta["features"] = [f.strip() for f in feat_match.group(1).split(",")]

    return meta


def extract_pyspark_snippets(features_content: str) -> list:
    """Extract individual withColumn blocks as readable snippets."""
    snippets = []
    pattern = re.compile(
        r'(#[^\n]+\n\s*)?auth_df\s*=\s*auth_df\.withColumn\s*\(\s*\n?.*?\)',
        re.DOTALL,
    )
    for m in pattern.finditer(features_content):
        snippets.append(m.group(0).strip())
    return snippets if snippets else [features_content]


def append_to_catalog(iteration: int, score: float, features_content: str, new_cols: list):
    meta = parse_docstring_meta(features_content)

    section = [
        f"\n## Iteration {iteration} | Score: {score:.1f}/10\n",
        f"### Hypothesis\n{meta['hypothesis']}\n",
        f"**Rationale:** {meta['rationale']}\n",
    ]

    if meta["sub_hypotheses"]:
        section.append("### Sub-Hypotheses\n")
        for sh in meta["sub_hypotheses"]:
            section.append(f"- {sh}\n")

    if meta["features"]:
        section.append(f"\n### Features Generated\n")
        for f in meta["features"]:
            section.append(f"- `{f}`\n")

    section.append(f"\n**Temporal windows:** {', '.join(meta['temporal_windows'])}\n")
    section.append(f"**New columns:** {', '.join(new_cols)}\n")

    section.append("\n### PySpark Implementation\n```python\n")
    section.append(features_content)
    section.append("\n```\n\n---\n")

    with open(CATALOG_PATH, "a") as f:
        f.writelines(section)


def init_catalog(problem: str):
    CATALOG_PATH.write_text(
        f"# Feature Engineering Catalog\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**Problem:** {problem.splitlines()[0]}\n\n"
        f"---\n"
    )


def catalog_summary_for_prompt() -> str:
    """Short summary of existing catalog for novelty checking."""
    if not CATALOG_PATH.exists():
        return ""
    content = CATALOG_PATH.read_text()
    hypotheses = re.findall(r"### Hypothesis\n(.+)", content)
    features = re.findall(r"\*\*New columns:\*\* (.+)", content)
    if not hypotheses:
        return ""
    lines = ["Existing validated hypotheses and features:"]
    for h, f in zip(hypotheses, features):
        lines.append(f"  - {h.strip()}  →  features: {f.strip()}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def print_status_table(results: list):
    print("\n" + "=" * 90)
    print(f"{'Iter':>4}  {'Status':>8}  {'Score':>5}  {'Layer':>5}  Hypothesis")
    print("-" * 90)
    for r in results:
        status_sym = "✓ KEPT" if r["status"] == "KEPT" else "✗ REVT"
        layer = r.get("failure_layer") or "  -  "
        hyp = r["hypothesis"][:50]
        print(f"{r['iter']:>4}  {status_sym:>8}  {r['score']:>5.1f}  {layer:>5}  {hyp}")
    print("=" * 90)


def run(max_iter: int = 15, dataset: str = "ieee"):
    client = anthropic.Anthropic()

    # ── Load data ──────────────────────────────────────────────────────────
    from data_loader import load_or_download
    from prepare import get_schema_info, load_problem_statement, schema_to_prompt_text
    from validate import run_validation

    print("\n" + "=" * 60)
    print("  Auto-Improving Feature Engineering Loop")
    print("=" * 60)

    auth_df, fraud_df = load_or_download(dataset)
    problem = load_problem_statement()
    schema = get_schema_info(auth_df, fraud_df)
    schema_text = schema_to_prompt_text(schema)

    print(f"  Auth table   : {auth_df.shape[0]:,} rows × {auth_df.shape[1]} cols")
    print(f"  Fraud table  : {fraud_df.shape[0]:,} confirmed fraud cases")
    print(f"  Fraud rate   : {len(fraud_df)/len(auth_df)*100:.1f}%")
    print()

    # ── Initialize ────────────────────────────────────────────────────────
    init_results_tsv()
    init_catalog(problem)

    # ── Generate hypotheses ───────────────────────────────────────────────
    try:
        hypotheses = generate_hypotheses(client, problem, schema_text)
    except Exception as e:
        err = str(e)
        if "credit balance is too low" in err or "insufficient" in err.lower():
            print(f"\n  ERROR: Insufficient Anthropic credits.")
            print(f"  Add credits at: https://console.anthropic.com/settings/billing")
            sys.exit(1)
        raise

    results_log = []
    iteration = 0
    consecutive_fails = {}  # hyp_id → fail count

    for hyp in hypotheses[:max_iter]:
        hyp_id = hyp.get("id", f"H{iteration}")
        iteration += 1

        print(f"\n[Iter {iteration:02d}] {hyp_id}: {hyp['hypothesis'][:65]}")
        print(f"         Sub-hypotheses: {len(hyp.get('sub_hypotheses', []))}")

        fail_count = consecutive_fails.get(hyp_id, 0)
        if fail_count >= 2:
            print(f"         Skipping — failed {fail_count} times already")
            continue

        # Collect failure history for this hypothesis
        failure_history = "\n".join([
            f"- Attempt failed at {r['failure_layer']}: {r['failure_reason']}"
            for r in results_log
            if r.get("hypothesis_id") == hyp_id and r["status"] == "REVERTED"
        ])

        # ── Generate features ────────────────────────────────────────────
        print("         Generating PySpark features...")
        try:
            features_content = generate_features(
                client, problem, schema_text, hyp,
                catalog_summary_for_prompt(), failure_history,
            )
        except Exception as e:
            err = str(e)
            if "credit balance is too low" in err or "insufficient" in err.lower():
                print(f"\n  {RED}ERROR: Insufficient Anthropic credits.{RESET}")
                print(f"  Add credits at: https://console.anthropic.com/settings/billing")
                sys.exit(1)
            print(f"         Generation error: {e}")
            continue

        FEATURES_PATH.write_text(features_content)

        # ── Validate ─────────────────────────────────────────────────────
        print("         Validating (L1 static → L2 Spark → L3 LLM judge)...")
        result = run_validation(
            features_content, auth_df, problem,
            catalog_summary_for_prompt(), client,
        )

        score = result.get("score", 0.0)
        new_cols = result.get("new_columns", [])
        feedback = result.get("feedback", "")
        failure_layer = result.get("failure_layer")
        failure_reason = result.get("failure_reason", "")

        # ── Keep or revert ────────────────────────────────────────────────
        if result["keep"]:
            print(f"         ✓ KEPT  score={score:.1f}  new_cols={new_cols}")
            append_to_catalog(iteration, score, features_content, new_cols)
            git_commit_features(iteration, score, hyp["hypothesis"], new_cols)
            status = "KEPT"
            consecutive_fails[hyp_id] = 0
        else:
            print(f"         ✗ REVERTED  failed at {failure_layer}: {failure_reason[:60]}")
            git_revert_features()
            status = "REVERTED"
            consecutive_fails[hyp_id] = fail_count + 1

        log_result(
            iteration, status, score, failure_layer,
            hyp["hypothesis"], new_cols, feedback,
        )

        entry = {
            "iter": iteration, "status": status, "score": score,
            "failure_layer": failure_layer, "failure_reason": failure_reason,
            "hypothesis": hyp["hypothesis"], "hypothesis_id": hyp_id,
            "new_cols": new_cols,
        }
        results_log.append(entry)

        # Brief pause to respect API rate limits
        time.sleep(1)

    # ── Final summary ─────────────────────────────────────────────────────
    print_status_table(results_log)
    kept = [r for r in results_log if r["status"] == "KEPT"]
    print(f"\n  Iterations   : {iteration}")
    print(f"  Kept         : {len(kept)}")
    print(f"  Reverted     : {iteration - len(kept)}")
    print(f"  Avg score    : {sum(r['score'] for r in kept)/max(len(kept),1):.1f}")
    print(f"\n  Feature catalog → {CATALOG_PATH}")
    print(f"  Iteration log  → {RESULTS_PATH}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-improving feature engineering loop")
    parser.add_argument("--max-iter", type=int, default=15, help="Max iterations (default 15)")
    parser.add_argument(
        "--dataset", choices=["ieee", "creditcard", "synthetic"], default="synthetic",
        help="Dataset source: ieee | creditcard (requires Kaggle creds) | synthetic (default)",
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate synthetic data and exit",
    )
    args = parser.parse_args()

    if args.generate_sample:
        from generate_sample import generate_sample_data
        from data_loader import _save
        from pathlib import Path
        print("Generating synthetic Visa/Mastercard-style data...")
        auth_df, fraud_df = generate_sample_data()
        _save(auth_df, fraud_df, Path(__file__).parent / "data")
        return

    run(max_iter=args.max_iter, dataset=args.dataset)


if __name__ == "__main__":
    main()
