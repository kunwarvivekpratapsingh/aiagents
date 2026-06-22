"""
End-to-end test for the auto-improving feature engineering system.

Tests the full pipeline with pre-written PySpark feature sets so the
complete loop (data → L1/L2/L3 validate → git keep/revert → catalog) can
be verified without an Anthropic API key.

Run: python test_e2e.py
"""

import os
import sys
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

FE_DIR = Path(__file__).parent
REPO_ROOT = FE_DIR.parent

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):print(f"  {RED}✗{RESET} {msg}")
def info(msg):print(f"  {CYAN}→{RESET} {msg}")
def head(msg):print(f"\n{BOLD}{CYAN}{msg}{RESET}")


# ── Pre-written feature sets (simulate what Claude would generate) ─────────────
# Each entry: (description, expected_keep, features_py_content)

MOCK_FEATURE_SETS = [

    # ── SHOULD KEEP: score 8/10 ─────────────────────────────────────────────
    ("H1: CVV failure + ecom channel = CNP fraud signal", True, '''\
"""
HYPOTHESIS: CVV failure combined with ecommerce channel is a strong card-not-present fraud signal
RATIONALE: Legitimate cardholders know their CVV; failures in online channels indicate stolen card data
SUB_HYPOTHESES:
  - SH1: CVV mismatch on ecom channel is direct fraud indicator
  - SH2: Multiple authentication failures compound risk
  - SH3: AVS mismatch alongside CVV failure confirms stolen credentials
FEATURES: cvv_fail_ecom, auth_failure_score, cvv_avs_double_fail
TEMPORAL_WINDOWS: T+0
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pyspark.sql.types as T


def engineer_features(auth_df: DataFrame) -> DataFrame:
    # SH1: CVV mismatch on ecom channel
    auth_df = auth_df.withColumn(
        "cvv_fail_ecom",
        F.when(
            (F.col("cvv_match") == 0) & (F.col("channel") == "ecom"), 1
        ).otherwise(0)
    )

    # SH2: Authentication failure score (CVV + 3DS + AVS)
    auth_df = auth_df.withColumn(
        "auth_failure_score",
        (F.col("cvv_match") == 0).cast(T.IntegerType())
        + F.when(F.col("threeds_result").isin("N", "U"), 1).otherwise(0)
        + F.when(F.col("avs_result").isin("N", "Z"), 1).otherwise(0)
    )

    # SH3: CVV and AVS double failure
    auth_df = auth_df.withColumn(
        "cvv_avs_double_fail",
        F.when(
            (F.col("cvv_match") == 0) & (F.col("avs_result").isin("N", "Z")), 1
        ).otherwise(0)
    )

    return auth_df
'''),

    # ── SHOULD KEEP: score 9/10 ─────────────────────────────────────────────
    ("H2: Geographic impossibility — IP country vs issuer country", True, '''\
"""
HYPOTHESIS: IP geolocation country differing from card issuer country signals CNP fraud
RATIONALE: Fraudsters using stolen cards often operate from countries different from the card issuer
SUB_HYPOTHESES:
  - SH1: IP country mismatch with issuer country (direct geo anomaly)
  - SH2: IP country mismatch with merchant country (proxy location mismatch)
  - SH3: Triple mismatch: IP, merchant, and issuer all different countries
FEATURES: ip_issuer_mismatch, ip_merchant_mismatch, triple_country_mismatch
TEMPORAL_WINDOWS: T+0
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pyspark.sql.types as T


def engineer_features(auth_df: DataFrame) -> DataFrame:
    # SH1: IP country vs issuer country mismatch
    auth_df = auth_df.withColumn(
        "ip_issuer_mismatch",
        F.when(
            F.col("ip_country").isNotNull()
            & (F.col("ip_country") != F.col("issuer_country")),
            1
        ).otherwise(0)
    )

    # SH2: IP country vs merchant country mismatch
    auth_df = auth_df.withColumn(
        "ip_merchant_mismatch",
        F.when(
            F.col("ip_country").isNotNull()
            & (F.col("ip_country") != F.col("merchant_country")),
            1
        ).otherwise(0)
    )

    # SH3: Triple country mismatch
    auth_df = auth_df.withColumn(
        "triple_country_mismatch",
        F.when(
            (F.col("ip_issuer_mismatch") == 1)
            & (F.col("ip_merchant_mismatch") == 1)
            & (F.col("is_international") == 1),
            1
        ).otherwise(0)
    )

    return auth_df
'''),

    # ── SHOULD REVERT: uses a fraud column (leakage) ─────────────────────────
    ("H3 [BAD — fraud leakage]: uses confirmed column", False, '''\
"""
HYPOTHESIS: Confirmed fraud transactions have distinct patterns
RATIONALE: Using actual fraud labels to build features
SUB_HYPOTHESES:
  - SH1: Fraud confirmed flag
FEATURES: is_confirmed_fraud
TEMPORAL_WINDOWS: T+0
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def engineer_features(auth_df: DataFrame) -> DataFrame:
    # This is WRONG — confirmed is a fraud label (leakage)
    auth_df = auth_df.withColumn(
        "is_confirmed_fraud",
        F.col("confirmed").cast("int")
    )
    return auth_df
'''),

    # ── SHOULD REVERT: uses future window ────────────────────────────────────
    ("H4 [BAD — future window]: rangeBetween positive upper bound", False, '''\
"""
HYPOTHESIS: Velocity around the transaction time
RATIONALE: Burst transactions indicate fraud
SUB_HYPOTHESES:
  - SH1: Transaction count in a window
FEATURES: txn_count_window
TEMPORAL_WINDOWS: T-10m to T+10m (WRONG!)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def engineer_features(auth_df: DataFrame) -> DataFrame:
    w = Window.partitionBy("user_id").orderBy(
        F.col("transaction_timestamp").cast("long")
    ).rangeBetween(-600, 600)   # BUG: positive upper uses future rows
    auth_df = auth_df.withColumn("txn_count_window", F.count("*").over(w))
    return auth_df
'''),

    # ── SHOULD REVERT: references non-existent column ────────────────────────
    ("H5 [BAD — bad column]: references nonexistent_score column", False, '''\
"""
HYPOTHESIS: External risk score indicates fraud
RATIONALE: Risk scores correlate with fraud
SUB_HYPOTHESES:
  - SH1: High external risk score
FEATURES: high_risk_flag
TEMPORAL_WINDOWS: T+0
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def engineer_features(auth_df: DataFrame) -> DataFrame:
    auth_df = auth_df.withColumn(
        "high_risk_flag",
        (F.col("external_risk_score") > 0.8).cast("int")  # column doesn't exist!
    )
    return auth_df
'''),

    # ── SHOULD KEEP: velocity features using previous_txn columns ───────────
    ("H6: Impossible travel — previous transaction country mismatch", True, '''\
"""
HYPOTHESIS: A card used in a different country shortly after a previous transaction signals impossible travel
RATIONALE: Physical travel between countries takes hours; rapid country changes indicate card compromise
SUB_HYPOTHESES:
  - SH1: Previous transaction country differs from current merchant country
  - SH2: Both current and previous transactions are international
  - SH3: Previous transaction was in a high-risk cross-border pattern
FEATURES: prev_country_mismatch, intl_travel_flag, cross_border_velocity
TEMPORAL_WINDOWS: T+0 (using previous_txn_country column)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pyspark.sql.types as T


def engineer_features(auth_df: DataFrame) -> DataFrame:
    # SH1: Previous transaction country differs from current
    auth_df = auth_df.withColumn(
        "prev_country_mismatch",
        F.when(
            F.col("previous_txn_country").isNotNull()
            & (F.col("previous_txn_country") != F.col("merchant_country")),
            1
        ).otherwise(0)
    )

    # SH2: International travel pattern (both ends are cross-border)
    auth_df = auth_df.withColumn(
        "intl_travel_flag",
        F.when(
            (F.col("prev_country_mismatch") == 1) & (F.col("is_international") == 1),
            1
        ).otherwise(0)
    )

    # SH3: Time delta since previous transaction (seconds)
    auth_df = auth_df.withColumn(
        "seconds_since_prev_txn",
        F.unix_timestamp(F.col("transaction_timestamp"), "yyyy-MM-dd HH:mm:ss")
        - F.unix_timestamp(F.col("previous_txn_timestamp"), "yyyy-MM-dd HH:mm:ss")
    )

    # Cross-border velocity: country changed AND < 3600 seconds since last txn
    auth_df = auth_df.withColumn(
        "cross_border_velocity",
        F.when(
            (F.col("prev_country_mismatch") == 1)
            & (F.col("seconds_since_prev_txn") < 3600)
            & (F.col("seconds_since_prev_txn") > 0),
            1
        ).otherwise(0)
    )

    return auth_df
'''),
]


# ── Mock L3 judge that mirrors expected_keep ───────────────────────────────────
def make_mock_l3_response(expected_keep: bool, description: str):
    score = 8.5 if expected_keep else 2.0
    return {
        "hypothesis_alignment": 3 if expected_keep else 0,
        "feature_operationalization": 2 if expected_keep else 1,
        "temporal_leakage_safety": 2 if expected_keep else 0,
        "code_quality": 1 if expected_keep else 1,
        "novelty": 0.5,
        "total": score,
        "feedback": (
            "Strong features with clear hypothesis-to-feature mapping."
            if expected_keep else
            "Feature set has critical issues preventing approval."
        ),
        "keep_recommendation": expected_keep,
        "layer": "L3",
        "pass": expected_keep,
    }


# ── Test runner ────────────────────────────────────────────────────────────────

def run_test(idx: int, description: str, expected_keep: bool, features_content: str,
             auth_df, problem: str) -> dict:
    from validate import layer1_static, layer2_dryrun

    print(f"\n{'─'*70}")
    print(f"  Test {idx}: {description}")
    print(f"  Expected: {'KEEP' if expected_keep else 'REVERT'}")

    auth_cols = list(auth_df.columns)

    # L1
    l1 = layer1_static(features_content, auth_cols)
    info(f"L1 static  : {'PASS' if l1['pass'] else 'FAIL'} — {l1['reason'][:60]}")

    if not l1["pass"]:
        actual_keep = False
        score = 0.0
        feedback = l1["reason"]
    else:
        # L2
        l2 = layer2_dryrun(features_content, auth_df)
        info(f"L2 dry-run : {'PASS' if l2['pass'] else 'FAIL'} — {l2['reason'][:60]}")

        if not l2["pass"]:
            actual_keep = False
            score = 0.0
            feedback = l2["reason"]
        else:
            # L3 (mocked)
            l3 = make_mock_l3_response(expected_keep, description)
            info(f"L3 LLM     : score={l3['total']}/10 — {l3['feedback'][:55]}")
            actual_keep = l3["pass"]
            score = l3["total"]
            feedback = l3["feedback"]

    # Keep / revert decision
    correct = actual_keep == expected_keep
    if correct:
        ok(f"Decision correct — {'KEPT' if actual_keep else 'REVERTED'} as expected")
    else:
        fail(f"Decision WRONG — got {'KEPT' if actual_keep else 'REVERTED'}, expected {'KEPT' if expected_keep else 'REVERTED'}")

    return {
        "idx": idx,
        "description": description,
        "expected_keep": expected_keep,
        "actual_keep": actual_keep,
        "correct": correct,
        "score": score,
        "feedback": feedback,
    }


def test_git_keep_revert(features_content: str):
    """Verify git commit (keep) and git checkout (revert) work on features.py."""
    features_path = FE_DIR / "features.py"
    original = features_path.read_text()

    # Write new content
    features_path.write_text(features_content)

    # Simulate keep: commit
    subprocess.run(["git", "-C", str(REPO_ROOT), "add", "feature_engineering/features.py"],
                   capture_output=True)
    r = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "commit", "-m", "test: e2e keep simulation"],
        capture_output=True, text=True
    )
    committed = r.returncode == 0

    # Simulate revert
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "checkout", "HEAD~1", "--", "feature_engineering/features.py"],
        capture_output=True
    )
    reverted_content = features_path.read_text()

    # Undo test commit
    subprocess.run(["git", "-C", str(REPO_ROOT), "reset", "--soft", "HEAD~1"],
                   capture_output=True)
    features_path.write_text(original)

    return committed, reverted_content != features_content


def test_catalog_and_log(results: list):
    """Verify results.tsv and feature_catalog.md are written correctly."""
    from run_loop import init_results_tsv, log_result, init_catalog, append_to_catalog

    # Clean up test artifacts
    test_results = FE_DIR / "results_test.tsv"
    test_catalog = FE_DIR / "catalog_test.md"

    import run_loop as rl
    orig_results = rl.RESULTS_PATH
    orig_catalog = rl.CATALOG_PATH
    rl.RESULTS_PATH = test_results
    rl.CATALOG_PATH = test_catalog

    try:
        init_results_tsv()
        init_catalog("Test problem statement")
        for r in results:
            if r["actual_keep"]:
                log_result(r["idx"], "KEPT", r["score"], None,
                           r["description"], ["feature_a", "feature_b"], r["feedback"])
                append_to_catalog(r["idx"], r["score"],
                                  MOCK_FEATURE_SETS[r["idx"] - 1][2], ["feature_a"])
            else:
                log_result(r["idx"], "REVERTED", r["score"], "L1",
                           r["description"], [], r["feedback"])

        rows = test_results.read_text().strip().splitlines()
        catalog_text = test_catalog.read_text()

        return len(rows) - 1, "## Iteration" in catalog_text  # -1 for header
    finally:
        rl.RESULTS_PATH = orig_results
        rl.CATALOG_PATH = orig_catalog
        if test_results.exists(): test_results.unlink()
        if test_catalog.exists(): test_catalog.unlink()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.chdir(FE_DIR)
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Auto-Improving Feature Engineering — End-to-End Test{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    # ── 1. Data loading ────────────────────────────────────────────────────
    head("[ 1 ] Data Loading")
    from prepare import load_data, get_schema_info, schema_to_prompt_text, load_problem_statement
    try:
        auth_df, fraud_df = load_data()
        schema = get_schema_info(auth_df, fraud_df)
        schema_text = schema_to_prompt_text(schema)
        problem = load_problem_statement()
        ok(f"auth.csv loaded    : {auth_df.shape[0]:,} rows × {auth_df.shape[1]} cols")
        ok(f"fraud.csv loaded   : {fraud_df.shape[0]:,} fraud cases ({fraud_df.shape[0]/auth_df.shape[0]*100:.1f}%)")
        ok(f"Schema extracted   : {len(schema['auth']['columns'])} auth columns described")
        ok(f"Problem statement  : {len(problem)} chars")
    except FileNotFoundError as e:
        fail(str(e))
        print("\n  Run: python generate_sample.py  first.")
        sys.exit(1)

    # ── 2. Validation pipeline ─────────────────────────────────────────────
    head("[ 2 ] Validation Pipeline (L1 Static + L2 Dry-run + L3 Mock Judge)")
    results = []
    for i, (desc, expected_keep, content) in enumerate(MOCK_FEATURE_SETS, start=1):
        r = run_test(i, desc, expected_keep, content, auth_df, problem)
        results.append(r)

    # ── 3. Git keep/revert ─────────────────────────────────────────────────
    head("[ 3 ] Git Keep / Revert Mechanics")
    committed, reverted = test_git_keep_revert(MOCK_FEATURE_SETS[0][2])
    if committed:
        ok("git commit (keep) works correctly")
    else:
        fail("git commit failed")
    if reverted:
        ok("git checkout (revert) correctly restores previous features.py")
    else:
        fail("git revert did not restore previous content")

    # ── 4. results.tsv + feature_catalog.md ───────────────────────────────
    head("[ 4 ] Logging — results.tsv + feature_catalog.md")
    logged_rows, catalog_has_sections = test_catalog_and_log(results)
    ok(f"results.tsv         : {logged_rows} rows logged")
    if catalog_has_sections:
        ok("feature_catalog.md  : hypothesis sections written correctly")
    else:
        fail("feature_catalog.md  : missing ## Iteration sections")

    # ── 5. Summary ─────────────────────────────────────────────────────────
    head("[ 5 ] Summary")
    total  = len(results)
    passed = sum(1 for r in results if r["correct"])
    kept   = sum(1 for r in results if r["actual_keep"])
    reverted_count = total - kept

    print(f"\n  Validation tests : {passed}/{total} correct decisions")
    print(f"  KEPT             : {kept}  (should be 3)")
    print(f"  REVERTED         : {reverted_count}  (should be 3)")
    print()
    for r in results:
        sym = f"{GREEN}✓{RESET}" if r["correct"] else f"{RED}✗{RESET}"
        action = "KEPT" if r["actual_keep"] else "REVT"
        print(f"  {sym} [{action}] score={r['score']:4.1f}  {r['description'][:55]}")

    print()
    if passed == total and committed and reverted and catalog_has_sections:
        print(f"  {GREEN}{BOLD}ALL TESTS PASSED ✓{RESET}")
        print()
        print(f"  {CYAN}To run the full live loop:{RESET}")
        print(f"    export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"    python run_loop.py --max-iter 15")
        print()
        print(f"  {CYAN}To use real Kaggle data (IEEE-CIS fraud dataset):{RESET}")
        print(f"    export KAGGLE_USERNAME=... KAGGLE_KEY=...")
        print(f"    python run_loop.py --dataset ieee --max-iter 15")
    else:
        print(f"  {RED}{BOLD}SOME TESTS FAILED ✗{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
