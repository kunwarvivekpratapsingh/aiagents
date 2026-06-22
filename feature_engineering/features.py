"""
THIS IS THE ONLY FILE THE AGENT MODIFIES.
Equivalent to kernel.cu in the auto-improving-kernel pattern.

CONTRACT:
  - Function: engineer_features(auth_df: DataFrame) -> DataFrame
  - Input:  PySpark DataFrame of auth table ONLY (no fraud labels)
  - Output: same DataFrame enriched with new feature columns
  - Rules:  Only look-back window functions (no positive rangeBetween)
            Only reference columns present in auth.csv schema
            Never import or reference fraud_df, is_fraud, confirmed, reported_date

HYPOTHESIS: Basic transaction and authentication signals indicate elevated fraud risk
RATIONALE: High transaction amounts, international transactions, and weak CNP authentication
           are baseline indicators of payment fraud risk
SUB_HYPOTHESES:
  - SH1: Log-scaled amount reduces right skew and exposes relative amount anomalies
  - SH2: Card-not-present transactions without 3DS verification have higher CNP fraud risk
  - SH3: International transactions (merchant country != issuer country) are higher risk
FEATURES: amount_log, ecom_no_3ds, is_international_flag
TEMPORAL_WINDOWS: T+0 (no look-back required for these point-in-time features)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pyspark.sql.types as T


def engineer_features(auth_df: DataFrame) -> DataFrame:
    """
    Input:  auth_df — Spark DataFrame of transaction data only (NO fraud labels)
    Output: auth_df enriched with new feature columns
    """
    # SH1: Log-scaled transaction amount
    auth_df = auth_df.withColumn(
        "amount_log",
        F.log1p(F.col("transaction_amount"))
    )

    # SH2: Ecom channel without 3DS verification (high CNP fraud signal)
    auth_df = auth_df.withColumn(
        "ecom_no_3ds",
        F.when(
            (F.col("channel") == "ecom") & (F.col("threeds_result").isin("N", "U")),
            1
        ).otherwise(0)
    )

    # SH3: International transaction flag
    auth_df = auth_df.withColumn(
        "is_international_flag",
        (F.col("merchant_country") != F.col("issuer_country")).cast(T.IntegerType())
    )

    return auth_df
