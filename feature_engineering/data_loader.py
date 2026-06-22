"""
Configurable data loader.
Priority: Kaggle download (if credentials set) → existing CSVs → synthetic generation.

Set credentials via env vars:
  export KAGGLE_USERNAME=your_username
  export KAGGLE_KEY=your_api_key

Or place ~/.kaggle/kaggle.json with {"username":"...","key":"..."}

Supported Kaggle datasets:
  - ieee-fraud-detection  (competition, richest schema, recommended)
  - mlg-ulb/creditcardfraud  (public dataset, simple schema)
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FE_DIR = Path(__file__).parent

# ── Kaggle dataset targets ─────────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "ieee": {
        "type": "competition",
        "name": "ieee-fraud-detection",
        "files": ["train_transaction.csv", "train_identity.csv"],
        "description": "IEEE-CIS Fraud Detection (590k transactions, rich schema)",
    },
    "creditcard": {
        "type": "dataset",
        "name": "mlg-ulb/creditcardfraud",
        "files": ["creditcard.csv"],
        "description": "ULB Credit Card Fraud (284k transactions, PCA features)",
    },
}


def _has_kaggle_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _setup_kaggle_credentials():
    """Write credentials from env vars to ~/.kaggle/kaggle.json if needed."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        creds = {"username": username, "key": key}
        cred_path = kaggle_dir / "kaggle.json"
        cred_path.write_text(json.dumps(creds))
        cred_path.chmod(0o600)
        return True
    return False


def download_ieee_fraud(target_dir: Path) -> tuple:
    """Download and adapt IEEE-CIS Fraud Detection dataset."""
    import subprocess, zipfile, tempfile

    print("Downloading IEEE-CIS Fraud Detection dataset from Kaggle...")
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", "ieee-fraud-detection", "-p", tmp],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

        # Unzip
        zip_path = Path(tmp) / "ieee-fraud-detection.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp)

        txn_path = next(Path(tmp).rglob("train_transaction.csv"), None)
        id_path = next(Path(tmp).rglob("train_identity.csv"), None)

        if txn_path is None:
            raise FileNotFoundError("train_transaction.csv not found in download")

        print("  Adapting IEEE-CIS schema to auth/fraud format...")
        txn_df = pd.read_csv(txn_path)
        id_df = pd.read_csv(id_path) if id_path else pd.DataFrame()

        auth_df, fraud_df = _adapt_ieee_schema(txn_df, id_df)
        _save(auth_df, fraud_df, target_dir)
        return auth_df, fraud_df


def _adapt_ieee_schema(txn_df: pd.DataFrame, id_df: pd.DataFrame) -> tuple:
    """Map IEEE-CIS columns to our auth/fraud schema."""
    if not id_df.empty and "TransactionID" in id_df.columns:
        df = txn_df.merge(id_df, on="TransactionID", how="left")
    else:
        df = txn_df.copy()

    # Build auth_df
    auth = pd.DataFrame()
    auth["transaction_id"] = df["TransactionID"].astype(str)
    auth["card_id"] = df.get("card1", pd.Series("UNKNOWN", index=df.index)).astype(str)
    auth["user_id"] = df.get("card2", pd.Series("UNKNOWN", index=df.index)).astype(str)
    auth["merchant_id"] = df.get("ProductCD", pd.Series("W", index=df.index)).astype(str)
    auth["merchant_category_code"] = df.get("card4", pd.Series("other", index=df.index))
    auth["merchant_category_desc"] = df.get("ProductCD", pd.Series("unknown", index=df.index))
    auth["transaction_amount"] = df["TransactionAmt"].fillna(0)
    auth["transaction_currency"] = "USD"

    # Convert TransactionDT (seconds from reference) to timestamp
    base = pd.Timestamp("2017-12-01")
    auth["transaction_timestamp"] = pd.to_datetime(
        df["TransactionDT"], unit="s", origin=base
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    auth["channel"] = df.get("DeviceType", pd.Series("unknown", index=df.index)).map(
        {"mobile": "mobile", "desktop": "ecom"}
    ).fillna("ecom")
    auth["card_present"] = 0  # CNP dataset
    auth["cvv_match"] = df.get("card6", pd.Series("credit", index=df.index)).map(
        {"credit": 1, "debit": 1}
    ).fillna(1).astype(int)
    auth["avs_result"] = df.get("addr1", pd.Series(np.nan, index=df.index)).apply(
        lambda x: "Y" if pd.notna(x) else "N"
    )
    auth["threeds_result"] = "U"
    auth["ip_country"] = df.get("id_15", pd.Series("NEW", index=df.index)).map(
        {"Found": "USA", "New": "OTHER", "Unknown": None}
    )
    auth["device_fingerprint"] = df.get("DeviceInfo", pd.Series(np.nan, index=df.index)).astype(str)
    auth["response_code"] = "approved"
    auth["issuer_country"] = "USA"
    auth["merchant_country"] = df.get("addr2", pd.Series("USA", index=df.index)).astype(str)
    auth["is_international"] = (auth["merchant_country"] != auth["issuer_country"]).astype(int)
    auth["merchant_city"] = df.get("addr1", pd.Series("UNKNOWN", index=df.index)).astype(str)
    auth["transaction_type"] = "online"

    # Derive previous_txn info from TransactionDT ordering
    auth["previous_txn_timestamp"] = auth["transaction_timestamp"]
    auth["previous_txn_country"] = auth["merchant_country"]

    # Build fraud_df from isFraud label
    fraud_mask = df["isFraud"] == 1
    fraud_txns = df[fraud_mask]
    fraud = pd.DataFrame({
        "transaction_id": fraud_txns["TransactionID"].astype(str),
        "fraud_type": "card_not_present",
        "reported_date": (
            pd.to_datetime(fraud_txns["TransactionDT"], unit="s", origin=pd.Timestamp("2017-12-01"))
            + pd.Timedelta(days=90)
        ).dt.strftime("%Y-%m-%d"),
        "confirmed": True,
        "fraud_amount": fraud_txns["TransactionAmt"].fillna(0),
    })

    return auth, fraud


def download_creditcard_fraud(target_dir: Path) -> tuple:
    """Download and adapt ULB Credit Card Fraud dataset."""
    import subprocess, zipfile, tempfile

    print("Downloading Credit Card Fraud Detection dataset from Kaggle...")
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", tmp],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

        zip_path = Path(tmp) / "creditcardfraud.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp)

        csv_path = next(Path(tmp).rglob("creditcard.csv"), None)
        if csv_path is None:
            raise FileNotFoundError("creditcard.csv not found")

        print("  Adapting ULB Credit Card schema to auth/fraud format...")
        df = pd.read_csv(csv_path)
        auth_df, fraud_df = _adapt_creditcard_schema(df)
        _save(auth_df, fraud_df, target_dir)
        return auth_df, fraud_df


def _adapt_creditcard_schema(df: pd.DataFrame) -> tuple:
    """Map ULB creditcard.csv to auth/fraud schema."""
    base = pd.Timestamp("2013-09-01")
    auth = pd.DataFrame()
    auth["transaction_id"] = [f"TXN{i:07d}" for i in range(len(df))]
    auth["card_id"] = [f"CARD{i % 5000:06d}" for i in range(len(df))]
    auth["user_id"] = [f"U{i % 3000:06d}" for i in range(len(df))]
    auth["merchant_id"] = [f"M{i % 1000:05d}" for i in range(len(df))]
    auth["merchant_category_code"] = 4816
    auth["merchant_category_desc"] = "ecom_services"
    auth["transaction_amount"] = df["Amount"].abs()
    auth["transaction_currency"] = "EUR"
    auth["transaction_timestamp"] = (
        base + pd.to_timedelta(df["Time"], unit="s")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    auth["channel"] = "ecom"
    auth["card_present"] = 0
    auth["cvv_match"] = 1
    auth["avs_result"] = "Y"
    auth["threeds_result"] = "U"
    auth["ip_country"] = "FRA"
    auth["device_fingerprint"] = np.nan
    auth["response_code"] = "approved"
    auth["issuer_country"] = "FRA"
    auth["merchant_country"] = "FRA"
    auth["is_international"] = 0
    auth["merchant_city"] = "PARIS"
    auth["transaction_type"] = "online"
    auth["previous_txn_timestamp"] = auth["transaction_timestamp"]
    auth["previous_txn_country"] = "FRA"

    # PCA features V1-V28 become auxiliary columns
    for col in [c for c in df.columns if c.startswith("V")]:
        auth[col] = df[col]

    fraud_mask = df["Class"] == 1
    fraud = pd.DataFrame({
        "transaction_id": auth.loc[fraud_mask, "transaction_id"].values,
        "fraud_type": "card_not_present",
        "reported_date": (
            base + pd.to_timedelta(df.loc[fraud_mask, "Time"], unit="s") + pd.Timedelta(days=90)
        ).dt.strftime("%Y-%m-%d"),
        "confirmed": True,
        "fraud_amount": df.loc[fraud_mask, "Amount"].abs().values,
    })

    return auth, fraud


def _save(auth_df: pd.DataFrame, fraud_df: pd.DataFrame, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    auth_df.to_csv(target_dir / "auth.csv", index=False)
    fraud_df.to_csv(target_dir / "fraud.csv", index=False)
    print(f"  Saved auth.csv  : {len(auth_df):,} rows")
    print(f"  Saved fraud.csv : {len(fraud_df):,} rows  ({len(fraud_df)/len(auth_df)*100:.1f}% fraud)")


def load_or_download(dataset: str = "ieee") -> tuple:
    """
    Load data using best available source:
      1. Existing auth.csv / fraud.csv in data/
      2. Kaggle download (if credentials set)
      3. Synthetic generation fallback

    Args:
        dataset: "ieee" | "creditcard"  (only used if downloading)
    """
    auth_path = DATA_DIR / "auth.csv"
    fraud_path = DATA_DIR / "fraud.csv"

    # Already have data
    if auth_path.exists() and fraud_path.exists():
        print(f"Loading existing data from {DATA_DIR}/")
        return pd.read_csv(auth_path), pd.read_csv(fraud_path)

    # Try Kaggle download
    if _has_kaggle_credentials():
        _setup_kaggle_credentials()
        try:
            if dataset == "creditcard":
                return download_creditcard_fraud(DATA_DIR)
            else:
                return download_ieee_fraud(DATA_DIR)
        except Exception as e:
            print(f"Kaggle download failed ({e}). Falling back to synthetic data.")

    # Fallback: synthetic
    print("No Kaggle credentials found. Generating synthetic data...")
    from generate_sample import generate_sample_data
    auth_df, fraud_df = generate_sample_data()
    _save(auth_df, fraud_df, DATA_DIR)
    return auth_df, fraud_df
