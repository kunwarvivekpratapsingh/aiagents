"""
Generates realistic Visa/Mastercard-style synthetic data for testing.
Run: python generate_sample.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def generate_sample_data(n_txns: int = 50_000, fraud_rate: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)

    countries = ["USA", "GBR", "DEU", "FRA", "CAN", "AUS", "IND", "CHN", "BRA", "MEX"]
    channels = ["pos", "ecom", "mobile", "atm"]
    mccs = {
        5411: "grocery", 5812: "restaurant", 4816: "ecom_services",
        5999: "misc_retail", 6011: "atm_cash", 7011: "hotel",
        5912: "pharmacy", 4111: "transport", 5732: "electronics", 5094: "jewelry",
    }
    fraud_types_list = ["card_not_present", "account_takeover", "counterfeit", "first_party"]

    n_users, n_merchants, n_devices, n_cards = 5000, 2000, 8000, 6000
    user_pool = [f"U{i:06d}" for i in range(n_users)]
    merchant_pool = [f"M{i:05d}" for i in range(n_merchants)]
    device_pool = [f"DEV{i:07d}" for i in range(n_devices)]
    card_pool = [f"CARD{i:07d}" for i in range(n_cards)]
    mcc_list = list(mccs.keys())

    is_fraud = rng.random(n_txns) < fraud_rate
    n_fraud = int(is_fraud.sum())
    n_legit = n_txns - n_fraud

    # ── channels ──────────────────────────────────────────────────────────────
    ch_fraud = rng.choice(channels, n_fraud, p=[0.20, 0.65, 0.10, 0.05])
    ch_legit = rng.choice(channels, n_legit, p=[0.55, 0.25, 0.15, 0.05])
    channels_arr = np.empty(n_txns, dtype=object)
    channels_arr[is_fraud] = ch_fraud
    channels_arr[~is_fraud] = ch_legit

    # ── amounts ───────────────────────────────────────────────────────────────
    amounts = np.empty(n_txns)
    amounts[is_fraud] = np.clip(np.exp(rng.normal(5.5, 1.2, n_fraud)), 1, 50_000)
    amounts[~is_fraud] = np.clip(np.exp(rng.normal(4.0, 0.8, n_legit)), 0.5, 10_000)
    amounts = np.round(amounts, 2)

    # ── countries ─────────────────────────────────────────────────────────────
    issuer_p = [0.45, 0.10, 0.08, 0.08, 0.07, 0.05, 0.05, 0.05, 0.04, 0.03]
    issuer_countries = rng.choice(countries, n_txns, p=issuer_p)

    same_country_p = np.where(is_fraud, 0.45, 0.90)
    use_same = rng.random(n_txns) < same_country_p
    merchant_countries = np.where(use_same, issuer_countries, rng.choice(countries, n_txns))

    # ── authentication signals ────────────────────────────────────────────────
    cvv_match = (rng.random(n_txns) < np.where(is_fraud, 0.35, 0.93)).astype(int)

    avs_opts = ["Y", "N", "P", "Z"]
    avs_fraud_p = [0.25, 0.40, 0.20, 0.15]
    avs_legit_p = [0.75, 0.10, 0.08, 0.07]

    threeds_opts = ["Y", "N", "A", "U"]
    tds_fraud_p = [0.15, 0.40, 0.25, 0.20]
    tds_legit_p = [0.70, 0.08, 0.12, 0.10]

    avs_results = np.empty(n_txns, dtype=object)
    threeds_results = np.empty(n_txns, dtype=object)
    for i in range(n_txns):
        online = channels_arr[i] in ("ecom", "mobile")
        avs_results[i] = rng.choice(avs_opts, p=(avs_fraud_p if is_fraud[i] else avs_legit_p)) if online else None
        threeds_results[i] = rng.choice(threeds_opts, p=(tds_fraud_p if is_fraud[i] else tds_legit_p)) if channels_arr[i] == "ecom" else None

    # ── IP country ────────────────────────────────────────────────────────────
    same_ip_p = np.where(is_fraud, 0.40, 0.85)
    use_same_ip = rng.random(n_txns) < same_ip_p
    ip_countries = np.where(
        np.isin(channels_arr, ["ecom", "mobile"]),
        np.where(use_same_ip, merchant_countries, rng.choice(countries, n_txns)),
        None,
    )

    # ── device fingerprints ───────────────────────────────────────────────────
    device_fps = np.empty(n_txns, dtype=object)
    online_mask = np.isin(channels_arr, ["ecom", "mobile"])
    device_fps[online_mask] = rng.choice(device_pool, online_mask.sum())
    # Fraud devices cluster in a smaller pool (reuse signal)
    fraud_online = is_fraud & online_mask
    if fraud_online.sum() > 0:
        reuse = rng.random(fraud_online.sum()) < 0.30
        new_devs = rng.choice(device_pool[:200], fraud_online.sum())
        existing = device_fps[fraud_online]
        existing[reuse] = new_devs[reuse]
        device_fps[fraud_online] = existing

    # ── timestamps ────────────────────────────────────────────────────────────
    base_ts = datetime(2024, 1, 1)
    day_offsets = rng.integers(0, 365, n_txns)
    hour_offsets = rng.integers(0, 24, n_txns)
    minute_offsets = rng.integers(0, 60, n_txns)
    second_offsets = rng.integers(0, 60, n_txns)

    timestamps = [
        (base_ts + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S")
        for d, h, m, s in zip(day_offsets, hour_offsets, minute_offsets, second_offsets)
    ]

    # ── previous transaction context (velocity features foundation) ───────────
    prev_delta = np.where(is_fraud,
                          rng.integers(30, 601, n_txns),      # fraud: 30s–10m ago
                          rng.integers(3_600, 604_800, n_txns))  # legit: 1h–7d ago
    fraud_fast = is_fraud & (rng.random(n_txns) > 0.35)
    prev_delta[~fraud_fast & is_fraud] = rng.integers(600, 86_400, (~fraud_fast & is_fraud).sum())

    prev_timestamps = [
        (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") - timedelta(seconds=int(d))).strftime("%Y-%m-%d %H:%M:%S")
        for ts, d in zip(timestamps, prev_delta)
    ]

    diff_prev_country = rng.random(n_txns) < np.where(is_fraud, 0.40, 0.05)
    prev_countries = np.where(
        diff_prev_country,
        rng.choice(countries, n_txns),
        merchant_countries,
    )

    # ── other columns ─────────────────────────────────────────────────────────
    txn_ids = [str(uuid.uuid4()) for _ in range(n_txns)]
    user_ids = rng.choice(user_pool, n_txns)
    merchant_ids = rng.choice(merchant_pool, n_txns)
    card_ids = rng.choice(card_pool, n_txns)
    mcc_arr = rng.choice(mcc_list, n_txns)

    card_present = np.where(np.isin(channels_arr, ["ecom", "mobile"]), 0, 1)

    txn_types = np.empty(n_txns, dtype=object)
    for i in range(n_txns):
        if channels_arr[i] == "ecom":
            txn_types[i] = "online"
        elif channels_arr[i] == "atm":
            txn_types[i] = "pin"
        else:
            txn_types[i] = rng.choice(["chip", "swipe", "contactless"], p=[0.55, 0.15, 0.30])

    auth_df = pd.DataFrame({
        "transaction_id": txn_ids,
        "card_id": card_ids,
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "merchant_category_code": mcc_arr,
        "merchant_category_desc": [mccs[m] for m in mcc_arr],
        "merchant_country": merchant_countries,
        "merchant_city": [f"CITY_{rng.integers(1, 500)}" for _ in range(n_txns)],
        "transaction_amount": amounts,
        "transaction_currency": "USD",
        "transaction_timestamp": timestamps,
        "transaction_type": txn_types,
        "card_present": card_present,
        "cvv_match": cvv_match,
        "avs_result": avs_results,
        "threeds_result": threeds_results,
        "ip_country": ip_countries,
        "device_fingerprint": device_fps,
        "response_code": rng.choice(["approved", "declined", "referred"], n_txns, p=[0.90, 0.07, 0.03]),
        "channel": channels_arr,
        "is_international": (merchant_countries != issuer_countries).astype(int),
        "issuer_country": issuer_countries,
        "previous_txn_timestamp": prev_timestamps,
        "previous_txn_country": prev_countries,
    })

    # ── fraud table ───────────────────────────────────────────────────────────
    fraud_idx = np.where(is_fraud)[0]
    fraud_txn_ids = np.array(txn_ids)[fraud_idx]
    fraud_amounts = amounts[fraud_idx]
    fraud_ch = channels_arr[fraud_idx]
    fraud_ts = np.array(timestamps)[fraud_idx]

    fraud_type_arr = []
    for ch in fraud_ch:
        if ch == "ecom":
            ft = rng.choice(fraud_types_list, p=[0.60, 0.20, 0.10, 0.10])
        elif ch == "mobile":
            ft = rng.choice(fraud_types_list, p=[0.20, 0.50, 0.15, 0.15])
        elif ch == "atm":
            ft = rng.choice(fraud_types_list, p=[0.05, 0.10, 0.70, 0.15])
        else:
            ft = rng.choice(fraud_types_list, p=[0.10, 0.25, 0.50, 0.15])
        fraud_type_arr.append(ft)

    reported_dates = [
        (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") + timedelta(days=int(rng.integers(60, 121)))).strftime("%Y-%m-%d")
        for ts in fraud_ts
    ]

    fraud_df = pd.DataFrame({
        "transaction_id": fraud_txn_ids,
        "fraud_type": fraud_type_arr,
        "reported_date": reported_dates,
        "confirmed": True,
        "fraud_amount": fraud_amounts,
    })

    return auth_df, fraud_df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating synthetic Visa/Mastercard-style data...")
    auth_df, fraud_df = generate_sample_data(n_txns=50_000, fraud_rate=0.05)

    auth_path = os.path.join(DATA_DIR, "auth.csv")
    fraud_path = os.path.join(DATA_DIR, "fraud.csv")
    auth_df.to_csv(auth_path, index=False)
    fraud_df.to_csv(fraud_path, index=False)

    print(f"  auth.csv   : {len(auth_df):,} transactions  →  {auth_path}")
    print(f"  fraud.csv  : {len(fraud_df):,} confirmed fraud cases  →  {fraud_path}")
    print(f"  fraud rate : {len(fraud_df)/len(auth_df)*100:.1f}%")
    print(f"  columns    : {list(auth_df.columns)}")
    print("\nDone. Now run: python run_loop.py")
