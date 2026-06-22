# Fraud Detection Problem Statement

## Objective
Detect fraudulent payment card transactions in real-time at the point of authorization.

## Business Context
A payment network (Visa/Mastercard-style) processes ~50,000 transactions per day.
Fraud occurs in approximately 5% of transactions, primarily in:
- Card-Not-Present (CNP) / ecommerce channel
- International transactions where the cardholder is not physically present
- Account takeover scenarios where a stolen card is used across multiple merchants

## Available Data at Authorization Time (T+0)
- Transaction details: amount, currency, timestamp, channel, merchant info
- Card/device signals: CVV match, AVS result, 3DS authentication result
- Geographic signals: merchant country, IP country, issuer country
- Historical context: previous transaction timestamp and country

## What We Are NOT Allowed To Use
- Fraud labels (only known 60-120 days after the transaction — reporting delay)
- Any data from the fraud table at inference time
- Future transactions (only look-back windows allowed)

## Key Fraud Patterns to Investigate
1. Card-not-present fraud exploiting weak authentication (no CVV/3DS)
2. Geographic impossibility — same card used in two distant countries minutes apart
3. Velocity attacks — burst of transactions in a short window
4. High-value transactions on new/unusual devices
5. IP country mismatch with merchant or issuer country
6. Unusual merchant category patterns for the card profile

## Success Metric
Features should be:
- Computable at inference time (T+0, using only auth table data)
- Grounded in a testable fraud hypothesis
- Non-redundant with each other
- Expressed as executable PySpark code
