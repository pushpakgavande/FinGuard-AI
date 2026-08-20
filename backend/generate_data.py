import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

NUM_TRANSACTIONS = 100

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------
# GENERATE PAYMENTS
# ---------------------------------------------------

payments = []

start_date = datetime(2026, 8, 1)

for i in range(1, NUM_TRANSACTIONS + 1):

    payment_id = f"PAY{i:04d}"
    order_id = f"ORD{i:04d}"

    date = start_date + timedelta(
        days=random.randint(0, 10)
    )

    amount = random.choice([
        500,
        750,
        1000,
        1500,
        2500,
        5000,
        7500,
        10000,
        15000,
        20000
    ])

    payments.append({
        "payment_id": payment_id,
        "order_id": order_id,
        "date": date.strftime("%Y-%m-%d"),
        "amount": amount,
        "status": "success",
        "customer": fake.name()
    })


payments_df = pd.DataFrame(payments)


# ---------------------------------------------------
# GENERATE SETTLEMENTS
# ---------------------------------------------------

settlements = []

for i in range(1, NUM_TRANSACTIONS + 1):

    payment = payments[i - 1]

    payment_id = payment["payment_id"]

    gross_amount = payment["amount"]

    fee = round(gross_amount * 0.02, 2)

    tax = round(fee * 0.18, 2)

    net_amount = round(
        gross_amount - fee - tax,
        2
    )

    settlement_date = (
        datetime.strptime(
            payment["date"],
            "%Y-%m-%d"
        )
        + timedelta(days=2)
    )

    settlement_id = f"SET{i:04d}"

    utr = f"UTR{i:06d}"

    settlements.append({
        "settlement_id": settlement_id,
        "payment_id": payment_id,
        "utr": utr,
        "date": settlement_date.strftime("%Y-%m-%d"),
        "gross_amount": gross_amount,
        "fee": fee,
        "tax": tax,
        "net_amount": net_amount
    })


settlements_df = pd.DataFrame(settlements)


# ---------------------------------------------------
# GENERATE BANK TRANSACTIONS
# ---------------------------------------------------

bank_transactions = []

for i in range(1, NUM_TRANSACTIONS + 1):

    settlement = settlements[i - 1]

    bank_id = f"BANK{i:04d}"

    bank_transactions.append({
        "bank_id": bank_id,
        "utr": settlement["utr"],
        "date": settlement["date"],
        "credit": settlement["net_amount"]
    })


bank_df = pd.DataFrame(bank_transactions)


# ---------------------------------------------------
# GENERATE LEDGER
# ---------------------------------------------------

ledger = []

for i in range(1, NUM_TRANSACTIONS + 1):

    payment = payments[i - 1]

    ledger_id = f"LED{i:04d}"

    ledger.append({
        "ledger_id": ledger_id,
        "payment_id": payment["payment_id"],
        "date": payment["date"],
        "type": "CREDIT",
        "amount": payment["amount"]
    })


ledger_df = pd.DataFrame(ledger)


# ---------------------------------------------------
# CREATE INTENTIONAL EXCEPTIONS
# ---------------------------------------------------

# 1. Missing settlement
settlements_df = settlements_df[
    settlements_df["payment_id"] != "PAY0020"
]


# 2. Amount mismatch
bank_df.loc[
    bank_df["bank_id"] == "BANK0030",
    "credit"
] -= 400


# 3. Wrong UTR
bank_df.loc[
    bank_df["bank_id"] == "BANK0040",
    "utr"
] = "UTR999999"


# 4. Duplicate bank transaction
duplicate = bank_df[
    bank_df["bank_id"] == "BANK0050"
].copy()

duplicate["bank_id"] = "BANK0101"

bank_df = pd.concat(
    [bank_df, duplicate],
    ignore_index=True
)


# 5. Date mismatch
bank_df.loc[
    bank_df["bank_id"] == "BANK0060",
    "date"
] = "2026-08-20"


# 6. Settlement fee mismatch
settlements_df.loc[
    settlements_df["settlement_id"] == "SET0070",
    "fee"
] += 300

settlements_df.loc[
    settlements_df["settlement_id"] == "SET0070",
    "net_amount"
] -= 300


# ---------------------------------------------------
# SAVE FILES
# ---------------------------------------------------

payments_df.to_csv(
    os.path.join(DATA_DIR, "payments.csv"),
    index=False
)

settlements_df.to_csv(
    os.path.join(DATA_DIR, "settlements.csv"),
    index=False
)

bank_df.to_csv(
    os.path.join(DATA_DIR, "bank_transactions.csv"),
    index=False
)

ledger_df.to_csv(
    os.path.join(DATA_DIR, "ledger.csv"),
    index=False
)


# ---------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------

print("\n====================================")
print("   FINGUARD AI DATA GENERATOR")
print("====================================")

print(f"\nPayments generated:     {len(payments_df)}")
print(f"Settlements generated: {len(settlements_df)}")
print(f"Bank transactions:     {len(bank_df)}")
print(f"Ledger records:         {len(ledger_df)}")

print("\nIntentional exceptions:")
print("1. Missing settlement      → PAY0020")
print("2. Amount mismatch         → BANK0030")
print("3. Wrong UTR               → BANK0040")
print("4. Duplicate transaction   → BANK0050")
print("5. Date mismatch           → BANK0060")
print("6. Fee mismatch            → SET0070")

print("\nData saved to:")
print(DATA_DIR)

print("\n====================================")