import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os


# CONFIGURATION

NUM_TRANSACTIONS = 500

# Number of records for each exception type
EXCEPTIONS_PER_TYPE = 10

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

fake = Faker()


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data"
)

os.makedirs(DATA_DIR, exist_ok=True)


# GENERATE PAYMENTS

payments = []

start_date = datetime(2026, 8, 1)

amount_options = [
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
]


for i in range(1, NUM_TRANSACTIONS + 1):

    payment_id = f"PAY{i:04d}"
    order_id = f"ORD{i:04d}"

    date = start_date + timedelta(
        days=random.randint(0, 10)
    )

    amount = random.choice(
        amount_options
    )

    payments.append({
        "payment_id": payment_id,
        "order_id": order_id,
        "date": date.strftime("%Y-%m-%d"),
        "amount": amount,
        "status": "success",
        "customer": fake.name()
    })


payments_df = pd.DataFrame(payments)


# GENERATE SETTLEMENTS

settlements = []

for i in range(1, NUM_TRANSACTIONS + 1):

    payment = payments[i - 1]

    payment_id = payment["payment_id"]

    gross_amount = payment["amount"]

    fee = round(
        gross_amount * 0.02,
        2
    )

    tax = round(
        fee * 0.18,
        2
    )

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


settlements_df = pd.DataFrame(
    settlements
)


# GENERATE BANK TRANSACTIONS

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


bank_df = pd.DataFrame(
    bank_transactions
)


# GENERATE LEDGER

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


ledger_df = pd.DataFrame(
    ledger
)


# CREATE GROUND TRUTH

ground_truth = []

for i in range(
    1,
    NUM_TRANSACTIONS + 1
):

    ground_truth.append({
        "payment_id": f"PAY{i:04d}",
        "expected_status": "MATCHED",
        "expected_exception": ""
    })


ground_truth_df = pd.DataFrame(
    ground_truth
)


# SELECT RANDOM EXCEPTION RECORDS

all_indices = list(
    range(NUM_TRANSACTIONS)
)

random.shuffle(all_indices)


required_exception_records = (
    EXCEPTIONS_PER_TYPE * 6
)

selected_indices = all_indices[
    :required_exception_records
]


exception_groups = {}

position = 0


exception_types = [
    "MISSING_SETTLEMENT",
    "BANK_AMOUNT_MISMATCH",
    "UTR_MISMATCH",
    "DUPLICATE_BANK_TRANSACTION",
    "DATE_MISMATCH",
    "FEE_MISMATCH"
]


for exception_type in exception_types:

    indices = selected_indices[
        position:
        position + EXCEPTIONS_PER_TYPE
    ]

    exception_groups[
        exception_type
    ] = indices

    position += EXCEPTIONS_PER_TYPE


# 1. MISSING SETTLEMENT

missing_settlement_ids = []

for index in exception_groups[
    "MISSING_SETTLEMENT"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    missing_settlement_ids.append(
        payment_id
    )


settlements_df = settlements_df[
    ~settlements_df["payment_id"].isin(
        missing_settlement_ids
    )
].copy()


for payment_id in missing_settlement_ids:

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "MISSING_SETTLEMENT"
    ]


# 2. BANK AMOUNT MISMATCH

amount_mismatch_bank_ids = []

for index in exception_groups[
    "BANK_AMOUNT_MISMATCH"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    settlement = settlements[
        index
    ]

    bank_id = f"BANK{index + 1:04d}"

    amount_mismatch_bank_ids.append(
        bank_id
    )

    bank_df.loc[
        bank_df["bank_id"] == bank_id,
        "credit"
    ] -= 400

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "BANK_AMOUNT_MISMATCH"
    ]


# 3. UTR MISMATCH

for index in exception_groups[
    "UTR_MISMATCH"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    bank_id = f"BANK{index + 1:04d}"



    bank_df.loc[
        bank_df["bank_id"] == bank_id,
        "utr"
    ] = f"WRONG{index + 1:06d}"

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "UTR_MISMATCH"
    ]


# 4. DUPLICATE BANK TRANSACTION

duplicate_count = 1

for index in exception_groups[
    "DUPLICATE_BANK_TRANSACTION"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    bank_id = f"BANK{index + 1:04d}"

    original = bank_df[
        bank_df["bank_id"] == bank_id
    ].copy()

    if not original.empty:

        duplicate = original.copy()

        duplicate["bank_id"] = (
            f"BANKD{duplicate_count:04d}"
        )

        bank_df = pd.concat(
            [
                bank_df,
                duplicate
            ],
            ignore_index=True
        )

        duplicate_count += 1

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "DUPLICATE_BANK_TRANSACTION"
    ]


# 5. DATE MISMATCH

for index in exception_groups[
    "DATE_MISMATCH"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    bank_id = f"BANK{index + 1:04d}"

    bank_df.loc[
        bank_df["bank_id"] == bank_id,
        "date"
    ] = "2026-08-20"

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "DATE_MISMATCH"
    ]


# 6. FEE MISMATCH

for index in exception_groups[
    "FEE_MISMATCH"
]:

    payment_id = payments_df.iloc[
        index
    ]["payment_id"]

    settlement_id = f"SET{index + 1:04d}"

    settlements_df.loc[
        settlements_df["settlement_id"]
        == settlement_id,
        "fee"
    ] += 300

    settlements_df.loc[
        settlements_df["settlement_id"]
        == settlement_id,
        "net_amount"
    ] -= 300

    ground_truth_df.loc[
        ground_truth_df["payment_id"]
        == payment_id,
        ["expected_status",
         "expected_exception"]
    ] = [
        "EXCEPTION",
        "FEE_MISMATCH"
    ]


# SAVE FILES

payments_df.to_csv(
    os.path.join(
        DATA_DIR,
        "payments.csv"
    ),
    index=False
)

settlements_df.to_csv(
    os.path.join(
        DATA_DIR,
        "settlements.csv"
    ),
    index=False
)

bank_df.to_csv(
    os.path.join(
        DATA_DIR,
        "bank_transactions.csv"
    ),
    index=False
)

ledger_df.to_csv(
    os.path.join(
        DATA_DIR,
        "ledger.csv"
    ),
    index=False
)

ground_truth_df.to_csv(
    os.path.join(
        DATA_DIR,
        "ground_truth.csv"
    ),
    index=False
)


# SUMMARY

print("\n")
print("=" * 65)
print("              FINGUARD AI")
print("        BENCHMARK V2 DATA GENERATOR")
print("=" * 65)

print(
    f"\nPayments generated      : "
    f"{len(payments_df)}"
)

print(
    f"Settlements generated   : "
    f"{len(settlements_df)}"
)

print(
    f"Bank transactions       : "
    f"{len(bank_df)}"
)

print(
    f"Ledger records          : "
    f"{len(ledger_df)}"
)

print(
    f"Ground truth records    : "
    f"{len(ground_truth_df)}"
)


print("\nInjected exceptions:")

for exception_type in exception_types:

    print(
        f"{exception_type:<30} : "
        f"{len(exception_groups[exception_type])}"
    )


print("\nTotal intentional exceptions:")

print(
    f"{required_exception_records}"
)


print("\nData saved to:")

print(DATA_DIR)

print("=" * 65)