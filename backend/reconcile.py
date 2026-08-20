import pandas as pd
import os
import time


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

payments = pd.read_csv(
    os.path.join(DATA_DIR, "payments.csv")
)

settlements = pd.read_csv(
    os.path.join(DATA_DIR, "settlements.csv")
)

bank = pd.read_csv(
    os.path.join(DATA_DIR, "bank_transactions.csv")
)


# ============================================================
# 2. START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# 3. RECONCILIATION
# ============================================================

results = []


for _, payment in payments.iterrows():

    payment_id = payment["payment_id"]
    payment_amount = payment["amount"]

    # --------------------------------------------------------
    # Find settlement
    # --------------------------------------------------------

    settlement_matches = settlements[
        settlements["payment_id"] == payment_id
    ]

    if settlement_matches.empty:

        results.append({
            "payment_id": payment_id,
            "status": "EXCEPTION",
            "exception_type": "MISSING_SETTLEMENT",
            "confidence": 0,
            "reason": "Payment exists but no settlement was found."
        })

        continue


    settlement = settlement_matches.iloc[0]

    settlement_id = settlement["settlement_id"]
    utr = settlement["utr"]

    gross_amount = settlement["gross_amount"]
    fee = settlement["fee"]
    tax = settlement["tax"]
    net_amount = settlement["net_amount"]


    # --------------------------------------------------------
    # Check payment amount vs settlement amount
    # --------------------------------------------------------

    if payment_amount != gross_amount:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "GROSS_AMOUNT_MISMATCH",
            "confidence": 20,
            "reason": (
                f"Payment amount ₹{payment_amount} "
                f"does not match settlement gross amount "
                f"₹{gross_amount}."
            )
        })

        continue


    # --------------------------------------------------------
    # Check expected settlement calculation
    # --------------------------------------------------------

    expected_net = round(
        gross_amount - fee - tax,
        2
    )

    if expected_net != net_amount:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "SETTLEMENT_CALCULATION_MISMATCH",
            "confidence": 30,
            "reason": (
                f"Expected net amount ₹{expected_net}, "
                f"but settlement reports ₹{net_amount}."
            )
        })

        continue


    # --------------------------------------------------------
    # Find bank transaction using UTR
    # --------------------------------------------------------

    bank_matches = bank[
        bank["utr"] == utr
    ]


    if bank_matches.empty:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "MISSING_BANK_TRANSACTION",
            "confidence": 40,
            "reason": (
                f"No bank transaction found for UTR {utr}."
            )
        })

        continue


    # --------------------------------------------------------
    # Detect duplicate bank transactions
    # --------------------------------------------------------

    if len(bank_matches) > 1:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "DUPLICATE_BANK_TRANSACTION",
            "confidence": 40,
            "reason": (
                f"Multiple bank transactions found "
                f"for UTR {utr}."
            )
        })

        continue


    bank_transaction = bank_matches.iloc[0]

    bank_id = bank_transaction["bank_id"]
    bank_credit = bank_transaction["credit"]


    # --------------------------------------------------------
    # Check bank amount
    # --------------------------------------------------------

    if round(bank_credit, 2) != round(net_amount, 2):

        difference = round(
            net_amount - bank_credit,
            2
        )

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "bank_id": bank_id,
            "status": "EXCEPTION",
            "exception_type": "BANK_AMOUNT_MISMATCH",
            "confidence": 60,
            "reason": (
                f"Expected bank credit ₹{net_amount}, "
                f"but received ₹{bank_credit}. "
                f"Difference: ₹{difference}."
            )
        })

        continue


    # --------------------------------------------------------
    # Check settlement date vs bank date
    # --------------------------------------------------------

    settlement_date = pd.to_datetime(
        settlement["date"]
    )

    bank_date = pd.to_datetime(
        bank_transaction["date"]
    )

    date_difference = abs(
        (bank_date - settlement_date).days
    )


    if date_difference > 3:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "bank_id": bank_id,
            "status": "EXCEPTION",
            "exception_type": "DATE_MISMATCH",
            "confidence": 70,
            "reason": (
                f"Settlement date: "
                f"{settlement_date.date()}, "
                f"Bank date: "
                f"{bank_date.date()}."
            )
        })

        continue


    # --------------------------------------------------------
    # Everything matches
    # --------------------------------------------------------

    results.append({
        "payment_id": payment_id,
        "settlement_id": settlement_id,
        "bank_id": bank_id,
        "status": "MATCHED",
        "exception_type": "",
        "confidence": 100,
        "reason": (
            "Payment, settlement and bank transaction "
            "all match successfully."
        )
    })


# ============================================================
# 4. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

total_records = len(results_df)

matched_records = len(
    results_df[
        results_df["status"] == "MATCHED"
    ]
)

exception_records = len(
    results_df[
        results_df["status"] == "EXCEPTION"
    ]
)

match_rate = (
    matched_records / total_records * 100
)


processing_time = time.time() - start_time

throughput = (
    total_records / processing_time
    if processing_time > 0
    else 0
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("              FINGUARD AI")
print("        RECONCILIATION ENGINE")
print("=" * 60)

print(f"\nTotal records processed : {total_records}")
print(f"Matched records         : {matched_records}")
print(f"Exceptions              : {exception_records}")

print(f"\nMatch rate              : {match_rate:.2f}%")

print(
    f"Processing time         : "
    f"{processing_time:.4f} seconds"
)

print(
    f"Throughput              : "
    f"{throughput:.2f} records/second"
)


# ============================================================
# 7. SHOW EXCEPTIONS
# ============================================================

print("\n")
print("=" * 60)
print("                    EXCEPTIONS")
print("=" * 60)

exceptions = results_df[
    results_df["status"] == "EXCEPTION"
]

for _, row in exceptions.iterrows():

    print(
        f"\n{row['payment_id']} "
        f"→ {row['exception_type']}"
    )

    print(
        f"Reason: {row['reason']}"
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    DATA_DIR,
    "reconciliation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n")
print("=" * 60)
print(f"Results saved to: {results_path}")
print("=" * 60)