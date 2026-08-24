import pandas as pd
import os
import time


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# 2. LOAD DATA
# ============================================================

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
# 3. START TIMER
# ============================================================

start_time = time.time()

results = []


# ============================================================
# 4. RECONCILE EACH PAYMENT
# ============================================================

for _, payment in payments.iterrows():

    payment_id = payment["payment_id"]
    payment_amount = float(payment["amount"])

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
            "confidence": 100,
            "reason": (
                "Payment exists but no settlement "
                "was found."
            )
        })

        continue

    settlement = settlement_matches.iloc[0]

    settlement_id = settlement["settlement_id"]
    utr = str(settlement["utr"])

    gross_amount = float(
        settlement["gross_amount"]
    )

    actual_fee = float(
        settlement["fee"]
    )

    actual_tax = float(
        settlement["tax"]
    )

    actual_net = float(
        settlement["net_amount"]
    )


    # --------------------------------------------------------
    # CHECK 1: PAYMENT vs SETTLEMENT GROSS AMOUNT
    # --------------------------------------------------------

    if round(payment_amount, 2) != round(gross_amount, 2):

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "GROSS_AMOUNT_MISMATCH",
            "confidence": 100,
            "reason": (
                f"Payment amount ₹{payment_amount:.2f} "
                f"does not match settlement gross amount "
                f"₹{gross_amount:.2f}."
            )
        })

        continue


    # --------------------------------------------------------
    # CHECK 2: EXPECTED FEE
    #
    # Our synthetic processor uses:
    # Fee = 2% of gross amount
    # --------------------------------------------------------

    expected_fee = round(
        gross_amount * 0.02,
        2
    )

    if round(actual_fee, 2) != expected_fee:

        fee_difference = round(
            actual_fee - expected_fee,
            2
        )

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "FEE_MISMATCH",
            "confidence": 100,
            "reason": (
                f"Expected fee ₹{expected_fee:.2f}, "
                f"but settlement reports "
                f"₹{actual_fee:.2f}. "
                f"Fee variance: ₹{fee_difference:.2f}."
            )
        })

        continue


    # --------------------------------------------------------
    # CHECK 3: TAX CALCULATION
    #
    # Tax = 18% of fee
    # --------------------------------------------------------

    expected_tax = round(
        expected_fee * 0.18,
        2
    )

    if round(actual_tax, 2) != expected_tax:

        tax_difference = round(
            actual_tax - expected_tax,
            2
        )

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "TAX_MISMATCH",
            "confidence": 100,
            "reason": (
                f"Expected tax ₹{expected_tax:.2f}, "
                f"but settlement reports "
                f"₹{actual_tax:.2f}. "
                f"Tax variance: ₹{tax_difference:.2f}."
            )
        })

        continue


    # --------------------------------------------------------
    # CHECK 4: NET SETTLEMENT CALCULATION
    # --------------------------------------------------------

    expected_net = round(
        gross_amount
        - expected_fee
        - expected_tax,
        2
    )

    if round(actual_net, 2) != expected_net:

        net_difference = round(
            actual_net - expected_net,
            2
        )

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "SETTLEMENT_NET_MISMATCH",
            "confidence": 95,
            "reason": (
                f"Expected settlement net "
                f"₹{expected_net:.2f}, "
                f"but settlement reports "
                f"₹{actual_net:.2f}. "
                f"Variance: ₹{net_difference:.2f}."
            )
        })

        continue


    # --------------------------------------------------------
    # CHECK 5: FIND BANK TRANSACTION BY UTR
    # --------------------------------------------------------

    bank_matches = bank[
        bank["utr"].astype(str) == utr
    ]

    # --------------------------------------------------------
    # DUPLICATE UTR
    # --------------------------------------------------------

    if len(bank_matches) > 1:

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "status": "EXCEPTION",
            "exception_type": "DUPLICATE_BANK_TRANSACTION",
            "confidence": 100,
            "reason": (
                f"Multiple bank transactions found "
                f"for UTR {utr}."
            )
        })

        continue

    # --------------------------------------------------------
    # EXACT UTR FOUND
    # --------------------------------------------------------

    if len(bank_matches) == 1:

        bank_transaction = bank_matches.iloc[0]
        bank_id = bank_transaction["bank_id"]
        bank_credit = float(bank_transaction["credit"])
        bank_date = pd.to_datetime(bank_transaction["date"])
        settlement_date = pd.to_datetime(settlement["date"])
        date_difference = abs((bank_date - settlement_date).days)

        if round(bank_credit, 2) != round(expected_net, 2):

            difference = round(bank_credit - expected_net, 2)

            results.append({
                "payment_id": payment_id,
                "settlement_id": settlement_id,
                "bank_id": bank_id,
                "status": "EXCEPTION",
                "exception_type": "BANK_AMOUNT_MISMATCH",
                "confidence": 100,
                "reason": (
                    f"Expected bank credit ₹{expected_net:.2f}, "
                    f"but received ₹{bank_credit:.2f}. "
                    f"Difference: ₹{difference:.2f}."
                )
            })

            continue

        if date_difference > 3:

            results.append({
                "payment_id": payment_id,
                "settlement_id": settlement_id,
                "bank_id": bank_id,
                "status": "EXCEPTION",
                "exception_type": "DATE_MISMATCH",
                "confidence": 100,
                "reason": (
                    f"Settlement date: {settlement_date.date()}, "
                    f"Bank date: {bank_date.date()}."
                )
            })

            continue

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "bank_id": bank_id,
            "status": "MATCHED",
            "exception_type": "",
            "confidence": 100,
            "reason": (
                "Payment, settlement and bank "
                "transaction match successfully."
            )
        })

        continue

    # ========================================================
    # NO EXACT UTR FOUND
    #
    # INVESTIGATE FOR UTR MISMATCH
    # ========================================================

    settlement_date = pd.to_datetime(settlement["date"])

    exact_date_candidates = bank[
        (
            bank["credit"].round(2)
            == round(expected_net, 2)
        )
        &
        (
            pd.to_datetime(bank["date"])
            == settlement_date
        )
    ].copy()

    if len(exact_date_candidates) == 1:

        candidate = exact_date_candidates.iloc[0]
        candidate_utr = str(candidate["utr"])
        candidate_bank_id = candidate["bank_id"]

        results.append({
            "payment_id": payment_id,
            "settlement_id": settlement_id,
            "bank_id": candidate_bank_id,
            "status": "EXCEPTION",
            "exception_type": "UTR_MISMATCH",
            "confidence": 99,
            "reason": (
                f"Expected UTR {utr}, but bank "
                f"transaction {candidate_bank_id} "
                f"has UTR {candidate_utr}. "
                f"Amount ₹{expected_net:.2f} and "
                f"date {settlement_date.date()} match."
            )
        })

        continue

    bank_candidates = bank[
        bank["credit"].round(2)
        == round(expected_net, 2)
    ].copy()

    if not bank_candidates.empty:

        bank_candidates["date_difference"] = (
            pd.to_datetime(bank_candidates["date"])
            - settlement_date
        ).abs().dt.days

        nearby_candidates = bank_candidates[
            bank_candidates["date_difference"] <= 3
        ]

        if len(nearby_candidates) == 1:

            candidate = nearby_candidates.iloc[0]
            candidate_utr = str(candidate["utr"])
            candidate_bank_id = candidate["bank_id"]

            results.append({
                "payment_id": payment_id,
                "settlement_id": settlement_id,
                "bank_id": candidate_bank_id,
                "status": "EXCEPTION",
                "exception_type": "UTR_MISMATCH",
                "confidence": 95,
                "reason": (
                    f"Expected UTR {utr}, but bank "
                    f"transaction {candidate_bank_id} "
                    f"contains UTR {candidate_utr}. "
                    f"Amount and date are consistent."
                )
            })

            continue

    results.append({
        "payment_id": payment_id,
        "settlement_id": settlement_id,
        "status": "EXCEPTION",
        "exception_type": "MISSING_BANK_TRANSACTION",
        "confidence": 90,
        "reason": (
            f"No bank transaction found for "
            f"UTR {utr}, and no sufficiently strong "
            f"candidate was found using amount and date."
        )
    })


# ============================================================
# 5. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 6. CALCULATE BASIC METRICS
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
    matched_records
    / total_records
    * 100
)


processing_time = (
    time.time() - start_time
)

throughput = (
    total_records
    / processing_time
    if processing_time > 0
    else 0
)


# ============================================================
# 7. DISPLAY REPORT
# ============================================================

print("\n")
print("=" * 65)
print("                 FINGUARD AI")
print("          SMART RECONCILIATION ENGINE")
print("=" * 65)

print(
    f"\nTotal records processed : {total_records}"
)

print(
    f"Matched records         : {matched_records}"
)

print(
    f"Exceptions              : {exception_records}"
)

print(
    f"\nMatch rate              : "
    f"{match_rate:.2f}%"
)

print(
    f"Processing time         : "
    f"{processing_time:.4f} seconds"
)

print(
    f"Throughput              : "
    f"{throughput:.2f} records/second"
)


# ============================================================
# 8. SHOW EXCEPTIONS
# ============================================================

exceptions = results_df[
    results_df["status"] == "EXCEPTION"
]


print("\n")
print("=" * 65)
print("                    EXCEPTIONS")
print("=" * 65)


for _, row in exceptions.iterrows():

    print(
        f"\n{row['payment_id']} "
        f"→ {row['exception_type']}"
    )

    print(
        f"Confidence: "
        f"{row['confidence']}%"
    )

    print(
        f"Reason: {row['reason']}"
    )


# ============================================================
# 9. SAVE RESULTS
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
print("=" * 65)

print(
    f"Results saved to: {results_path}"
)

print("=" * 65)