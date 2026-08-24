import os
import time

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
payments = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
settlements = pd.read_csv(os.path.join(DATA_DIR, "settlements.csv"))
bank = pd.read_csv(os.path.join(DATA_DIR, "bank_transactions.csv"))
start_time = time.time()
results = []


def add_result(payment_id, status, exception_type="", confidence=0, reason="",
               settlement_id="", bank_id="", evidence="",
               recommended_action=""):
    results.append({
        "payment_id": payment_id,
        "settlement_id": settlement_id,
        "bank_id": bank_id,
        "status": status,
        "exception_type": exception_type,
        "confidence": confidence,
        "reason": reason,
        "evidence": evidence,
        "recommended_action": recommended_action,
    })


for _, payment in payments.iterrows():
    payment_id = payment["payment_id"]
    payment_amount = float(payment["amount"])
    settlement_matches = settlements[settlements["payment_id"] == payment_id]

    if settlement_matches.empty:
        add_result(
            payment_id, "EXCEPTION", "MISSING_SETTLEMENT", 100,
            "Payment exists but no settlement was found.",
            evidence="Payment record exists, but no matching settlement record was found.",
            recommended_action="Investigate why the payment has not appeared in settlement records.",
        )
        continue

    settlement = settlement_matches.iloc[0]
    settlement_id = settlement["settlement_id"]
    utr = str(settlement["utr"])
    gross_amount = float(settlement["gross_amount"])
    actual_fee = float(settlement["fee"])
    actual_tax = float(settlement["tax"])
    actual_net = float(settlement["net_amount"])

    if round(payment_amount, 2) != round(gross_amount, 2):
        add_result(
            payment_id, "EXCEPTION", "GROSS_AMOUNT_MISMATCH", 100,
            f"Payment amount INR {payment_amount:.2f} does not match settlement gross amount INR {gross_amount:.2f}.",
            settlement_id,
            evidence=f"Payment amount: INR {payment_amount:.2f}; Settlement gross amount: INR {gross_amount:.2f}.",
            recommended_action="Investigate the payment and settlement amount discrepancy.",
        )
        continue

    expected_fee = round(gross_amount * 0.02, 2)
    if round(actual_fee, 2) != expected_fee:
        difference = round(actual_fee - expected_fee, 2)
        add_result(
            payment_id, "EXCEPTION", "FEE_MISMATCH", 100,
            f"Expected fee INR {expected_fee:.2f}, but settlement reports INR {actual_fee:.2f}. Fee variance: INR {difference:.2f}.",
            settlement_id,
            evidence=f"Expected fee: INR {expected_fee:.2f}; Actual fee: INR {actual_fee:.2f}; Variance: INR {difference:.2f}.",
            recommended_action="Review the settlement fee calculation and verify the applicable fee rule.",
        )
        continue

    expected_tax = round(expected_fee * 0.18, 2)
    if round(actual_tax, 2) != expected_tax:
        difference = round(actual_tax - expected_tax, 2)
        add_result(
            payment_id, "EXCEPTION", "TAX_MISMATCH", 100,
            f"Expected tax INR {expected_tax:.2f}, but settlement reports INR {actual_tax:.2f}. Tax variance: INR {difference:.2f}.",
            settlement_id,
            evidence=f"Expected tax: INR {expected_tax:.2f}; Actual tax: INR {actual_tax:.2f}; Variance: INR {difference:.2f}.",
            recommended_action="Review the tax calculation and verify the applicable tax rule.",
        )
        continue

    expected_net = round(gross_amount - expected_fee - expected_tax, 2)
    if round(actual_net, 2) != expected_net:
        difference = round(actual_net - expected_net, 2)
        add_result(
            payment_id, "EXCEPTION", "SETTLEMENT_NET_MISMATCH", 95,
            f"Expected settlement net INR {expected_net:.2f}, but settlement reports INR {actual_net:.2f}. Variance: INR {difference:.2f}.",
            settlement_id,
            evidence=f"Expected net: INR {expected_net:.2f}; Actual net: INR {actual_net:.2f}; Variance: INR {difference:.2f}.",
            recommended_action="Review gross amount, fee and tax components to identify the source of the net settlement variance.",
        )
        continue

    settlement_date = pd.to_datetime(settlement["date"])
    bank_matches = bank[bank["utr"].astype(str) == utr]

    if len(bank_matches) > 1:
        add_result(
            payment_id, "EXCEPTION", "DUPLICATE_BANK_TRANSACTION", 100,
            f"Multiple bank transactions found for UTR {utr}.", settlement_id,
            evidence=f"UTR {utr} appears in {len(bank_matches)} bank transactions.",
            recommended_action="Investigate duplicate bank entries and determine which transaction is valid.",
        )
        continue

    if len(bank_matches) == 1:
        bank_transaction = bank_matches.iloc[0]
        bank_id = bank_transaction["bank_id"]
        bank_credit = float(bank_transaction["credit"])
        bank_date = pd.to_datetime(bank_transaction["date"])
        date_difference = abs((bank_date - settlement_date).days)

        if round(bank_credit, 2) != expected_net:
            difference = round(bank_credit - expected_net, 2)
            add_result(
                payment_id, "EXCEPTION", "BANK_AMOUNT_MISMATCH", 100,
                f"Expected bank credit INR {expected_net:.2f}, but received INR {bank_credit:.2f}. Difference: INR {difference:.2f}.",
                settlement_id, bank_id,
                evidence=f"Expected bank credit: INR {expected_net:.2f}; Actual bank credit: INR {bank_credit:.2f}; Variance: INR {difference:.2f}.",
                recommended_action="Verify the bank credit amount against the settlement record.",
            )
            continue

        if date_difference > 3:
            add_result(
                payment_id, "EXCEPTION", "DATE_MISMATCH", 100,
                f"Settlement date: {settlement_date.date()}, Bank date: {bank_date.date()}.",
                settlement_id, bank_id,
                evidence=f"Settlement date: {settlement_date.date()}; Bank date: {bank_date.date()}; Difference: {date_difference} days.",
                recommended_action="Verify the settlement and bank transaction dates and investigate the timing difference.",
            )
            continue

        add_result(
            payment_id, "MATCHED", settlement_id=settlement_id, bank_id=bank_id,
            confidence=100,
            reason="Payment, settlement and bank transaction match successfully.",
            evidence=f"Payment amount INR {payment_amount:.2f} matches gross settlement amount; expected net INR {expected_net:.2f} matches bank credit; UTR matches; dates are within tolerance.",
            recommended_action="No action required.",
        )
        continue

    exact_date_candidates = bank[
        (bank["credit"].round(2) == expected_net)
        & (pd.to_datetime(bank["date"]) == settlement_date)
    ].copy()
    if len(exact_date_candidates) == 1:
        candidate = exact_date_candidates.iloc[0]
        add_result(
            payment_id, "EXCEPTION", "UTR_MISMATCH", 99,
            f"Expected UTR {utr}, but bank transaction {candidate['bank_id']} has UTR {candidate['utr']}. Amount and date match.",
            settlement_id, candidate["bank_id"],
            evidence=f"Expected UTR: {utr}; Bank UTR: {candidate['utr']}; Amount: INR {expected_net:.2f}; Date: {settlement_date.date()}.",
            recommended_action="Verify the bank UTR against the settlement record.",
        )
        continue

    bank_candidates = bank[bank["credit"].round(2) == expected_net].copy()
    if not bank_candidates.empty:
        bank_candidates["date_difference"] = (
            pd.to_datetime(bank_candidates["date"]) - settlement_date
        ).abs().dt.days
        nearby_candidates = bank_candidates[bank_candidates["date_difference"] <= 3]
        if len(nearby_candidates) == 1:
            candidate = nearby_candidates.iloc[0]
            add_result(
                payment_id, "EXCEPTION", "UTR_MISMATCH", 95,
                f"Expected UTR {utr}, but bank transaction {candidate['bank_id']} contains UTR {candidate['utr']}. Amount and date are consistent.",
                settlement_id, candidate["bank_id"],
                evidence=f"Expected UTR: {utr}; Bank UTR: {candidate['utr']}; Expected amount: INR {expected_net:.2f}; Date difference: {int(candidate['date_difference'])} days.",
                recommended_action="Verify the bank UTR against the settlement record.",
            )
            continue

    add_result(
        payment_id, "EXCEPTION", "MISSING_BANK_TRANSACTION", 90,
        f"No bank transaction found for UTR {utr}, and no sufficiently strong candidate was found using amount and date.",
        settlement_id,
        evidence=f"Expected UTR: {utr}; Expected bank amount: INR {expected_net:.2f}; No sufficiently strong bank candidate found.",
        recommended_action="Investigate whether the settlement is missing from the bank statement.",
    )


results_df = pd.DataFrame(results)
total_records = len(results_df)
matched_records = len(results_df[results_df["status"] == "MATCHED"])
exception_records = len(results_df[results_df["status"] == "EXCEPTION"])
match_rate = matched_records / total_records * 100 if total_records else 0
processing_time = time.time() - start_time
throughput = total_records / processing_time if processing_time > 0 else 0
results_path = os.path.join(DATA_DIR, "reconciliation_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 65)
print("                 FINGUARD AI")
print("          SMART RECONCILIATION ENGINE")
print("=" * 65)
print(f"\nTotal records processed : {total_records}")
print(f"Matched records         : {matched_records}")
print(f"Exceptions              : {exception_records}")
print(f"\nMatch rate              : {match_rate:.2f}%")
print(f"Processing time         : {processing_time:.4f} seconds")
print(f"Throughput              : {throughput:.2f} records/second")
print("\n")
print("=" * 65)
print("                    EXCEPTIONS")
print("=" * 65)

exceptions = results_df[
    results_df["status"] == "EXCEPTION"
]

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

    print(
        f"Evidence: {row['evidence']}"
    )

    print(
        f"Recommended action: "
        f"{row['recommended_action']}"
    )
print(f"\nResults saved to        : {results_path}")
