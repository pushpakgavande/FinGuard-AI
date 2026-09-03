import os
import time

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fin_guard_matcher.pkl"
)

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
# LOAD AI MODEL
# ============================================================

try:
    ai_model = joblib.load(MODEL_PATH)
    AI_AVAILABLE = True
except Exception as error:
    ai_model = None
    AI_AVAILABLE = False

    print(
        f"\nWARNING: AI model could not be loaded: {error}"
    )


AI_FEATURES = [
    "amount_difference",
    "date_difference",
    "utr_match",
    "gross_amount_difference",
    "fee_difference",
    "tax_difference",
    "net_amount_difference"
]


# ============================================================
# AI CONFIGURATION
# ============================================================

AI_MIN_CONFIDENCE = 0.85
AI_MIN_MARGIN = 0.10


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()

results = []


# ============================================================
# RESULT HELPER
# ============================================================

def add_result(
    payment_id,
    status,
    exception_type="",
    confidence=0,
    reason="",
    settlement_id="",
    bank_id="",
    evidence="",
    recommended_action="",
    ai_score=None,
    ai_decision=""
):

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
        "ai_score": ai_score,
        "ai_decision": ai_decision,
    })


# ============================================================
# AI CANDIDATE SCORING
# ============================================================

def score_bank_candidates(
    settlement,
    expected_net,
    expected_fee,
    expected_tax,
    candidates
):

    if not AI_AVAILABLE:
        return None

    if candidates.empty:
        return None

    candidate_data = candidates.copy()

    candidate_data["credit"] = pd.to_numeric(
        candidate_data["credit"],
        errors="coerce"
    )

    candidate_data["normalized_date"] = pd.to_datetime(
        candidate_data["date"],
        errors="coerce"
    )

    settlement_date = pd.to_datetime(
        settlement["date"]
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

    gross_amount = float(
        settlement["gross_amount"]
    )

    # --------------------------------------------------------
    # Build the same feature structure used during training.
    # --------------------------------------------------------

    candidate_data["amount_difference"] = (
        candidate_data["credit"]
        - expected_net
    ).abs()

    candidate_data["date_difference"] = (
        candidate_data["normalized_date"]
        - settlement_date
    ).abs().dt.days

    expected_utr = str(
        settlement["utr"]
    )

    candidate_data["utr_match"] = (
        candidate_data["utr"]
        .astype(str)
        == expected_utr
    ).astype(int)

    candidate_data["gross_amount_difference"] = (
        candidate_data["credit"]
        - gross_amount
    ).abs()

    candidate_data["fee_difference"] = (
        abs(actual_fee - expected_fee)
    )

    candidate_data["tax_difference"] = (
        abs(actual_tax - expected_tax)
    )

    candidate_data["net_amount_difference"] = (
        candidate_data["credit"]
        - actual_net
    ).abs()

    # --------------------------------------------------------
    # Remove rows with invalid feature values.
    # --------------------------------------------------------

    feature_data = candidate_data[
        AI_FEATURES
    ].copy()

    feature_data = feature_data.fillna(0)

    # --------------------------------------------------------
    # Predict probability of MATCH.
    # --------------------------------------------------------

    probabilities = ai_model.predict_proba(
        feature_data
    )

    # The positive class is label 1.
    positive_class_index = list(
        ai_model.classes_
    ).index(1)

    candidate_data["ai_score"] = (
        probabilities[
            :,
            positive_class_index
        ]
    )

    # --------------------------------------------------------
    # Rank candidates.
    # --------------------------------------------------------

    candidate_data = candidate_data.sort_values(
        "ai_score",
        ascending=False
    ).reset_index(drop=True)

    return candidate_data


# ============================================================
# RECONCILIATION ENGINE
# ============================================================

for _, payment in payments.iterrows():

    payment_id = payment["payment_id"]

    payment_amount = float(
        payment["amount"]
    )

    settlement_matches = settlements[
        settlements["payment_id"]
        == payment_id
    ]


    # ========================================================
    # MISSING SETTLEMENT
    # ========================================================

    if settlement_matches.empty:

        add_result(
            payment_id,
            "EXCEPTION",
            "MISSING_SETTLEMENT",
            100,
            "Payment exists but no settlement was found.",
            evidence=(
                "Payment record exists, but no matching "
                "settlement record was found."
            ),
            recommended_action=(
                "Investigate why the payment has not "
                "appeared in settlement records."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # SETTLEMENT DATA
    # ========================================================

    settlement = settlement_matches.iloc[0]

    settlement_id = settlement[
        "settlement_id"
    ]

    utr = str(
        settlement["utr"]
    )

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


    # ========================================================
    # GROSS AMOUNT CHECK
    # ========================================================

    if round(payment_amount, 2) != round(
        gross_amount,
        2
    ):

        add_result(
            payment_id,
            "EXCEPTION",
            "GROSS_AMOUNT_MISMATCH",
            100,
            (
                f"Payment amount INR "
                f"{payment_amount:.2f} does not match "
                f"settlement gross amount INR "
                f"{gross_amount:.2f}."
            ),
            settlement_id,
            evidence=(
                f"Payment amount: INR "
                f"{payment_amount:.2f}; "
                f"Settlement gross amount: INR "
                f"{gross_amount:.2f}."
            ),
            recommended_action=(
                "Investigate the payment and settlement "
                "amount discrepancy."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # FEE CHECK
    # ========================================================

    expected_fee = round(
        gross_amount * 0.02,
        2
    )

    if round(actual_fee, 2) != expected_fee:

        difference = round(
            actual_fee - expected_fee,
            2
        )

        add_result(
            payment_id,
            "EXCEPTION",
            "FEE_MISMATCH",
            100,
            (
                f"Expected fee INR "
                f"{expected_fee:.2f}, but settlement "
                f"reports INR {actual_fee:.2f}. "
                f"Fee variance: INR "
                f"{difference:.2f}."
            ),
            settlement_id,
            evidence=(
                f"Expected fee: INR "
                f"{expected_fee:.2f}; "
                f"Actual fee: INR "
                f"{actual_fee:.2f}; "
                f"Variance: INR "
                f"{difference:.2f}."
            ),
            recommended_action=(
                "Review the settlement fee calculation "
                "and verify the applicable fee rule."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # TAX CHECK
    # ========================================================

    expected_tax = round(
        expected_fee * 0.18,
        2
    )

    if round(actual_tax, 2) != expected_tax:

        difference = round(
            actual_tax - expected_tax,
            2
        )

        add_result(
            payment_id,
            "EXCEPTION",
            "TAX_MISMATCH",
            100,
            (
                f"Expected tax INR "
                f"{expected_tax:.2f}, but settlement "
                f"reports INR {actual_tax:.2f}. "
                f"Tax variance: INR "
                f"{difference:.2f}."
            ),
            settlement_id,
            evidence=(
                f"Expected tax: INR "
                f"{expected_tax:.2f}; "
                f"Actual tax: INR "
                f"{actual_tax:.2f}; "
                f"Variance: INR "
                f"{difference:.2f}."
            ),
            recommended_action=(
                "Review the tax calculation and "
                "verify the applicable tax rule."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # NET AMOUNT CHECK
    # ========================================================

    expected_net = round(
        gross_amount
        - expected_fee
        - expected_tax,
        2
    )

    if round(actual_net, 2) != expected_net:

        difference = round(
            actual_net - expected_net,
            2
        )

        add_result(
            payment_id,
            "EXCEPTION",
            "SETTLEMENT_NET_MISMATCH",
            95,
            (
                f"Expected settlement net INR "
                f"{expected_net:.2f}, but settlement "
                f"reports INR {actual_net:.2f}. "
                f"Variance: INR "
                f"{difference:.2f}."
            ),
            settlement_id,
            evidence=(
                f"Expected net: INR "
                f"{expected_net:.2f}; "
                f"Actual net: INR "
                f"{actual_net:.2f}; "
                f"Variance: INR "
                f"{difference:.2f}."
            ),
            recommended_action=(
                "Review gross amount, fee and tax "
                "components to identify the source "
                "of the net settlement variance."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # BANK RECONCILIATION
    # ========================================================

    settlement_date = pd.to_datetime(
        settlement["date"]
    )

    bank_matches = bank[
        bank["utr"].astype(str)
        == utr
    ]


    # ========================================================
    # DUPLICATE BANK TRANSACTION
    # ========================================================

    if len(bank_matches) > 1:

        add_result(
            payment_id,
            "EXCEPTION",
            "DUPLICATE_BANK_TRANSACTION",
            100,
            (
                f"Multiple bank transactions found "
                f"for UTR {utr}."
            ),
            settlement_id,
            evidence=(
                f"UTR {utr} appears in "
                f"{len(bank_matches)} bank transactions."
            ),
            recommended_action=(
                "Investigate duplicate bank entries "
                "and determine which transaction is valid."
            ),
            ai_decision="NOT_USED"
        )

        continue


    # ========================================================
    # EXACT UTR FOUND
    # ========================================================

    if len(bank_matches) == 1:

        bank_transaction = (
            bank_matches.iloc[0]
        )

        bank_id = bank_transaction[
            "bank_id"
        ]

        bank_credit = float(
            bank_transaction["credit"]
        )

        bank_date = pd.to_datetime(
            bank_transaction["date"]
        )

        date_difference = abs(
            (bank_date - settlement_date).days
        )


        # ----------------------------------------------------
        # BANK AMOUNT MISMATCH
        # ----------------------------------------------------

        if round(
            bank_credit,
            2
        ) != expected_net:

            difference = round(
                bank_credit - expected_net,
                2
            )

            add_result(
                payment_id,
                "EXCEPTION",
                "BANK_AMOUNT_MISMATCH",
                100,
                (
                    f"Expected bank credit INR "
                    f"{expected_net:.2f}, but received "
                    f"INR {bank_credit:.2f}. "
                    f"Difference: INR "
                    f"{difference:.2f}."
                ),
                settlement_id,
                bank_id,
                evidence=(
                    f"Expected bank credit: INR "
                    f"{expected_net:.2f}; "
                    f"Actual bank credit: INR "
                    f"{bank_credit:.2f}; "
                    f"Variance: INR "
                    f"{difference:.2f}."
                ),
                recommended_action=(
                    "Verify the bank credit amount "
                    "against the settlement record."
                ),
                ai_decision="NOT_USED"
            )

            continue


        # ----------------------------------------------------
        # DATE MISMATCH
        # ----------------------------------------------------

        if date_difference > 3:

            add_result(
                payment_id,
                "EXCEPTION",
                "DATE_MISMATCH",
                100,
                (
                    f"Settlement date: "
                    f"{settlement_date.date()}, "
                    f"Bank date: "
                    f"{bank_date.date()}."
                ),
                settlement_id,
                bank_id,
                evidence=(
                    f"Settlement date: "
                    f"{settlement_date.date()}; "
                    f"Bank date: "
                    f"{bank_date.date()}; "
                    f"Difference: "
                    f"{date_difference} days."
                ),
                recommended_action=(
                    "Verify the settlement and bank "
                    "transaction dates and investigate "
                    "the timing difference."
                ),
                ai_decision="NOT_USED"
            )

            continue


        # ----------------------------------------------------
        # SUCCESSFUL EXACT MATCH
        # ----------------------------------------------------

        add_result(
            payment_id,
            "MATCHED",
            settlement_id=settlement_id,
            bank_id=bank_id,
            confidence=100,
            reason=(
                "Payment, settlement and bank "
                "transaction match successfully."
            ),
            evidence=(
                f"Payment amount INR "
                f"{payment_amount:.2f} matches gross "
                f"settlement amount; expected net INR "
                f"{expected_net:.2f} matches bank credit; "
                f"UTR matches; dates are within tolerance."
            ),
            recommended_action="No action required.",
            ai_decision="NOT_REQUIRED_EXACT_MATCH"
        )

        continue


    # ========================================================
    # NO EXACT UTR
    #
    # AI CANDIDATE SEARCH
    # ========================================================

    bank_candidates = bank.copy()

    bank_candidates["credit"] = pd.to_numeric(
        bank_candidates["credit"],
        errors="coerce"
    )

    bank_candidates["normalized_date"] = pd.to_datetime(
        bank_candidates["date"],
        errors="coerce"
    )

    bank_candidates = bank_candidates.dropna(
        subset=[
            "credit",
            "normalized_date"
        ]
    ).copy()


    # --------------------------------------------------------
    # First narrow the search to plausible candidates.
    #
    # We use a wider financial window than the old exact
    # amount/date search so that AI can actually rank
    # ambiguous candidates.
    # --------------------------------------------------------

    bank_candidates["date_difference"] = (
        bank_candidates["normalized_date"]
        - settlement_date
    ).abs().dt.days

    bank_candidates["amount_difference"] = (
        bank_candidates["credit"]
        - expected_net
    ).abs()


    # Candidates within a reasonable financial window.
    candidate_pool = bank_candidates[
        (
            bank_candidates["date_difference"]
            <= 7
        )
        &
        (
            bank_candidates["amount_difference"]
            <= max(
                500,
                expected_net * 0.10
            )
        )
    ].copy()


    # --------------------------------------------------------
    # Don't let the expected UTR be considered as a
    # candidate here. It was already checked above.
    # --------------------------------------------------------

    candidate_pool = candidate_pool[
        candidate_pool["utr"].astype(str)
        != utr
    ].copy()


    # ========================================================
    # AI SCORING
    # ========================================================

    ranked_candidates = score_bank_candidates(
        settlement,
        expected_net,
        expected_fee,
        expected_tax,
        candidate_pool
    )


    if (
        ranked_candidates is not None
        and not ranked_candidates.empty
    ):

        best = ranked_candidates.iloc[0]

        best_score = float(
            best["ai_score"]
        )

        best_bank_id = str(
            best["bank_id"]
        )

        best_utr = str(
            best["utr"]
        )

        best_amount = float(
            best["credit"]
        )

        best_date = pd.to_datetime(
            best["normalized_date"]
        )

        best_date_difference = int(
            best["date_difference"]
        )


        if len(ranked_candidates) > 1:

            second_score = float(
                ranked_candidates.iloc[1]["ai_score"]
            )

        else:

            second_score = 0.0


        score_margin = (
            best_score
            - second_score
        )


        # ====================================================
        # HIGH-CONFIDENCE AI CANDIDATE
        # ====================================================

        if (
            best_score >= AI_MIN_CONFIDENCE
            and score_margin >= AI_MIN_MARGIN
        ):

            add_result(
                payment_id,
                "EXCEPTION",
                "UTR_MISMATCH",
                round(
                    best_score * 100,
                    2
                ),
                (
                    f"AI identified bank transaction "
                    f"{best_bank_id} as the strongest "
                    f"candidate with a "
                    f"{best_score * 100:.2f}% match score, "
                    f"but its UTR does not match the "
                    f"settlement UTR."
                ),
                settlement_id,
                best_bank_id,
                evidence=(
                    f"Expected UTR: {utr}; "
                    f"Candidate UTR: {best_utr}; "
                    f"Expected amount: INR "
                    f"{expected_net:.2f}; "
                    f"Candidate amount: INR "
                    f"{best_amount:.2f}; "
                    f"Date difference: "
                    f"{best_date_difference} days; "
                    f"AI score: "
                    f"{best_score * 100:.2f}%; "
                    f"Score margin over second candidate: "
                    f"{score_margin * 100:.2f}%."
                ),
                recommended_action=(
                    "Verify the bank UTR against the "
                    "settlement record. AI identified "
                    "a strong candidate but did not "
                    "override the UTR control."
                ),
                ai_score=round(
                    best_score,
                    4
                ),
                ai_decision="HIGH_CONFIDENCE_UTR_MISMATCH"
            )

            continue


        # ====================================================
        # AMBIGUOUS AI RESULT
        # ====================================================

        add_result(
            payment_id,
            "EXCEPTION",
            "AMBIGUOUS_BANK_CANDIDATE",
            round(
                best_score * 100,
                2
            ),
            (
                f"AI found a strongest bank candidate "
                f"{best_bank_id} with a "
                f"{best_score * 100:.2f}% match score, "
                f"but the candidate did not meet the "
                f"confidence and separation thresholds."
            ),
            settlement_id,
            best_bank_id,
            evidence=(
                f"Best candidate: {best_bank_id}; "
                f"Candidate UTR: {best_utr}; "
                f"Expected UTR: {utr}; "
                f"AI score: "
                f"{best_score * 100:.2f}%; "
                f"Second-best score: "
                f"{second_score * 100:.2f}%; "
                f"Score margin: "
                f"{score_margin * 100:.2f}%."
            ),
            recommended_action=(
                "Review the AI-ranked bank candidates "
                "and verify the correct transaction "
                "before resolving the exception."
            ),
            ai_score=round(
                best_score,
                4
            ),
            ai_decision="AMBIGUOUS_REVIEW"
        )

        continue


    # ========================================================
    # NO AI CANDIDATE
    # ========================================================

    add_result(
        payment_id,
        "EXCEPTION",
        "MISSING_BANK_TRANSACTION",
        90,
        (
            f"No bank transaction found for UTR "
            f"{utr}, and no sufficiently strong "
            f"AI candidate was found."
        ),
        settlement_id,
        evidence=(
            f"Expected UTR: {utr}; "
            f"Expected bank amount: INR "
            f"{expected_net:.2f}; "
            f"No sufficiently strong AI "
            f"candidate found."
        ),
        recommended_action=(
            "Investigate whether the settlement "
            "is missing from the bank statement."
        ),
        ai_decision=(
            "NO_CANDIDATE"
            if AI_AVAILABLE
            else "AI_UNAVAILABLE"
        )
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# CALCULATE METRICS
# ============================================================

total_records = len(
    results_df
)

matched_records = len(
    results_df[
        results_df["status"]
        == "MATCHED"
    ]
)

exception_records = len(
    results_df[
        results_df["status"]
        == "EXCEPTION"
    ]
)

match_rate = (
    matched_records
    / total_records
    * 100
    if total_records
    else 0
)

processing_time = (
    time.time()
    - start_time
)

throughput = (
    total_records
    / processing_time
    if processing_time > 0
    else 0
)


# ============================================================
# AI METRICS
# ============================================================

ai_used = len(
    results_df[
        results_df["ai_decision"].astype(str)
        .str.contains(
            "HIGH_CONFIDENCE|AMBIGUOUS",
            regex=True,
            na=False
        )
    ]
)

high_confidence_ai = len(
    results_df[
        results_df["ai_decision"].astype(str)
        == "HIGH_CONFIDENCE_UTR_MISMATCH"
    ]
)

ambiguous_ai = len(
    results_df[
        results_df["ai_decision"].astype(str)
        == "AMBIGUOUS_REVIEW"
    ]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    DATA_DIR,
    "reconciliation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "                 FINGUARD AI"
)

print(
    "       AI-ASSISTED RECONCILIATION ENGINE"
)

print(
    "=" * 70
)

print(
    f"\nAI model loaded         : "
    f"{'YES' if AI_AVAILABLE else 'NO'}"
)

print(
    f"AI model path           : "
    f"{MODEL_PATH}"
)

print(
    f"\nTotal records processed : "
    f"{total_records}"
)

print(
    f"Matched records         : "
    f"{matched_records}"
)

print(
    f"Exceptions              : "
    f"{exception_records}"
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

print(
    "\n" + "-" * 70
)

print(
    "                    AI SUMMARY"
)

print(
    "-" * 70
)

print(
    f"AI-assisted cases       : "
    f"{ai_used}"
)

print(
    f"High-confidence AI      : "
    f"{high_confidence_ai}"
)

print(
    f"AI escalated/review     : "
    f"{ambiguous_ai}"
)

print(
    "-" * 70
)


# ============================================================
# EXCEPTIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "                    EXCEPTIONS"
)

print(
    "=" * 70
)


exceptions = results_df[
    results_df["status"]
    == "EXCEPTION"
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
        f"Reason: "
        f"{row['reason']}"
    )

    print(
        f"Evidence: "
        f"{row['evidence']}"
    )

    print(
        f"Recommended action: "
        f"{row['recommended_action']}"
    )

    if pd.notna(
        row["ai_score"]
    ):

        print(
            f"AI score: "
            f"{float(row['ai_score']) * 100:.2f}%"
        )

    print(
        f"AI decision: "
        f"{row['ai_decision']}"
    )


print(
    f"\nResults saved to        : "
    f"{results_path}"
)

print(
    "=" * 70
)