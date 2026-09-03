import os
import pickle
import time
import numpy as np
import pandas as pd


# ============================================================
# FINGUARD AI
# AI FINANCE CONTROLLER V7
#
# INTELLIGENT REVIEW & AI DISAGREEMENT HANDLING
#
# V6 deterministic engine remains authoritative.
#
# V7 adds:
#   - AI agreement detection
#   - disagreement reason
#   - review priority
#   - review category
#   - confidence-aware decisions
#   - human review queue
#   - exception automation metrics
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PAYMENTS_PATH = os.path.join(
    DATA_DIR,
    "payments.csv"
)

SETTLEMENTS_PATH = os.path.join(
    DATA_DIR,
    "settlements.csv"
)

BANK_PATH = os.path.join(
    DATA_DIR,
    "bank_transactions.csv"
)

GROUND_TRUTH_PATH = os.path.join(
    DATA_DIR,
    "ground_truth.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fin_guard_exception_classifier_v2.pkl"
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "ai_controller_v7_results.csv"
)

REVIEW_QUEUE_PATH = os.path.join(
    DATA_DIR,
    "ai_human_review_queue.csv"
)


# ============================================================
# CONFIDENCE POLICY
# ============================================================

AUTO_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.60


# ============================================================

# ============================================================
# REUSABLE V7 CONTROLLER
# ============================================================

def run_controller(
    payments_df,
    settlements_df,
    bank_df,
    ground_truth_df=None,
    output_path=None,
    review_queue_path=None,
    model_path=None,
    verbose=True,
):
    """
    Run FinGuard AI V7 using supplied pandas DataFrames.

    ground_truth_df is optional:
      - Demo/benchmark mode: provide ground truth to calculate accuracy.
      - Live/upload mode: omit ground truth; accuracy is reported as N/A.

    Returns:
        results_df, review_queue_df, summary
    """
    # HEADER
    # ============================================================

    print("\n")
    print("=" * 70)

    print(
        "                 FINGUARD AI"
    )

    print(
        "          AI FINANCE CONTROLLER V7"
    )

    print(
        "     INTELLIGENT REVIEW & AI DISAGREEMENT"
    )

    print("=" * 70)


    # ============================================================
    # LOAD DATA
    # ============================================================

    payments = payments_df.copy()
    settlements = settlements_df.copy()
    bank = bank_df.copy()

    if ground_truth_df is None:
        ground_truth = pd.DataFrame(
            columns=["payment_id", "expected_status", "expected_exception"]
        )
        benchmark_available = False
    else:
        ground_truth = ground_truth_df.copy()
        benchmark_available = True

    effective_model_path = model_path or MODEL_PATH
    effective_output_path = output_path
    effective_review_path = review_queue_path

    # NORMALIZE NUMERIC DATA
    # ============================================================

    payments["amount"] = pd.to_numeric(
        payments["amount"],
        errors="coerce"
    )

    settlements["gross_amount"] = pd.to_numeric(
        settlements["gross_amount"],
        errors="coerce"
    )

    settlements["fee"] = pd.to_numeric(
        settlements["fee"],
        errors="coerce"
    )

    settlements["tax"] = pd.to_numeric(
        settlements["tax"],
        errors="coerce"
    )

    settlements["net_amount"] = pd.to_numeric(
        settlements["net_amount"],
        errors="coerce"
    )

    bank["credit"] = pd.to_numeric(
        bank["credit"],
        errors="coerce"
    )


    # ============================================================
    # NORMALIZE DATES
    # ============================================================

    payments["date"] = pd.to_datetime(
        payments["date"],
        errors="coerce"
    )

    settlements["date"] = pd.to_datetime(
        settlements["date"],
        errors="coerce"
    )

    bank["date"] = pd.to_datetime(
        bank["date"],
        errors="coerce"
    )


    # ============================================================
    # GROUND TRUTH
    #
    # Used ONLY for benchmark evaluation.
    # Never used as an AI input.
    # ============================================================

    ground_truth_map = {}

    for _, row in ground_truth.iterrows():

        payment_id = row["payment_id"]

        status = str(
            row["expected_status"]
        ).strip().upper()

        exception = str(
            row["expected_exception"]
        ).strip().upper()

        if status == "EXCEPTION":

            ground_truth_map[
                payment_id
            ] = exception

        else:

            ground_truth_map[
                payment_id
            ] = "NO_EXCEPTION"


    # ============================================================
    # SETTLEMENT LOOKUP
    # ============================================================

    settlement_lookup = (
        settlements
        .drop_duplicates(
            "payment_id"
        )
        .set_index(
            "payment_id"
        )
    )


    # ============================================================
    # LOAD AI MODEL
    # ============================================================

    if not os.path.exists(
        effective_model_path
    ):

        raise FileNotFoundError(
            "\nAI model not found:\n"
            + effective_model_path
        )


    with open(
        MODEL_PATH,
        "rb"
    ) as f:

        package = pickle.load(
            f
        )


    if isinstance(
        package,
        dict
    ):

        model = package["model"]

        AI_FEATURES = package.get(
            "features",
            []
        )

    else:

        model = package

        AI_FEATURES = []


    if not AI_FEATURES:

        AI_FEATURES = [

            "payment_amount",
            "settlement_exists",
            "gross_amount",
            "fee",
            "tax",
            "net_amount",
            "expected_fee",
            "expected_tax",
            "expected_net",
            "fee_difference",
            "tax_difference",
            "net_difference",
            "bank_transaction_exists",
            "bank_amount_difference",
            "bank_transaction_count",
            "date_difference",
            "closest_bank_date_difference",
            "date_within_tolerance",
            "utr_present",
            "utr_match_evidence",
            "matching_utr_count",
            "duplicate_signature_count",
            "duplicate_evidence"

        ]


    # ============================================================
    # HELPER
    # ============================================================

    def safe_float(value):

        try:

            value = float(value)

            if np.isnan(value):

                return 0.0

            return value

        except Exception:

            return 0.0


    # ============================================================
    # DETERMINE CONFIDENCE BAND
    # ============================================================

    def confidence_band(
        confidence
    ):

        if confidence >= AUTO_THRESHOLD:

            return "HIGH"

        if confidence >= REVIEW_THRESHOLD:

            return "MEDIUM"

        return "LOW"


    # ============================================================
    # DETERMINE DISAGREEMENT REASON
    # ============================================================

    def get_disagreement_reason(
        deterministic_exception,
        ai_prediction,
        confidence
    ):

        if (
            deterministic_exception
            ==
            ai_prediction
        ):

            return "AI_AGREES_WITH_ENGINE"

        if (
            deterministic_exception
            ==
            "NO_EXCEPTION"
            and
            ai_prediction
            !=
            "NO_EXCEPTION"
        ):

            return (
                "AI_DETECTED_EXCEPTION_NOT_CONFIRMED"
            )

        if (
            deterministic_exception
            !=
            "NO_EXCEPTION"
            and
            ai_prediction
            ==
            "NO_EXCEPTION"
        ):

            return (
                "AI_FAILED_TO_DETECT_EXCEPTION"
            )

        return (
            "AI_EXCEPTION_TYPE_DISAGREEMENT"
        )


    # ============================================================
    # DETERMINE REVIEW PRIORITY
    # ============================================================

    def get_review_priority(
        deterministic_exception,
        ai_prediction,
        confidence
    ):

        # Deterministic evidence is authoritative.
        # Any disagreement gets priority.

        if (
            deterministic_exception
            !=
            ai_prediction
        ):

            if confidence < 0.60:

                return "CRITICAL"

            return "HIGH"


        if (
            deterministic_exception
            !=
            "NO_EXCEPTION"
        ):

            if confidence >= 0.90:

                return "MEDIUM"

            return "HIGH"


        if confidence < 0.60:

            return "HIGH"

        if confidence < 0.90:

            return "MEDIUM"

        return "LOW"


    # ============================================================
    # DETERMINE REVIEW CATEGORY
    # ============================================================

    def get_review_category(
        deterministic_exception,
        ai_prediction
    ):

        if (
            deterministic_exception
            ==
            ai_prediction
        ):

            if (
                deterministic_exception
                ==
                "NO_EXCEPTION"
            ):

                return "NORMAL_RECONCILIATION"

            return "CONFIRMED_EXCEPTION"


        if (
            deterministic_exception
            ==
            "NO_EXCEPTION"
        ):

            return "AI_FALSE_POSITIVE_RISK"


        if (
            ai_prediction
            ==
            "NO_EXCEPTION"
        ):

            return "AI_FALSE_NEGATIVE_RISK"


        return "EXCEPTION_CLASS_DISAGREEMENT"


    # ============================================================
    # DETERMINISTIC ENGINE
    #
    # THIS IS THE SAME CORE LOGIC THAT PRODUCED 100% ACCURACY
    # IN V6.
    # ============================================================

    def deterministic_engine(
        payment_id
    ):

        payment = payments[
            payments["payment_id"]
            ==
            payment_id
        ].iloc[0]


        payment_amount = safe_float(
            payment["amount"]
        )


        # --------------------------------------------------------
        # SETTLEMENT
        # --------------------------------------------------------

        if payment_id in settlement_lookup.index:

            settlement = settlement_lookup.loc[
                payment_id
            ]

            settlement_exists = True

            settlement_date = settlement[
                "date"
            ]

            gross_amount = safe_float(
                settlement["gross_amount"]
            )

            fee = safe_float(
                settlement["fee"]
            )

            tax = safe_float(
                settlement["tax"]
            )

            net_amount = safe_float(
                settlement["net_amount"]
            )

            settlement_utr = str(
                settlement["utr"]
            ).strip()

        else:

            settlement_exists = False

            settlement_date = pd.NaT

            gross_amount = payment_amount

            fee = 0.0
            tax = 0.0
            net_amount = 0.0

            settlement_utr = ""


        # --------------------------------------------------------
        # EXPECTED FINANCIAL VALUES
        # --------------------------------------------------------

        expected_fee = round(
            gross_amount * 0.02,
            2
        )

        expected_tax = round(
            expected_fee * 0.18,
            2
        )

        expected_net = round(
            gross_amount
            -
            expected_fee
            -
            expected_tax,
            2
        )


        fee_difference = abs(
            fee
            -
            expected_fee
        )

        tax_difference = abs(
            tax
            -
            expected_tax
        )

        net_difference = abs(
            net_amount
            -
            expected_net
        )


        # --------------------------------------------------------
        # UTR
        # --------------------------------------------------------

        utr_present = int(
            settlement_utr != ""
        )


        if (
            settlement_exists
            and
            settlement_utr
        ):

            matching_bank_rows = bank[
                bank["utr"]
                .astype(str)
                .str.strip()
                ==
                settlement_utr
            ].copy()

        else:

            matching_bank_rows = pd.DataFrame()


        matching_utr_count = len(
            matching_bank_rows
        )


        utr_match_evidence = int(
            matching_utr_count > 0
        )


        # --------------------------------------------------------
        # DUPLICATE
        #
        # Same UTR occurring at least twice.
        # --------------------------------------------------------

        duplicate_evidence = int(
            matching_utr_count >= 2
        )

        duplicate_signature_count = (
            matching_utr_count
        )


        # --------------------------------------------------------
        # BANK AMOUNT
        # --------------------------------------------------------

        if matching_utr_count > 0:

            bank_amount_difference = float(
                (
                    matching_bank_rows["credit"]
                    -
                    net_amount
                ).abs().min()
            )

        else:

            bank_amount_difference = 0.0


        # --------------------------------------------------------
        # BANK DATE
        # --------------------------------------------------------

        if (
            matching_utr_count > 0
            and
            not pd.isna(
                settlement_date
            )
        ):

            date_differences = (
                matching_bank_rows["date"]
                -
                settlement_date
            ).abs().dt.days

            date_difference = int(
                date_differences.min()
            )

        else:

            date_difference = 0


        closest_bank_date_difference = (
            date_difference
        )


        date_within_tolerance = int(
            date_difference <= 2
        )


        # --------------------------------------------------------
        # UTR MISMATCH
        # --------------------------------------------------------

        if (
            settlement_exists
            and
            matching_utr_count == 0
        ):

            wrong_utr_candidates = bank[
                (
                    (
                        bank["credit"]
                        -
                        net_amount
                    ).abs()
                    <= 0.01
                )
                &
                (
                    (
                        bank["date"]
                        -
                        settlement_date
                    ).abs().dt.days
                    <= 1
                )
            ]

        else:

            wrong_utr_candidates = (
                pd.DataFrame()
            )


        utr_mismatch_evidence = int(
            len(
                wrong_utr_candidates
            )
            > 0
        )


        # --------------------------------------------------------
        # AUTHORITATIVE EXCEPTION
        # --------------------------------------------------------

        if not settlement_exists:

            exception = (
                "MISSING_SETTLEMENT"
            )

        elif duplicate_evidence:

            exception = (
                "DUPLICATE_BANK_TRANSACTION"
            )

        elif (
            matching_utr_count == 0
            and
            utr_mismatch_evidence
        ):

            exception = (
                "UTR_MISMATCH"
            )

        elif (
            matching_utr_count > 0
            and
            date_difference > 2
        ):

            exception = (
                "DATE_MISMATCH"
            )

        elif (
            matching_utr_count > 0
            and
            fee_difference > 0.01
        ):

            exception = (
                "FEE_MISMATCH"
            )

        elif (
            matching_utr_count > 0
            and
            bank_amount_difference > 1.0
        ):

            exception = (
                "BANK_AMOUNT_MISMATCH"
            )

        else:

            exception = (
                "NO_EXCEPTION"
            )


        # --------------------------------------------------------
        # AI FEATURES
        # --------------------------------------------------------

        feature_values = {

            "payment_amount":
                payment_amount,

            "settlement_exists":
                int(settlement_exists),

            "gross_amount":
                gross_amount,

            "fee":
                fee,

            "tax":
                tax,

            "net_amount":
                net_amount,

            "expected_fee":
                expected_fee,

            "expected_tax":
                expected_tax,

            "expected_net":
                expected_net,

            "fee_difference":
                fee_difference,

            "tax_difference":
                tax_difference,

            "net_difference":
                net_difference,

            "bank_transaction_exists":
                int(matching_utr_count > 0),

            "bank_amount_difference":
                bank_amount_difference,

            "bank_transaction_count":
                matching_utr_count,

            "date_difference":
                date_difference,

            "closest_bank_date_difference":
                closest_bank_date_difference,

            "date_within_tolerance":
                date_within_tolerance,

            "utr_present":
                utr_present,

            "utr_match_evidence":
                utr_match_evidence,

            "matching_utr_count":
                matching_utr_count,

            "duplicate_signature_count":
                duplicate_signature_count,

            "duplicate_evidence":
                duplicate_evidence

        }


        ai_row = {}

        for feature in AI_FEATURES:

            ai_row[
                feature
            ] = feature_values.get(
                feature,
                0.0
            )


        return (
            exception,
            feature_values,
            ai_row
        )


    # ============================================================
    # PROCESS ALL TRANSACTIONS
    # ============================================================

    results = []

    start_time = time.perf_counter()


    for _, payment in payments.iterrows():

        payment_id = payment[
            "payment_id"
        ]


        # --------------------------------------------------------
        # DETERMINISTIC ENGINE
        # --------------------------------------------------------

        (
            deterministic_exception,
            feature_values,
            ai_row

        ) = deterministic_engine(
            payment_id
        )


        # --------------------------------------------------------
        # AI
        # --------------------------------------------------------

        X = pd.DataFrame(
            [ai_row]
        )

        X = X.replace(
            [np.inf, -np.inf],
            0
        )

        X = X.fillna(
            0
        )


        probabilities = model.predict_proba(
            X
        )[0]


        classes = list(
            model.classes_
        )


        best_index = int(
            np.argmax(
                probabilities
            )
        )


        ai_prediction = str(
            classes[
                best_index
            ]
        )


        ai_confidence = float(
            probabilities[
                best_index
            ]
        )


        # --------------------------------------------------------
        # SECOND BEST
        # --------------------------------------------------------

        sorted_indices = np.argsort(
            probabilities
        )[::-1]


        if len(
            sorted_indices
        ) > 1:

            second_index = int(
                sorted_indices[1]
            )

            second_prediction = str(
                classes[
                    second_index
                ]
            )

            second_confidence = float(
                probabilities[
                    second_index
                ]
            )

        else:

            second_prediction = ""

            second_confidence = 0.0


        # --------------------------------------------------------
        # AI AGREEMENT
        # --------------------------------------------------------

        ai_agrees = int(
            deterministic_exception
            ==
            ai_prediction
        )


        disagreement_reason = (
            get_disagreement_reason(
                deterministic_exception,
                ai_prediction,
                ai_confidence
            )
        )


        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        band = confidence_band(
            ai_confidence
        )


        # --------------------------------------------------------
        # REVIEW PRIORITY
        # --------------------------------------------------------

        priority = get_review_priority(
            deterministic_exception,
            ai_prediction,
            ai_confidence
        )


        category = get_review_category(
            deterministic_exception,
            ai_prediction
        )


        # --------------------------------------------------------
        # FINAL CONTROLLER DECISION
        #
        # DETERMINISTIC ENGINE REMAINS AUTHORITATIVE.
        # --------------------------------------------------------

        if (
            deterministic_exception
            ==
            "NO_EXCEPTION"
        ):

            if (
                ai_agrees
                and
                ai_confidence
                >= AUTO_THRESHOLD
            ):

                controller_decision = (
                    "AUTO-CLEARED"
                )

            elif (
                ai_agrees
                and
                ai_confidence
                >= REVIEW_THRESHOLD
            ):

                controller_decision = (
                    "REVIEW RECOMMENDED"
                )

            else:

                controller_decision = (
                    "MANDATORY REVIEW"
                )

        else:

            if (
                ai_agrees
                and
                ai_confidence
                >= AUTO_THRESHOLD
            ):

                controller_decision = (
                    "AUTO-CLASSIFIED EXCEPTION"
                )

            elif (
                ai_agrees
                and
                ai_confidence
                >= REVIEW_THRESHOLD
            ):

                controller_decision = (
                    "REVIEW RECOMMENDED"
                )

            else:

                controller_decision = (
                    "MANDATORY REVIEW"
                )


        # --------------------------------------------------------
        # HUMAN REVIEW
        # --------------------------------------------------------

        human_review_required = int(
            controller_decision
            ==
            "MANDATORY REVIEW"
        )


        automated = int(
            controller_decision
            in [
                "AUTO-CLEARED",
                "AUTO-CLASSIFIED EXCEPTION"
            ]
        )


        # --------------------------------------------------------
        # GROUND TRUTH
        # --------------------------------------------------------

        actual_exception = (
            ground_truth_map.get(
                payment_id,
                "UNKNOWN"
            )
        )


        deterministic_correct = int(
            deterministic_exception
            ==
            actual_exception
        )


        ai_correct = int(
            ai_prediction
            ==
            actual_exception
        )


        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        results.append({

            "payment_id":
                payment_id,

            "deterministic_exception":
                deterministic_exception,

            "ai_prediction":
                ai_prediction,

            "ai_confidence":
                round(
                    ai_confidence,
                    4
                ),

            "confidence_percent":
                round(
                    ai_confidence * 100,
                    2
                ),

            "confidence_band":
                band,

            "second_prediction":
                second_prediction,

            "second_confidence":
                round(
                    second_confidence,
                    4
                ),

            "ai_agrees_with_engine":
                ai_agrees,

            "disagreement_reason":
                disagreement_reason,

            "review_category":
                category,

            "review_priority":
                priority,

            "controller_decision":
                controller_decision,

            "automated":
                automated,

            "human_review_required":
                human_review_required,

            "actual_exception":
                actual_exception,

            "deterministic_correct":
                deterministic_correct,

            "ai_correct":
                ai_correct,

            "settlement_exists":
                feature_values[
                    "settlement_exists"
                ],

            "bank_transaction_exists":
                feature_values[
                    "bank_transaction_exists"
                ],

            "bank_transaction_count":
                feature_values[
                    "bank_transaction_count"
                ],

            "bank_amount_difference":
                round(
                    feature_values[
                        "bank_amount_difference"
                    ],
                    2
                ),

            "date_difference":
                feature_values[
                    "date_difference"
                ],

            "date_within_tolerance":
                feature_values[
                    "date_within_tolerance"
                ],

            "utr_present":
                feature_values[
                    "utr_present"
                ],

            "utr_match_evidence":
                feature_values[
                    "utr_match_evidence"
                ],

            "matching_utr_count":
                feature_values[
                    "matching_utr_count"
                ],

            "duplicate_evidence":
                feature_values[
                    "duplicate_evidence"
                ],

            "fee_difference":
                round(
                    feature_values[
                        "fee_difference"
                    ],
                    2
                ),

            "tax_difference":
                round(
                    feature_values[
                        "tax_difference"
                    ],
                    2
                ),

            "net_difference":
                round(
                    feature_values[
                        "net_difference"
                    ],
                    2
                )

        })


    # ============================================================
    # DATAFRAME
    # ============================================================

    df = pd.DataFrame(
        results
    )


    total = len(
        df
    )


    # ============================================================
    # ACCURACY
    # ============================================================

    deterministic_correct = int(
        df[
            "deterministic_correct"
        ].sum()
    )

    ai_correct = int(
        df[
            "ai_correct"
        ].sum()
    )


    deterministic_accuracy = (
        deterministic_correct / total * 100
        if total > 0 and benchmark_available
        else None
    )


    ai_accuracy = (
        ai_correct / total * 100
        if total > 0 and benchmark_available
        else None
    )


    # ============================================================
    # AGREEMENT
    # ============================================================

    agreement_count = int(
        df[
            "ai_agrees_with_engine"
        ].sum()
    )


    disagreement_count = (
        total -
        agreement_count
    )


    agreement_rate = (
        agreement_count
        /
        total
        *
        100
    )


    # ============================================================
    # AUTOMATION
    # ============================================================

    automated_count = int(
        df[
            "automated"
        ].sum()
    )

    human_review_count = int(
        df[
            "human_review_required"
        ].sum()
    )


    automation_rate = (
        automated_count
        /
        total
        *
        100
    )


    human_review_rate = (
        human_review_count
        /
        total
        *
        100
    )


    # ============================================================
    # FINANCE SUMMARY
    # ============================================================

    matched = int(
        (
            df[
                "deterministic_exception"
            ]
            ==
            "NO_EXCEPTION"
        ).sum()
    )

    exceptions = (
        total -
        matched
    )

    match_rate = (
        matched /
        total *
        100
    )


    # ============================================================
    # EXCEPTION AUTOMATION
    # ============================================================

    if benchmark_available:
        exception_rows = df[
            df[
                "actual_exception"
            ]
            !=
            "NO_EXCEPTION"
        ]
    else:
        exception_rows = pd.DataFrame()


    correct_exception_auto = int(
        (
            (
                exception_rows[
                    "deterministic_exception"
                ]
                ==
                exception_rows[
                    "actual_exception"
                ]
            )
            &
            (
                exception_rows[
                    "automated"
                ]
                ==
                1
            )
        ).sum()
    )


    if len(exception_rows) > 0:

        exception_automation_rate = (
            correct_exception_auto
            /
            len(exception_rows)
            *
            100
        )

    else:

        exception_automation_rate = 0.0


    # ============================================================
    # CONFIDENCE COUNTS
    # ============================================================

    high = int(
        (
            df[
                "confidence_band"
            ]
            ==
            "HIGH"
        ).sum()
    )

    medium = int(
        (
            df[
                "confidence_band"
            ]
            ==
            "MEDIUM"
        ).sum()
    )

    low = int(
        (
            df[
                "confidence_band"
            ]
            ==
            "LOW"
        ).sum()
    )


    # ============================================================
    # REVIEW PRIORITY
    # ============================================================

    priority_breakdown = (
        df[
            "review_priority"
        ]
        .value_counts()
    )


    category_breakdown = (
        df[
            "review_category"
        ]
        .value_counts()
    )


    disagreement_breakdown = (
        df[
            "disagreement_reason"
        ]
        .value_counts()
    )


    decision_breakdown = (
        df[
            "controller_decision"
        ]
        .value_counts()
    )


    # ============================================================
    # ACTUAL EXCEPTIONS
    # ============================================================

    actual_breakdown = (
        df[
            "actual_exception"
        ]
        .value_counts()
    )


    # ============================================================
    # AI PREDICTIONS
    # ============================================================

    ai_breakdown = (
        df[
            "ai_prediction"
        ]
        .value_counts()
    )


    # ============================================================
    # HUMAN REVIEW QUEUE
    # ============================================================

    review_queue = df[
        df[
            "human_review_required"
        ]
        ==
        1
    ].copy()


    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3
    }


    review_queue[
        "_priority_order"
    ] = review_queue[
        "review_priority"
    ].map(
        priority_order
    ).fillna(99)


    review_queue = (
        review_queue
        .sort_values(
            [
                "_priority_order",
                "ai_confidence"
            ],
            ascending=[
                True,
                True
            ]
        )
        .drop(
            columns=[
                "_priority_order"
            ]
        )
    )


    if effective_review_path:
        review_queue.to_csv(
            effective_review_path,
            index=False
        )


    # ============================================================
    # SAVE MAIN OUTPUT
    # ============================================================

    if effective_output_path:
        df.to_csv(
            effective_output_path,
            index=False
        )


    # ============================================================
    # THROUGHPUT
    # ============================================================

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    throughput = (
        total /
        elapsed
        if elapsed > 0
        else 0
    )


    # ============================================================
    # REPORT
    # ============================================================

    print("\n")
    print("=" * 70)

    print(
        "                 FINANCE-OPS SUMMARY"
    )

    print("-" * 70)

    print(
        f"Transactions processed : "
        f"{total}"
    )

    print(
        f"Matched                : "
        f"{matched}"
    )

    print(
        f"Exceptions             : "
        f"{exceptions}"
    )

    print(
        f"Match rate             : "
        f"{match_rate:.2f}%"
    )


    print("\n")
    print("=" * 70)

    print(
        "             DETERMINISTIC ENGINE"
    )

    print("-" * 70)

    print(
        f"Evidence accuracy      : "
        f"{deterministic_accuracy:.2f}%"
    )

    print(
        f"Correct evidence       : "
        f"{deterministic_correct}/{total}"
    )


    print("\n")
    print("=" * 70)

    print(
        "                    AI TRIAGE"
    )

    print("-" * 70)

    print(
        f"AI prediction accuracy : "
        f"{ai_accuracy:.2f}%"
    )

    print(
        f"AI correct predictions : "
        f"{ai_correct}/{total}"
    )

    print(
        f"AI/Engine agreement    : "
        f"{agreement_rate:.2f}%"
    )

    print(
        f"AI/Engine disagreements: "
        f"{disagreement_count}"
    )


    print("\n")
    print("AI confidence bands:")

    print(
        f"High   (>=90%)        : "
        f"{high}"
    )

    print(
        f"Medium (60-90%)       : "
        f"{medium}"
    )

    print(
        f"Low    (<60%)        : "
        f"{low}"
    )


    print("\n")
    print("=" * 70)

    print(
        "             DISAGREEMENT ANALYSIS"
    )

    print("-" * 70)

    for name, count in (
        disagreement_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")
    print("=" * 70)

    print(
        "               REVIEW PRIORITY"
    )

    print("-" * 70)

    for name, count in (
        priority_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")
    print("=" * 70)

    print(
        "                REVIEW CATEGORY"
    )

    print("-" * 70)

    for name, count in (
        category_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")
    print("=" * 70)

    print(
        "               CONTROLLER DECISIONS"
    )

    print("-" * 70)

    for name, count in (
        decision_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")

    print(
        f"Automated records      : "
        f"{automated_count}"
    )

    print(
        f"Human review records   : "
        f"{human_review_count}"
    )

    print(
        f"Automation rate        : "
        f"{automation_rate:.2f}%"
    )

    print(
        f"Human review rate      : "
        f"{human_review_rate:.2f}%"
    )


    print("\n")
    print("=" * 70)

    print(
        "             EXCEPTION AUTOMATION"
    )

    print("-" * 70)

    print(
        f"Correctly auto-classified : "
        f"{correct_exception_auto}"
    )

    print(
        f"Exception automation rate : "
        f"{exception_automation_rate:.2f}%"
    )


    print("\n")
    print("=" * 70)

    print(
        "              ACTUAL BENCHMARK"
    )

    print("-" * 70)

    for name, count in (
        actual_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")
    print("=" * 70)

    print(
        "                 AI PREDICTIONS"
    )

    print("-" * 70)

    for name, count in (
        ai_breakdown.items()
    ):

        print(
            f"{name:<40}: {count}"
        )


    print("\n")
    print("=" * 70)

    print(
        "              HUMAN REVIEW QUEUE"
    )

    print("-" * 70)

    print(
        f"Records requiring review : "
        f"{len(review_queue)}"
    )

    print(
        "Queue saved to:"
    )

    print(
        effective_review_path
    )


    print("\n")
    print("=" * 70)

    print(
        "                    THROUGHPUT"
    )

    print("-" * 70)

    print(
        f"Processing time        : "
        f"{elapsed:.4f} seconds"
    )

    print(
        f"Throughput             : "
        f"{throughput:.2f} records/sec"
    )


    print("\n")
    print("=" * 70)

    print(
        "                       OUTPUT"
    )

    print("-" * 70)

    print(
        "Results saved to:"
    )

    print(
        effective_output_path
    )


    print("\n")
    print("=" * 70)

    print(
        "                 CONFIDENCE POLICY"
    )

    print("-" * 70)

    print(
        ">= 90%  : AUTO"
    )

    print(
        "60-90%  : REVIEW RECOMMENDED"
    )

    print(
        "< 60%   : MANDATORY REVIEW"
    )

    print("=" * 70)

    # Structured API result for dashboard / API integrations.
    summary = {
        "transactions_processed": total,
        "matched": matched,
        "exceptions": exceptions,
        "match_rate": match_rate,
        "benchmark_available": benchmark_available,
        "deterministic_accuracy": deterministic_accuracy,
        "ai_accuracy": ai_accuracy,
        "agreement_rate": agreement_rate,
        "disagreement_count": disagreement_count,
        "automated_count": automated_count,
        "human_review_count": human_review_count,
        "automation_rate": automation_rate,
        "human_review_rate": human_review_rate,
        "exception_automation_rate": exception_automation_rate,
        "high_confidence": high,
        "medium_confidence": medium,
        "low_confidence": low,
        "throughput_records_per_sec": throughput,
    }

    return df, review_queue, summary


__all__ = ["run_controller"]