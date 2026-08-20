import pandas as pd
import os


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# 2. LOAD DATA
# ============================================================

ground_truth_path = os.path.join(
    DATA_DIR,
    "ground_truth.csv"
)

predictions_path = os.path.join(
    DATA_DIR,
    "reconciliation_results.csv"
)

ground_truth = pd.read_csv(ground_truth_path)
predictions = pd.read_csv(predictions_path)


# ============================================================
# 3. CLEAN COLUMNS
# ============================================================

ground_truth["expected_status"] = (
    ground_truth["expected_status"]
    .astype(str)
    .str.strip()
)

ground_truth["expected_exception"] = (
    ground_truth["expected_exception"]
    .fillna("")
    .astype(str)
    .str.strip()
)

predictions["status"] = (
    predictions["status"]
    .astype(str)
    .str.strip()
)

predictions["exception_type"] = (
    predictions["exception_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. MERGE GROUND TRUTH AND PREDICTIONS
# ============================================================

evaluation = ground_truth.merge(
    predictions,
    on="payment_id",
    how="left"
)


# ============================================================
# 5. MAKE SURE WE HAVE 100 RECORDS
# ============================================================

total_records = int(len(evaluation))


# ============================================================
# 6. STATUS COMPARISON
# ============================================================

evaluation["status_correct"] = (
    evaluation["expected_status"]
    == evaluation["status"]
)


correct_status = int(
    evaluation["status_correct"].sum()
)

incorrect_status = (
    total_records - correct_status
)


accuracy = (
    correct_status / total_records * 100
)


# ============================================================
# 7. EXPECTED EXCEPTIONS
# ============================================================

expected_exceptions = evaluation[
    evaluation["expected_status"] == "EXCEPTION"
]

expected_exception_count = int(
    len(expected_exceptions)
)


# ============================================================
# 8. DETECTED EXCEPTIONS
# ============================================================

detected_exceptions = evaluation[
    evaluation["status"] == "EXCEPTION"
]

detected_exception_count = int(
    len(detected_exceptions)
)


# ============================================================
# 9. CORRECT EXCEPTION CLASSIFICATION
# ============================================================

correct_exception_types = evaluation[
    (evaluation["expected_status"] == "EXCEPTION")
    &
    (evaluation["status"] == "EXCEPTION")
    &
    (
        evaluation["expected_exception"]
        == evaluation["exception_type"]
    )
]

correct_exception_count = int(
    len(correct_exception_types)
)


# ============================================================
# 10. EXCEPTION RECALL
# ============================================================

if expected_exception_count > 0:

    exception_recall = (
        correct_exception_count
        / expected_exception_count
        * 100
    )

else:

    exception_recall = 0


# ============================================================
# 11. MATCH RECORDS
# ============================================================

actual_matches = evaluation[
    evaluation["expected_status"] == "MATCHED"
]

predicted_matches = evaluation[
    evaluation["status"] == "MATCHED"
]

correct_matches = evaluation[
    (evaluation["expected_status"] == "MATCHED")
    &
    (evaluation["status"] == "MATCHED")
]


actual_match_count = int(
    len(actual_matches)
)

predicted_match_count = int(
    len(predicted_matches)
)

correct_match_count = int(
    len(correct_matches)
)


# ============================================================
# 12. PRECISION
# ============================================================

if predicted_match_count > 0:

    precision = (
        correct_match_count
        / predicted_match_count
        * 100
    )

else:

    precision = 0


# ============================================================
# 13. RECALL
# ============================================================

if actual_match_count > 0:

    recall = (
        correct_match_count
        / actual_match_count
        * 100
    )

else:

    recall = 0


# ============================================================
# 14. F1 SCORE
# ============================================================

if precision + recall > 0:

    f1_score = (
        2 * precision * recall
        / (precision + recall)
    )

else:

    f1_score = 0


# ============================================================
# 15. FALSE MATCHES
# ============================================================

false_matches = evaluation[
    (evaluation["expected_status"] == "EXCEPTION")
    &
    (evaluation["status"] == "MATCHED")
]

false_match_count = int(
    len(false_matches)
)


# ============================================================
# 16. MISSED EXCEPTIONS
# ============================================================

missed_exceptions = evaluation[
    (evaluation["expected_status"] == "EXCEPTION")
    &
    (evaluation["status"] != "EXCEPTION")
]

missed_exception_count = int(
    len(missed_exceptions)
)


# ============================================================
# 17. WRONG EXCEPTION CLASSIFICATION
# ============================================================

wrong_classification = evaluation[
    (evaluation["expected_status"] == "EXCEPTION")
    &
    (evaluation["status"] == "EXCEPTION")
    &
    (
        evaluation["expected_exception"]
        != evaluation["exception_type"]
    )
]

wrong_classification_count = int(
    len(wrong_classification)
)


# ============================================================
# 18. DISPLAY REPORT
# ============================================================

print("\n")
print("=" * 65)
print("                 FINGUARD AI")
print("              EVALUATION REPORT")
print("=" * 65)


print("\n--- DATASET ---")

print(
    f"Total records              : {total_records}"
)

print(
    f"Expected matches           : {actual_match_count}"
)

print(
    f"Expected exceptions        : {expected_exception_count}"
)


print("\n--- STATUS ACCURACY ---")

print(
    f"Correct predictions        : {correct_status}"
)

print(
    f"Incorrect predictions      : {incorrect_status}"
)

print(
    f"Overall accuracy           : {accuracy:.2f}%"
)


print("\n--- MATCH PERFORMANCE ---")

print(
    f"Correct matches            : {correct_match_count}"
)

print(
    f"Predicted matches          : {predicted_match_count}"
)

print(
    f"Precision                  : {precision:.2f}%"
)

print(
    f"Recall                     : {recall:.2f}%"
)

print(
    f"F1 Score                   : {f1_score:.2f}%"
)


print("\n--- EXCEPTION PERFORMANCE ---")

print(
    f"Expected exceptions        : {expected_exception_count}"
)

print(
    f"Detected exceptions        : {detected_exception_count}"
)

print(
    f"Correct exception types    : {correct_exception_count}"
)

print(
    f"Exception recall           : {exception_recall:.2f}%"
)

print(
    f"Missed exceptions          : {missed_exception_count}"
)

print(
    f"False matches              : {false_match_count}"
)

print(
    f"Wrong classifications      : {wrong_classification_count}"
)


# ============================================================
# 19. SHOW WRONG CLASSIFICATIONS
# ============================================================

if wrong_classification_count > 0:

    print("\n")
    print("=" * 65)
    print("             WRONG CLASSIFICATIONS")
    print("=" * 65)

    for _, row in wrong_classification.iterrows():

        print(
            f"\n{row['payment_id']}"
        )

        print(
            f"Expected : {row['expected_exception']}"
        )

        print(
            f"Predicted: {row['exception_type']}"
        )


# ============================================================
# 20. SHOW MISSED EXCEPTIONS
# ============================================================

if missed_exception_count > 0:

    print("\n")
    print("=" * 65)
    print("               MISSED EXCEPTIONS")
    print("=" * 65)

    for _, row in missed_exceptions.iterrows():

        print(
            f"\n{row['payment_id']}"
        )

        print(
            f"Expected: {row['expected_exception']}"
        )


# ============================================================
# 21. SAVE EVALUATION
# ============================================================

evaluation_path = os.path.join(
    DATA_DIR,
    "evaluation_results.csv"
)

evaluation.to_csv(
    evaluation_path,
    index=False
)


print("\n")
print("=" * 65)

print(
    "Detailed evaluation saved to:"
)

print(
    evaluation_path
)

print("=" * 65)