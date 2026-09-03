import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# FINGUARD AI
# FINAL EXCEPTION AI BENCHMARK
#
# TRAINING:
#   exception_ai_training_augmented.csv
#
# FINAL TEST:
#   exception_ai_dataset.csv
#
# The 500-record benchmark is NEVER used for training.
# ============================================================


SEED = 42


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


TRAINING_PATH = os.path.join(
    DATA_DIR,
    "exception_ai_training_augmented.csv"
)

BENCHMARK_PATH = os.path.join(
    DATA_DIR,
    "exception_ai_dataset.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fin_guard_exception_classifier_final.pkl"
)

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "exception_ai_final_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

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

    "bank_amount_difference"

]


TARGET = "label"


# ============================================================
# LOAD DATA
# ============================================================

training = pd.read_csv(
    TRAINING_PATH
)

benchmark = pd.read_csv(
    BENCHMARK_PATH
)


# ============================================================
# VALIDATE
# ============================================================

missing_training = [
    feature
    for feature in FEATURES
    if feature not in training.columns
]

missing_benchmark = [
    feature
    for feature in FEATURES
    if feature not in benchmark.columns
]


if missing_training:

    raise RuntimeError(
        "Training dataset is missing: "
        + str(missing_training)
    )


if missing_benchmark:

    raise RuntimeError(
        "Benchmark dataset is missing: "
        + str(missing_benchmark)
    )


# ============================================================
# CLEAN
# ============================================================

for dataframe in [
    training,
    benchmark
]:

    dataframe[FEATURES] = (
        dataframe[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    dataframe[TARGET] = (
        dataframe[TARGET]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

X_train = training[
    FEATURES
]

y_train = training[
    TARGET
]


# ============================================================
# PREPARE FINAL BENCHMARK
# ============================================================

X_test = benchmark[
    FEATURES
]

y_test = benchmark[
    TARGET
]


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 72)

print(
    "                 FINGUARD AI"
)

print(
    "          FINAL EXCEPTION AI BENCHMARK"
)

print(
    "      AUGMENTED TRAINING / CLEAN TEST"
)

print("=" * 72)


print(
    f"\nTraining records : "
    f"{len(training)}"
)

print(
    f"Benchmark records: "
    f"{len(benchmark)}"
)


print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts()
)


print(
    "\nFINAL BENCHMARK class distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# TRAIN FINAL MODEL
#
# Random Forest was selected from our previous augmented
# training experiment with Macro F1 = 83.81%.
# ============================================================

print("\n")
print(
    "Training final Random Forest..."
)


model = RandomForestClassifier(

    n_estimators=500,

    max_depth=12,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=SEED,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 72)

print(
    "              FINAL BENCHMARK RESULTS"
)

print("=" * 72)


print(
    f"\nTest Accuracy        : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Macro Precision      : "
    f"{precision * 100:.2f}%"
)


print(
    f"Macro Recall         : "
    f"{recall * 100:.2f}%"
)


print(
    f"Macro F1             : "
    f"{macro_f1 * 100:.2f}%"
)


print(
    f"Weighted F1          : "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print(
    "Classification Report:"
)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    benchmark[
        TARGET
    ].unique()
)


cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print(
    "Confusion Matrix:"
)

print(
    pd.DataFrame(
        cm,

        index=[
            f"Actual: {label}"
            for label in labels
        ],

        columns=[
            f"Predicted: {label}"
            for label in labels
        ]

    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n")
print(
    "Feature Importance:"
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        {
            "model":
                model,

            "features":
                FEATURES,

            "classes":
                list(
                    model.classes_
                )
        },
        file
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {

        "model":
            "Random Forest",

        "training_records":
            len(training),

        "benchmark_records":
            len(benchmark),

        "accuracy":
            accuracy,

        "macro_precision":
            precision,

        "macro_recall":
            recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1

    }
])


results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = benchmark[
    ["payment_id"]
].copy()


prediction_output[
    "actual_exception"
] = y_test.values


prediction_output[
    "predicted_exception"
] = predictions


prediction_output[
    "correct"
] = (
    prediction_output[
        "actual_exception"
    ]
    ==
    prediction_output[
        "predicted_exception"
    ]
)


prediction_path = os.path.join(
    DATA_DIR,
    "exception_ai_final_predictions.csv"
)


prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 72)

print(
    "                 FINAL MODEL"
)

print("=" * 72)


print(
    "\nModel:"
)

print(
    "Random Forest"
)


print(
    "\nTraining data:"
)

print(
    "4,220 augmented records"
)


print(
    "\nFinal benchmark:"
)

print(
    "500 untouched original records"
)


print(
    "\nFinal Macro F1:"
)

print(
    f"{macro_f1 * 100:.2f}%"
)


print(
    "\nModel saved to:"
)

print(
    MODEL_PATH
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)


print(
    "\nPredictions saved to:"
)

print(
    prediction_path
)


print("=" * 72)