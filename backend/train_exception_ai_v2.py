import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# ============================================================
# FINGUARD AI
# EXCEPTION AI MODEL TRAINING V2
# OPERATIONAL EVIDENCE FEATURES
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


DATASET_PATH = os.path.join(
    DATA_DIR,
    "exception_ai_v2_dataset.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fin_guard_exception_classifier_v2.pkl"
)

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "exception_ai_v2_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# DISPLAY HEADER
# ============================================================

print("\n")
print("=" * 70)

print(
    "                 FINGUARD AI"
)

print(
    "       EXCEPTION AI MODEL TRAINING V2"
)

print(
    "          OPERATIONAL EVIDENCE"
)

print("=" * 70)


print(
    f"\nTotal records : {len(df)}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "label"


if TARGET not in df.columns:

    raise ValueError(
        f"\nTarget column '{TARGET}' "
        "was not found in the dataset."
    )


# ============================================================
# V2 OPERATIONAL FEATURES
#
# Raw UTR is deliberately excluded.
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
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    print("\nMissing required features:")

    for feature in missing_features:

        print(
            f"- {feature}"
        )

    raise ValueError(
        "\nThe V2 dataset does not contain "
        "all required operational features."
    )


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].astype(str)


X = X.replace(
    [float("inf"), float("-inf")],
    0
)

X = X.fillna(
    0
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print(
    "Class distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# IMPORTANT:
#
# The V2 dataset already contains the benchmark records.
# We use a deterministic split here so results are reproducible.
#
# Stratification is used so every exception class appears
# in train/test where possible.
# ============================================================

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\n")
print(
    "Dataset split:"
)

print(
    f"Training   : {len(X_train)}"
)

print(
    f"Test       : {len(X_test)}"
)


print("\n")
print(
    "Test class distribution:"
)

print(
    y_test.value_counts()
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),


    "Random Forest":

        RandomForestClassifier(

            n_estimators=500,

            max_depth=None,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1

        ),


    "Gradient Boosting":

        GradientBoostingClassifier(

            n_estimators=200,

            learning_rate=0.05,

            max_depth=3,

            random_state=42

        )

}


# ============================================================
# TRAINING
# ============================================================

model_results = []


trained_models = {}


for model_name, model in models.items():

    print("\n")
    print(
        f"Training {model_name}..."
    )


    model.fit(
        X_train,
        y_train
    )


    trained_models[
        model_name
    ] = model


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    model_results.append({

        "model":
            model_name,

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

    })


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")

    print(
        model_name
    )

    print(
        f"Test Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Test Macro Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Test Macro F1       : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Test Weighted F1    : "
        f"{weighted_f1 * 100:.2f}%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    model_results
)


# Select based primarily on Macro F1
# because all exception classes matter.

best_row = results_df.loc[
    results_df[
        "macro_f1"
    ].idxmax()
]


best_model_name = best_row[
    "model"
]

best_model = trained_models[
    best_model_name
]


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = best_model.predict(
    X_test
)


# ============================================================
# FINAL REPORT
# ============================================================

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

final_macro_f1 = f1_score(
    y_test,
    final_predictions,
    average="macro",
    zero_division=0
)

final_weighted_f1 = f1_score(
    y_test,
    final_predictions,
    average="weighted",
    zero_division=0
)


print("\n")
print("=" * 70)

print(
    "                 MODEL COMPARISON"
)

print("=" * 70)


for _, row in results_df.iterrows():

    print("\n")

    print(
        row["model"]
    )

    print(
        f"Accuracy       : "
        f"{row['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{row['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall   : "
        f"{row['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1       : "
        f"{row['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Weighted F1    : "
        f"{row['weighted_f1'] * 100:.2f}%"
    )


# ============================================================
# FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)

print(
    "             FINAL AI EXCEPTION MODEL V2"
)

print("=" * 70)


print(
    f"\nSelected model : "
    f"{best_model_name}"
)

print(
    f"Test Accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Macro Precision: "
    f"{final_precision * 100:.2f}%"
)

print(
    f"Macro Recall   : "
    f"{final_recall * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{final_macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1    : "
    f"{final_weighted_f1 * 100:.2f}%"
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
        final_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)


cm = confusion_matrix(
    y_test,
    final_predictions,
    labels=labels
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual: {x}"
        for x in labels
    ],
    columns=[
        f"Predicted: {x}"
        for x in labels
    ]
)


print(
    "Confusion Matrix:"
)

print(
    cm_df
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n")
print(
    "Feature Importance:"
)


if best_model_name == "Random Forest":

    importances = (
        best_model
        .feature_importances_
    )

elif best_model_name == "Gradient Boosting":

    importances = (
        best_model
        .feature_importances_
    )

else:

    # Logistic Regression is a pipeline.
    # Use absolute coefficient magnitude.

    classifier = best_model.named_steps[
        "model"
    ]

    importances = np.abs(
        classifier.coef_
    ).mean(
        axis=0
    )


importance_df = pd.DataFrame({

    "feature":
        FEATURES,

    "importance":
        importances

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_package = {

    "model":
        best_model,

    "features":
        FEATURES,

    "version":
        "V2",

    "description":
        "FinGuard AI operational-evidence exception classifier",

    "classes":
        list(
            best_model.classes_
        )

}


with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model_package,
        file
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)

print(
    "                 FINAL MODEL"
)

print("=" * 70)


print(
    f"\nModel saved to:"
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


print("\n")
print(
    "AI features used:"
)

for feature in FEATURES:

    print(
        f"- {feature}"
    )


print("\n")
print(
    "Important:"
)

print(
    "- Raw UTR is NOT used as an AI feature."
)

print(
    "- Operational UTR evidence is used."
)

print(
    "- Duplicate evidence is explicitly represented."
)

print(
    "- Date evidence is explicitly represented."
)

print(
    "- Macro F1 is used for model selection."
)

print("=" * 70)