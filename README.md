# FinGuard AI

## AI-Powered Financial Reconciliation & Exception Management

FinGuard AI is an AI-assisted financial reconciliation system that automatically compares payment, settlement, and bank transaction records to identify mismatches and financial exceptions.

The system combines deterministic reconciliation rules with an AI-based exception classifier and a controller layer to explain discrepancies, assess confidence, recommend actions, and route uncertain cases for human review.

---

## Live Demo

[![Open Live Demo](https://img.shields.io/badge/Live%20Demo-FinGuard%20AI-10b981?style=for-the-badge)](https://finguard-ai-demo.streamlit.app/)

**Try FinGuard AI:**  
https://finguard-ai-demo.streamlit.app/

Explore the AI-powered financial reconciliation dashboard using the included synthetic benchmark dataset.

## Key Results

| Metric | Result |
|---|---:|
| Transactions Evaluated | 500 |
| Successfully Matched | 440 |
| Match Rate | 88% |
| AI Benchmark Accuracy | 96% |
| Deterministic Evidence Accuracy | 100% |
| AI / Engine Agreement | 96% |
| High-Confidence Decisions | 490 |
| Human Review Required | 6% |
| Controller Throughput | ~166 records/sec |

---

## Problem Statement

Financial systems often maintain multiple records for the same transaction:

- Payment records
- Settlement records
- Bank transactions
- Ledger records

Reconciling these records manually can be time-consuming and error-prone, especially when transaction volumes increase.

FinGuard AI automates this process and identifies transactions that require investigation.

---

## How FinGuard AI Works


                Payment Records
                       |
                       v
              Settlement Matching
                       |
                       v
              Financial Validation
                       |
          +------------+------------+
          |            |            |
       Fee Check    Amount Check   Date Check
          |            |            |
          +------------+------------+
                       |
                       v
               Bank Reconciliation
                       |
          +------------+------------+
          |            |            |
       UTR Check   Amount Check   Date Check
          |            |            |
          +------------+------------+
                       |
                       v
             Deterministic Engine
                       |
                       v
             AI Exception Classifier
                       |
                       v
                Controller V7
                       |
          +------------+------------+
          |                         |
     Auto Resolution          Human Review
          |                         |
          +------------+------------+
                       |
                       v
              Dashboard & Reports
 

### Exception Detection

FinGuard AI currently identifies the following exception types:

- `MISSING_SETTLEMENT`
- `DUPLICATE_BANK_TRANSACTION`
- `UTR_MISMATCH`
- `DATE_MISMATCH`
- `FEE_MISMATCH`
- `BANK_AMOUNT_MISMATCH`

The deterministic reconciliation engine provides the primary evidence for each exception.



### AI Exception Classification

The AI layer predicts the most likely exception category for each transaction.

The model uses transaction and reconciliation context while avoiding direct use of the UTR as an AI feature to reduce the risk of target leakage.

The system evaluates:

- AI prediction
- Prediction confidence
- Second-best prediction
- Agreement with the deterministic engine
- Disagreement reason
- Review priority

This allows the AI model to assist reconciliation rather than blindly replacing deterministic financial checks.


### Controller & Human-in-the-Loop

FinGuard AI uses a controller layer to combine deterministic reconciliation evidence with AI predictions.

The controller can:

- Automatically clear transactions when evidence and AI prediction agree with sufficient confidence
- Identify AI/engine disagreements
- Assign confidence bands
- Prioritize exceptions
- Route uncertain cases to human review
- Provide a complete decision record for auditing

This creates a human-in-the-loop workflow rather than treating the AI prediction as the final authority.
 

### Dashboard

FinGuard AI includes a Streamlit dashboard for monitoring reconciliation and reviewing exceptions.

```text
Dashboard
   ├── Overview & Metrics
   ├── Exception Breakdown
   ├── Controller Decisions
   └── AI Confidence

Review Queue
   ├── Priority Filtering
   ├── Exception Filtering
   ├── Transaction Selection
   └── Full Record Inspection

Transactions
   ├── Payment Search
   ├── Reconciliation Details
   └── AI Assessment

AI Analysis
   ├── AI Accuracy
   ├── Engine Agreement
   ├── Disagreements
   └── Confidence Analysis

 
### Demo & Benchmark Data

The repository currently uses synthetic demo/benchmark data so that the complete reconciliation and AI workflow can be demonstrated consistently and reproducibly.
The demo dataset contains intentionally constructed transaction scenarios, including different types of reconciliation errors. These errors are fixed/injected into the synthetic data deliberately to test whether FinGuard AI can correctly detect and classify the corresponding exceptions.
This allows the project to demonstrate the complete workflow without requiring access to sensitive real-world financial data.

Current Benchmark
The benchmark contains 500 synthetic payment transactions.

| Exception Type | Count |
|---|---:|
| `NO_EXCEPTION` | 440 |
| `DATE_MISMATCH` | 10 |
| `FEE_MISMATCH` | 10 |
| `MISSING_SETTLEMENT` | 10 |
| `DUPLICATE_BANK_TRANSACTION` | 10 |
| `UTR_MISMATCH` | 10 |
| `BANK_AMOUNT_MISMATCH` | 10 |

The benchmark ground-truth file is used for evaluation and reproducibility.



### Future Data Ingestion

The current version uses the included demo/benchmark CSV files.
A planned extension is to allow users to provide their own CSV files containing payment, settlement, and bank transaction data.

The future workflow can be extended to:

```text
User CSV Files
      |
      v
Data Validation
      |
      v
Reconciliation Engine
      |
      v
AI Exception Classification
      |
      v
Controller V7
      |
      +---- Automatic Resolution
      |
      +---- Human Review
      |
      v
Dashboard
This would allow FinGuard AI to move from a fixed demonstration dataset toward a reusable reconciliation platform capable of processing organization-specific transaction data.

Current status: The repository demonstrates the complete reconciliation, AI classification, controller, and dashboard workflow using synthetic demo data. Custom CSV ingestion is planned as a future enhancement.


### Project Structure

```text
FinGuard-AI/
│
├── backend/
│   ├── ai_controller.py
│   ├── controller_engine.py
│   ├── evaluate.py
│   ├── evaluate_exception_ai.py
│   ├── reconcile.py
│   └── train_exception_ai_v2.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── payments.csv
│   ├── settlements.csv
│   ├── bank_transactions.csv
│   ├── ground_truth.csv
│   ├── ai_controller_v7_results.csv
│   └── ai_human_review_queue.csv
│
├── models/
│   └── fin_guard_exception_classifier_v2.pkl
│
├── .gitignore
├── README.md
└── requirements.txt

### Installation

Clone the repository:
git clone https://github.com/pushpakgavande/FinGuard-AI.git
cd FinGuard-AI

Create a virtual environment:
python -m venv venv
Windows

Activate the virtual environment:
.\venv\Scripts\Activate.ps1

Install Dependencies:
pip install -r requirements.txt

 Run the Dashboard:

Start the Streamlit application:
python -m streamlit run dashboard/app.py

The dashboard will open in your browser.


### Evaluation

The project includes evaluation scripts for measuring reconciliation and AI performance.

Run the reconciliation evaluation:
python backend/evaluate.py

Run the AI-specific evaluation:
python backend/evaluate_exception_ai.py

### Design Principles

#### Deterministic Evidence First
Financial reconciliation decisions are grounded in deterministic transaction checks.

#### AI as an Assistant
The AI model assists with exception classification rather than replacing the reconciliation engine.

#### Human-in-the-Loop
Low-confidence or conflicting decisions can be routed for manual investigation.

#### Leakage-Aware Evaluation
UTR identifiers are excluded from AI features to reduce the risk of directly exposing transaction identity information to the classifier.

#### Auditability
The controller records predictions, confidence, evidence, disagreements, and review decisions.


### Technology Stack

- **Python** — Core application and backend development
- **Pandas** — Data processing and transaction analysis
- **Scikit-learn** — AI-based exception classification
- **Streamlit** — Interactive web dashboard
- **Joblib** — Model serialization and loading
- **CSV** — Synthetic financial transaction and benchmark datasets


### Future Enhancements

- **Custom CSV Data Ingestion** — Allow users to provide their own transaction datasets
- **Secure File Upload** — Add validated and secure file upload capabilities
- **Database Integration** — Support scalable persistent transaction storage
- **Real-Time Transaction Processing** — Enable continuous reconciliation and exception detection
- **Advanced Anomaly Detection** — Detect more complex and previously unseen transaction patterns
- **Explainable AI** — Provide clear reasoning behind AI-based exception decisions
- **Role-Based Review Workflows** — Support different permissions and responsibilities for reviewers
- **Production-Grade Authentication & Audit Logging** — Strengthen security, access control, and traceability
- **Cloud Deployment** — Enable scalable deployment on cloud infrastructure


### 🎯 Current Scope

The current version focuses on:

- **Financial Transaction Reconciliation** — Comparing payment, settlement, and bank transaction records
- **Deterministic Exception Detection** — Identifying rule-based reconciliation discrepancies
- **AI-Based Exception Classification** — Classifying detected exceptions using machine learning
- **Controller-Based Decisioning** — Combining deterministic results and AI predictions for final decisions
- **Confidence-Based Routing** — Routing transactions based on AI confidence and controller rules
- **Human Review Workflow** — Sending uncertain or conflicting cases for manual review
- **Interactive Dashboard Visualization** — Providing an interface for monitoring and investigating reconciliation results
- **Reproducible Synthetic Benchmark Evaluation** — Evaluating system performance using a controlled synthetic dataset
The included datasets are synthetic demonstration data with intentionally constructed exception scenarios. They are provided to demonstrate and evaluate the system.

Custom user-provided CSV ingestion and processing are planned as a future enhancement.


### Project

FinGuard AI
AI-powered financial reconciliation and exception management system.
Built to demonstrate the integration of deterministic financial validation, machine learning, automated decision control, and human-in-the-loop review workflows.