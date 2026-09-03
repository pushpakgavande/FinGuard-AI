import os

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Paths
# NOTE: These were redacted in the shared file. Adjust to match your project layout.
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RESULTS_PATH = os.path.join(DATA_DIR, "ai_controller_v7_results.csv")
REVIEW_PATH = os.path.join(DATA_DIR, "ai_human_review_queue.csv")

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Theme / Styling
# A cohesive, professional fintech palette: deep slate surfaces + emerald accent.
# --------------------------------------------------------------------------------------
PRIMARY = "#10b981"      # emerald accent
PRIMARY_SOFT = "#064e3b"
BG = "#0b0f17"           # app background
SURFACE = "#131a26"      # cards / panels
SURFACE_2 = "#1b2534"    # hover / secondary
BORDER = "#24303f"
TEXT = "#e6edf5"
MUTED = "#8a97a8"

st.markdown(
    f"""
    <style>
    :root {{
        --primary: {PRIMARY};
        --bg: {BG};
        --surface: {SURFACE};
        --border: {BORDER};
        --text: {TEXT};
        --muted: {MUTED};
    }}

    /* ---- Base ---- */
    .stApp {{
        background: radial-gradient(1200px 600px at 100% -10%, #10233b 0%, {BG} 55%);
        color: {TEXT};
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }}
    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }}

    /* ---- Typography ---- */
    h1, h2, h3, h4 {{
        color: {TEXT};
        letter-spacing: -0.01em;
        font-weight: 700;
    }}
    p, span, label, div {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

    /* ---- Brand header ---- */
    .fg-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 22px;
        border: 1px solid {BORDER};
        border-radius: 16px;
        background: linear-gradient(135deg, {SURFACE} 0%, #0e1622 100%);
        margin-bottom: 8px;
    }}
    .fg-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 46px; height: 46px;
        border-radius: 12px;
        background: {PRIMARY_SOFT};
        border: 1px solid {PRIMARY};
        font-size: 24px;
    }}
    .fg-title {{ font-size: 26px; font-weight: 800; margin: 0; line-height: 1.1; }}
    .fg-sub {{ font-size: 13px; color: {MUTED}; margin: 2px 0 0; }}
    .fg-pill {{
        margin-left: auto;
        font-size: 12px;
        font-weight: 600;
        color: {PRIMARY};
        background: {PRIMARY_SOFT};
        border: 1px solid {PRIMARY};
        padding: 6px 12px;
        border-radius: 999px;
        white-space: nowrap;
    }}

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 18px 14px;
        transition: border-color .15s ease, transform .15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {PRIMARY};
        transform: translateY(-2px);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {MUTED} !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 30px !important;
        font-weight: 800 !important;
        color: {TEXT} !important;
    }}

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {{
        background: #0a0e15;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 6px 4px;
        font-weight: 500;
    }}

    /* ---- Panels / expander / dataframe ---- */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        background: {SURFACE};
    }}

    /* ---- Inputs ---- */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: {SURFACE} !important;
        border-color: {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
    }}

    /* ---- Section labels ---- */
    .fg-section {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: {MUTED};
        margin: 6px 0 2px;
    }}

    /* ---- Dividers ---- */
    hr {{ border-color: {BORDER} !important; }}

    /* ---- Alerts spacing ---- */
    [data-testid="stAlert"] {{ border-radius: 12px; }}

    /* Hide default Streamlit chrome for a cleaner app feel */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def brand_header(subtitle: str, pill: str = "Controller V7"):
    st.markdown(
        f"""
        <div class="fg-header">
            <div class="fg-badge">🛡️</div>
            <div>
                <p class="fg-title">FinGuard AI</p>
                <p class="fg-sub">{subtitle}</p>
            </div>
            <div class="fg-pill">{pill}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
if not os.path.exists(RESULTS_PATH):
    brand_header("AI-powered financial reconciliation and exception management")
    st.error("V7 controller results not found. Run `python backend\\ai_controller.py` first.")
    st.stop()

df = pd.read_csv(RESULTS_PATH)
review_df = (
    pd.read_csv(REVIEW_PATH)
    if os.path.exists(REVIEW_PATH)
    else df[df["human_review_required"] == 1].copy()
)

total = len(df)
matched = int((df["deterministic_exception"] == "NO_EXCEPTION").sum())
exceptions = total - matched
match_rate = matched / total * 100 if total else 0
ai_accuracy = df["ai_correct"].mean() * 100 if total else 0
deterministic_accuracy = df["deterministic_correct"].mean() * 100 if total else 0
automation_rate = df["automated"].mean() * 100 if total else 0
review_count = int(df["human_review_required"].sum())
agreement_rate = df["ai_agrees_with_engine"].mean() * 100 if total else 0

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 4px 14px;">
            <span style="font-size:22px;">🛡️</span>
            <span style="font-size:18px;font-weight:800;color:{TEXT};">FinGuard AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Dashboard", "Review Queue", "Transactions", "AI Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="fg-section">System</div>', unsafe_allow_html=True)
    st.caption("Controller: V7")
    st.caption("Deterministic engine + AI triage")
    st.divider()
    st.markdown('<div class="fg-section">Live Snapshot</div>', unsafe_allow_html=True)
    st.caption(f"Transactions processed: {total:,}")
    st.caption(f"Pending review: {review_count:,}")

# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
if page == "Dashboard":
    brand_header("Finance operations overview")
    st.markdown("### Key Performance Indicators")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{total:,}")
    c2.metric("Match Rate", f"{match_rate:.2f}%")
    c3.metric("Exceptions", f"{exceptions:,}")
    c4.metric("Automation Rate", f"{automation_rate:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Deterministic Accuracy", f"{deterministic_accuracy:.2f}%")
    c6.metric("AI Accuracy", f"{ai_accuracy:.2f}%")
    c7.metric("AI / Engine Agreement", f"{agreement_rate:.2f}%")
    c8.metric("Human Review", f"{review_count:,}")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("#### Exception Breakdown")
        counts = df[df["actual_exception"] != "NO_EXCEPTION"]["actual_exception"].value_counts()
        if len(counts):
            st.bar_chart(counts, color=PRIMARY)
        else:
            st.info("No exceptions found.")
    with right:
        st.markdown("#### AI Predictions")
        st.bar_chart(df["ai_prediction"].value_counts(), color=PRIMARY)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Controller Decisions")
        st.bar_chart(df["controller_decision"].value_counts(), color=PRIMARY)
    with right:
        st.markdown("#### AI Confidence")
        st.bar_chart(df["confidence_band"].value_counts(), color=PRIMARY)

elif page == "Review Queue":
    brand_header("Human review queue", pill="Manual Triage")

    if review_df.empty:
        st.success("No transactions currently require review.")
    else:
        st.markdown("### Filters")
        f1, f2, f3 = st.columns(3)
        with f1:
            vals = sorted(review_df["review_priority"].dropna().unique().tolist())
            p = st.multiselect("Priority", vals, default=vals)
        with f2:
            vals = sorted(review_df["review_category"].dropna().unique().tolist())
            cat = st.multiselect("Review Category", vals, default=vals)
        with f3:
            vals = sorted(review_df["deterministic_exception"].dropna().unique().tolist())
            exc = st.multiselect("Exception", vals, default=vals)

        filtered = review_df[
            review_df["review_priority"].isin(p)
            & review_df["review_category"].isin(cat)
            & review_df["deterministic_exception"].isin(exc)
        ].copy()

        st.divider()
        m1, m2 = st.columns([1, 3])
        m1.metric("Records in Queue", len(filtered))

        cols = [
            "payment_id", "review_priority", "review_category", "deterministic_exception",
            "ai_prediction", "ai_confidence", "disagreement_reason",
            "bank_amount_difference", "date_difference", "matching_utr_count",
        ]
        cols = [c for c in cols if c in filtered.columns]
        st.dataframe(filtered[cols], use_container_width=True, hide_index=True)

        if not filtered.empty:
            st.markdown("#### Inspect Record")
            selected = st.selectbox("Select payment", filtered["payment_id"].tolist())
            with st.expander("Full record detail", expanded=True):
                st.json(filtered[filtered["payment_id"] == selected].iloc[0].to_dict())

elif page == "Transactions":
    brand_header("Transaction investigation", pill="Search & Trace")

    search = st.text_input("Search Payment ID", placeholder="Example: PAY0001")
    display = df.copy()
    if search:
        display = display[
            display["payment_id"].astype(str).str.contains(search, case=False, na=False)
        ]

    st.caption(f"Showing {len(display):,} transaction(s)")
    st.dataframe(display, use_container_width=True, hide_index=True)

    if not display.empty:
        st.divider()
        st.markdown("#### Transaction Detail")
        selected = st.selectbox("Select transaction", display["payment_id"].tolist())
        record = display[display["payment_id"] == selected].iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Reconciliation**")
            st.write(f"Deterministic result: `{record['deterministic_exception']}`")
            st.write(f"Actual benchmark: `{record['actual_exception']}`")
        with c2:
            st.markdown("**AI Assessment**")
            st.write(f"Prediction: `{record['ai_prediction']}`")
            st.write(f"Confidence: {float(record['ai_confidence']) * 100:.2f}%")
            st.write(f"Agreement: `{'YES' if record['ai_agrees_with_engine'] else 'NO'}`")

        st.info(record["controller_decision"])

else:  # AI Analysis
    brand_header("AI analysis & model diagnostics", pill="Model Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI Accuracy", f"{ai_accuracy:.2f}%")
    c2.metric("AI / Engine Agreement", f"{agreement_rate:.2f}%")
    c3.metric("AI Disagreements", f"{total - int(df['ai_agrees_with_engine'].sum())}")

    st.divider()
    st.markdown("#### AI / Engine Disagreements")
    disagreements = df[df["ai_agrees_with_engine"] == 0]
    if disagreements.empty:
        st.success("No AI disagreements.")
    else:
        st.dataframe(
            disagreements[
                ["payment_id", "deterministic_exception", "ai_prediction",
                 "ai_confidence", "disagreement_reason", "review_priority"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Disagreement Reasons")
        st.bar_chart(df["disagreement_reason"].value_counts(), color=PRIMARY)
    with right:
        st.markdown("#### Review Priority")
        st.bar_chart(df["review_priority"].value_counts(), color=PRIMARY)

    st.markdown("#### Confidence Distribution")
    conf = df[["payment_id", "ai_confidence"]].copy().set_index("payment_id") * 100
    st.bar_chart(conf, color=PRIMARY)

st.divider()
st.caption("FinGuard AI • V7 Intelligent Reconciliation Controller")
