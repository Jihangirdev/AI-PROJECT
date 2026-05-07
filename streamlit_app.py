import pandas as pd
import streamlit as st

from model import predict_one
from train_utils import get_saved_or_train_model


def make_row(skills, qualification, experience, education, language, certification, soft_skills):
    return pd.DataFrame([{
        "skills": skills,
        "qualification": qualification,
        "experience_years": experience,
        "education_level": education,
        "language": language,
        "certification": certification,
        "soft_skills": soft_skills,
    }])


def add_css():
    st.markdown(
        """
        <style>
        /* ---------- Page ---------- */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 28%),
                #f5f7fb;
            color: #172033;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        header {
            background: transparent !important;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        /* ---------- Header ---------- */
        .hero {
            margin-bottom: 26px;
        }

        .badge {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: #eaf2ff;
            color: #2563eb;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 12px;
            border: 1px solid #dbeafe;
        }

        .app-title {
            font-size: 42px;
            font-weight: 900;
            color: #172033;
            letter-spacing: -1.2px;
            margin: 0;
            line-height: 1.1;
        }

        .app-subtitle {
            margin-top: 12px;
            max-width: 680px;
            font-size: 16px;
            color: #667085;
            line-height: 1.6;
        }

        /* ---------- Cards ---------- */
        .info-card {
            background: #ffffff;
            border: 1px solid #e6eaf2;
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 14px 40px rgba(17, 24, 39, 0.06);
            margin-bottom: 18px;
        }

        .info-title {
            color: #172033;
            font-size: 20px;
            font-weight: 850;
            margin-bottom: 8px;
        }

        .info-text {
            color: #667085;
            font-size: 14px;
            line-height: 1.6;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e6eaf2;
            border-radius: 22px;
            padding: 22px 22px 24px 22px;
            box-shadow: 0 14px 40px rgba(17, 24, 39, 0.06);
        }

        /* ---------- Labels ---------- */
        label,
        .stTextInput label,
        .stTextArea label,
        .stSelectbox label,
        .stNumberInput label {
            color: #263247 !important;
            font-size: 13px !important;
            font-weight: 800 !important;
        }

        /* ---------- Inputs ---------- */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            background: #f3f6fb !important;
            color: #172033 !important;
            border: 1px solid #e4e8f0 !important;
            border-radius: 14px !important;
            font-size: 15px !important;
            padding: 13px 14px !important;
            box-shadow: none !important;
        }

        .stTextArea textarea {
            min-height: 98px !important;
        }

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {
            background: #ffffff !important;
            border: 1px solid #3b82f6 !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12) !important;
        }

        /* ---------- Selectboxes ---------- */
        div[data-baseweb="select"] > div {
            background: #f3f6fb !important;
            border: 1px solid #e4e8f0 !important;
            border-radius: 14px !important;
            color: #172033 !important;
            min-height: 46px !important;
            box-shadow: none !important;
        }

        div[data-baseweb="select"] span {
            color: #172033 !important;
        }

        div[data-baseweb="popover"] {
            border-radius: 14px !important;
        }

        /* ---------- Number buttons ---------- */
        button[data-testid="stNumberInputStepUp"],
        button[data-testid="stNumberInputStepDown"] {
            background: #f3f6fb !important;
            border: none !important;
            color: #172033 !important;
        }

        /* ---------- Button ---------- */
        div.stButton > button {
            width: 100%;
            height: 50px;
            border-radius: 15px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            font-size: 15px;
            font-weight: 850;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.24);
            transition: all 0.16s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            color: white;
            border: none;
        }

        div.stButton > button:active {
            transform: translateY(0px);
        }

        /* ---------- Result ---------- */
        .result-card {
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.96), rgba(30, 64, 175, 0.96));
            border-radius: 24px;
            padding: 28px;
            color: white;
            box-shadow: 0 18px 45px rgba(37, 99, 235, 0.25);
            margin-bottom: 20px;
        }

        .result-small {
            color: #dbeafe;
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 10px;
        }

        .result-role {
            color: #ffffff;
            font-size: 34px;
            font-weight: 900;
            line-height: 1.15;
            margin-bottom: 12px;
        }

        .result-confidence {
            color: #eff6ff;
            font-size: 15px;
        }

        /* ---------- Probability box ---------- */
        .prob-box {
            background: #ffffff;
            border: 1px solid #e6eaf2;
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 14px 40px rgba(17, 24, 39, 0.06);
            margin-top: 16px;
        }

        .prob-title {
            font-size: 17px;
            font-weight: 850;
            color: #172033;
            margin-bottom: 18px;
        }

        .prob-row {
            margin-bottom: 16px;
        }

        .prob-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 7px;
            font-size: 14px;
        }

        .prob-name {
            color: #263247;
            font-weight: 750;
        }

        .prob-value {
            color: #2563eb;
            font-weight: 850;
        }

        .bar-bg {
            width: 100%;
            height: 10px;
            background: #e9eef7;
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563eb, #22c55e);
            border-radius: 999px;
        }

        /* ---------- Empty state ---------- */
        .hint-box {
            background: #ffffff;
            border: 1px dashed #b9c3d4;
            border-radius: 22px;
            padding: 24px;
            color: #536078;
            font-size: 14px;
            line-height: 1.75;
            box-shadow: 0 14px 40px rgba(17, 24, 39, 0.05);
        }

        .hint-title {
            color: #172033;
            font-weight: 850;
            margin-bottom: 10px;
            font-size: 16px;
        }

        /* ---------- Expander / dataframe ---------- */
        .streamlit-expanderHeader {
            color: #263247 !important;
            font-weight: 750 !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e6eaf2;
        }

        /* ---------- Markdown cleanup ---------- */
        h1, h2, h3, h4, p {
            color: #172033;
        }

        @media (max-width: 900px) {
            .app-title {
                font-size: 34px;
            }

            .result-role {
                font-size: 28px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_probs(probs):
    for _, row in probs.head(5).iterrows():
        role = row["job_role"]
        percent = round(row["probability"] * 100, 2)

        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-top">
                    <span class="prob-name">{role}</span>
                    <span class="prob-value">{percent}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{percent}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Job Role Finder",
        page_icon="💼",
        layout="wide",
    )

    add_css()

    model = get_saved_or_train_model()

    st.markdown(
        """
        <div class="hero">
            <div class="badge">INF375 Machine Learning Project</div>
            <div class="app-title">Job Role Finder</div>
            <div class="app-subtitle">
                A simple machine learning tool that suggests an IT role based on skills,
                experience, education, certification and soft skills.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Candidate profile</div>
                <div class="info-text">
                    Enter a candidate profile below. The model mainly uses skills and related background fields for prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("candidate_form"):
            skills = st.text_area(
                "Skills",
                value="Python, FastAPI, REST API, PostgreSQL, Docker, SQL",
                height=110,
            )

            col1, col2 = st.columns(2)

            with col1:
                qualification = st.text_input(
                    "Qualification / project",
                    value="Backend project",
                )

                experience = st.number_input(
                    "Experience years",
                    min_value=0,
                    max_value=20,
                    value=1,
                    step=1,
                )

                education = st.selectbox(
                    "Education",
                    ["Bachelor", "College", "Master", "Bootcamp", "Certification", "Unknown"],
                )

            with col2:
                language = st.selectbox(
                    "Language",
                    ["English", "Russian", "Kazakh", "Turkish", "German"],
                )

                certification = st.selectbox(
                    "Certificate",
                    [
                        "None",
                        "AWS Cloud Practitioner",
                        "Google Data Analytics",
                        "Cisco CCNA",
                        "Oracle Java",
                        "Meta Frontend",
                        "TensorFlow Developer",
                        "ISTQB Foundation",
                        "Kubernetes Basics",
                        "Unknown",
                    ],
                )

                soft_skills = st.text_input(
                    "Soft skills",
                    value="Communication, Problem Solving",
                )

            submitted = st.form_submit_button("Check role")

    with right:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">Prediction output</div>
                <div class="info-text">
                    The system shows the strongest predicted role and other close alternatives.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if submitted:
            row = make_row(
                skills,
                qualification,
                experience,
                education,
                language,
                certification,
                soft_skills,
            )

            role, probs = predict_one(model, row)
            confidence = round(probs.iloc[0]["probability"] * 100, 2)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-small">Suggested role</div>
                    <div class="result-role">{role}</div>
                    <div class="result-confidence">Confidence: <b>{confidence}%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="prob-box">
                    <div class="prob-title">Top possible roles</div>
                """,
                unsafe_allow_html=True,
            )

            show_probs(probs)

            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Show full probability table"):
                table = probs.copy()
                table["probability"] = (table["probability"] * 100).round(2)
                st.dataframe(table, width="stretch", hide_index=True)

        else:
            st.markdown(
                """
                <div class="hint-box">
                    <div class="hint-title">No prediction yet</div>
                    Try one of these examples:
                    <br>• Java, Spring Boot, SQL, Docker
                    <br>• React, TypeScript, HTML, CSS
                    <br>• Python, Pandas, Excel, Power BI
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()