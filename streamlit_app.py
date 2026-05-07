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
        .block-container { max-width: 1180px; padding-top: 2.2rem; }
        .app-title { font-size: 38px; font-weight: 800; margin-bottom: 6px; }
        .app-subtitle { font-size: 16px; color: #9ca3af; margin-bottom: 26px; }
        .panel {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 18px;
            padding: 24px;
            background: rgba(30, 41, 59, 0.35);
            margin-bottom: 18px;
        }
        .panel-title { font-size: 20px; font-weight: 750; margin-bottom: 6px; }
        .panel-text { color: #9ca3af; font-size: 14px; }
        .result-card {
            padding: 24px;
            border-radius: 20px;
            background: linear-gradient(135deg, #172554, #111827);
            border: 1px solid rgba(147, 197, 253, 0.25);
            margin-bottom: 20px;
        }
        .result-small {
            color: #bfdbfe;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .result-role { color: white; font-size: 30px; font-weight: 800; margin-bottom: 10px; }
        .result-confidence { color: #d1d5db; font-size: 15px; }
        .prob-row { margin-bottom: 15px; }
        .prob-top { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px; }
        .prob-name { color: #e5e7eb; font-weight: 600; }
        .prob-value { color: #93c5fd; font-weight: 700; }
        .bar-bg {
            width: 100%;
            height: 10px;
            background: rgba(148, 163, 184, 0.25);
            border-radius: 100px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #22c55e);
            border-radius: 100px;
        }
        .hint-box {
            border-radius: 16px;
            padding: 18px;
            background: rgba(15, 23, 42, 0.35);
            border: 1px dashed rgba(148, 163, 184, 0.45);
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.6;
        }
        div.stButton > button {
            width: 100%;
            height: 48px;
            border-radius: 14px;
            font-weight: 700;
            background: #2563eb;
            color: white;
            border: 1px solid rgba(59, 130, 246, 0.8);
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
    st.set_page_config(page_title="Job Role Finder", page_icon="💼", layout="wide")
    add_css()

    model = get_saved_or_train_model()

    st.markdown('<div class="app-title">Job Role Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">A small ML tool that suggests an IT role from skills and background.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Candidate information</div>
                <div class="panel-text">Fill the fields below. Skills are the most important part for prediction.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("candidate_form"):
            skills = st.text_area("Skills", value="Python, FastAPI, REST API, PostgreSQL, Docker, SQL", height=110)

            col1, col2 = st.columns(2)

            with col1:
                qualification = st.text_input("Qualification / project", value="Backend project")
                experience = st.number_input("Experience years", min_value=0, max_value=20, value=1, step=1)
                education = st.selectbox("Education", ["Bachelor", "College", "Master", "Bootcamp", "Certification", "Unknown"])

            with col2:
                language = st.selectbox("Language", ["English", "Russian", "Kazakh", "Turkish", "German"])
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
                soft_skills = st.text_input("Soft skills", value="Communication, Problem Solving")

            submitted = st.form_submit_button("Check role")

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Output</div>
                <div class="panel-text">The result shows the best role and close alternatives.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if submitted:
            row = make_row(skills, qualification, experience, education, language, certification, soft_skills)
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

            st.markdown("**Top possible roles**")
            show_probs(probs)

            with st.expander("Show full probability table"):
                table = probs.copy()
                table["probability"] = (table["probability"] * 100).round(2)
                st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.markdown(
                """
                <div class="hint-box">
                    Result is empty now.<br><br>
                    Try examples:
                    <br>• Java, Spring Boot, SQL, Docker
                    <br>• React, TypeScript, HTML, CSS
                    <br>• Python, Pandas, Excel, Power BI
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
