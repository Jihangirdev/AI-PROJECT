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


def show_project_info():
    with st.expander("About this project"):
        st.markdown(
            """
            ### Skill-Based Job Role Predictor

            This project predicts a suitable IT job role from a candidate profile.

            The model uses:

            - skills
            - experience years
            - education level
            - certification
            - soft skills

            The model does **not** use `qualification`, `source`, or `language` for training.
            This prevents target leakage because qualification can contain job titles.

            Models used:

            - Dummy Classifier baseline
            - Logistic Regression
            - KNN
            - Random Forest
            - Extra Trees

            The final model is selected by macro F1-score.
            """
        )


def main():
    st.set_page_config(
        page_title="Job Role Finder",
        page_icon="💼",
        layout="wide",
    )

    model = get_saved_or_train_model()

    st.title("Job Role Finder")
    st.write(
        "This is a simple machine learning tool that suggests an IT role based on skills."
    )

    show_project_info()

    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("Candidate profile")

        with st.form("candidate_form"):
            skills = st.text_area(
                "Skills",
                value="Python, FastAPI, REST API, PostgreSQL, Docker, SQL",
                height=120,
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
        st.subheader("Prediction output")

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

            st.success(f"Suggested role: {role}")
            st.metric("Confidence", f"{confidence}%")

            st.write("Top possible roles:")

            top_probs = probs.head(5).copy()
            top_probs["probability"] = (top_probs["probability"] * 100).round(2)

            st.dataframe(
                top_probs,
                width="stretch",
                hide_index=True,
            )

            st.bar_chart(
                top_probs.set_index("job_role")["probability"]
            )

            with st.expander("Show full probability table"):
                full_table = probs.copy()
                full_table["probability"] = (full_table["probability"] * 100).round(2)
                st.dataframe(
                    full_table,
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("Fill the form and click check role to see the result.")

            st.write("Example inputs:")
            st.markdown(
                """
                - Java, Spring Boot, SQL, Docker
                - React, TypeScript, HTML, CSS
                - Python, Pandas, Excel, Power BI
                """
            )


if __name__ == "__main__":
    main()