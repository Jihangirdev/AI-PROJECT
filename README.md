# Skill-Based Job Role Predictor

This is our INF375 Final Project.  
The project predicts a possible IT job role based on a person's skills, experience, education, certifications, and soft skills.

The main idea is simple: if a candidate enters skills like `Python, SQL, pandas, machine learning`, the system should predict which job role fits best, for example `Machine Learning Engineer` or `Data Analyst`.

## Live Project

Website: https://inf375final.streamlit.app/

Video demo: 

## Project Goal

The goal of this project is to build a supervised machine learning system that can classify a candidate into one of several IT job roles.

The model predicts one of these 8 roles:

- Backend Developer
- Frontend Developer
- Mobile Developer
- Data Analyst
- Machine Learning Engineer
- DevOps Engineer
- QA Engineer
- Cybersecurity Analyst

This project uses classical machine learning algorithms, not deep learning, because our dataset is small and the task fits better with models covered in the course.

## Why This Project Is Interesting

Many students and beginner developers know some technologies, but they are not always sure which job role matches their skills.  
For example, a person may know Python, SQL, Docker, and REST API. These skills can match more than one role.

Our project tries to solve this problem by predicting the most likely role and also showing the top possible roles with confidence scores.

## Dataset

The final dataset contains 398 candidate profiles and 9 columns.

The dataset is a hybrid dataset:

- 289 custom rows created manually and synthetically
- 109 real job-description rows collected using the Adzuna API

Some manually labelled examples were based on real job listings from hh.kz.  
The synthetic rows were added to make the dataset more balanced across all job roles.

Main columns in the dataset:

- `skills`
- `qualification`
- `experience_years`
- `education_level`
- `language`
- `certification`
- `soft_skills`
- `job_role`
- `source`

The target column is:

```text
job_role
```

## Important Data Leakage Fix

At first, we used almost all columns as model input. However, we found that the `qualification` column often contained the job role name directly, for example:

```text
Senior Backend Developer
QA Engineer
Frontend Developer
```

This created data leakage, because the model could read the answer from the input.

To fix this, we removed these columns from training:

```text
qualification
source
language
```

The final model uses only safe features:

```text
skills
experience_years
education_level
certification
soft_skills
```

After this fix, the score became lower but more honest and realistic.

## Machine Learning Methods Used

We tested several supervised machine learning models:

- Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Extra Trees

The text columns were converted using TF-IDF vectorization.  
Categorical columns were encoded using OneHotEncoder.  
The numeric column `experience_years` was scaled using StandardScaler.

The full preprocessing and model are wrapped inside a scikit-learn Pipeline, so the same steps are used during training and prediction.

## Final Model

The best final model was:

```text
Tuned Random Forest Classifier
```

It was selected using GridSearchCV with macro F1 score.

Final result:

```text
Accuracy: 0.870
Macro F1 Score: 0.870
Cross-validation Macro F1: 0.786
```

We used macro F1 because the dataset is not perfectly balanced.  
Accuracy alone can be misleading when some classes have more examples than others.

## Evaluation

The project includes several evaluation outputs:

- model comparison chart
- confusion matrix
- learning curve
- class distribution chart
- feature importance chart
- error analysis CSV

The confusion matrix showed that some roles are easier to predict than others.  
For example, Cybersecurity Analyst was predicted very well because cybersecurity skills are more specific.  
Mobile Developer was harder because mobile skills often overlap with Frontend skills, especially when React Native or JavaScript appears.

## How to Run the Project


### Install requirements

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python main.py
```

### 5. Run the Streamlit web app

```bash
streamlit run streamlit_app.py
```

### 6. Run the desktop app

```bash
python desktop_app.py
```

## Example Input

```text
Skills: Python, pandas, scikit-learn, SQL, machine learning
Experience: 2 years
Education: Bachelor
Certification: None
Soft skills: analytical thinking, problem solving
```

Possible output:

```text
Predicted role: Machine Learning Engineer
Confidence: 78%
Top matches:
1. Machine Learning Engineer
2. Data Analyst
3. Backend Developer
```

## Main Features

- Predicts IT job role from candidate profile
- Shows confidence score
- Shows top possible job roles
- Compares several ML algorithms
- Uses TF-IDF for skill text
- Includes confusion matrix and learning curve
- Includes Streamlit web interface
- Includes Tkinter desktop interface
- Saves trained model for reuse

## Team Members

- Bayazit Madina
- Zhetesuly Jihangir
- Kuanyshova Fariza

## Course

INF375 Final Project  
Skill-Based Job Role Predictor
