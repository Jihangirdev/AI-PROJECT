# Skill-Based Job Role Predictor

This is our INF375 Final Project.

The project predicts a possible IT job role based on a person's skills, experience, education, certifications, and soft skills.

The main idea is simple: if a candidate enters skills like `Python, SQL, pandas, machine learning`, the system predicts which IT role fits best, for example `Machine Learning Engineer` or `Data Analyst`.

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

This project uses classical machine learning algorithms, not deep learning. The dataset is not large enough for deep learning, and the task fits better with models covered in the course.

## Why This Project Is Interesting

Many students and beginner developers know some technologies, but they are not always sure which job role matches their skills.

For example, a person may know Python, SQL, Docker, and REST API. These skills can match more than one role, such as Backend Developer, Data Analyst, DevOps Engineer, or Machine Learning Engineer.

This project tries to solve that problem by predicting the most likely role and also showing the top possible roles with confidence scores.

## Dataset

The final dataset contains **398 candidate profiles and 9 columns**.

The dataset is hybrid:

- 289 custom rows created manually/synthetically
- 109 real job-description rows collected using the Adzuna API

The dataset source distribution is:

```text
current_custom_dataset: 72.6%
adzuna_api: 27.4%
```

The class distribution is:

```text
Backend Developer: 57
Frontend Developer: 57
Machine Learning Engineer: 54
DevOps Engineer: 53
Data Analyst: 49
QA Engineer: 45
Cybersecurity Analyst: 43
Mobile Developer: 40
```

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

At first, almost all columns were used as model input. However, the `qualification` column sometimes contained the job role name directly, for example:

```text
Senior Backend Developer
QA Engineer
Frontend Developer
```

This caused data leakage because the model could indirectly read the correct answer from the input.

To fix this, these columns were removed from training:

```text
qualification
source
language
```

The final model uses only these safe features:

```text
skills
experience_years
education_level
certification
soft_skills
```

This made the score lower than the earlier version, but the result became more honest and realistic.

## Machine Learning Methods Used

We tested several supervised machine learning models:

- Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Extra Trees

Preprocessing steps:

- Text columns are converted using TF-IDF vectorization.
- Categorical columns are encoded using OneHotEncoder.
- The numeric column `experience_years` is scaled using StandardScaler.

The full preprocessing and model are wrapped inside a scikit-learn Pipeline. This means the same preprocessing steps are used during both training and prediction.

## Final Model

The best final model was:

```text
Tuned Random Forest Classifier
```

It was selected using GridSearchCV and macro F1-score.

Best parameters:

```text
max_depth: None
max_features: log2
min_samples_leaf: 1
n_estimators: 350
```

Final test result:

```text
Accuracy: 0.87
Macro F1 Score: 0.87
Wrong predictions: 13 out of 100
```

Cross-validation result for the selected tuned Random Forest:

```text
CV Macro F1: 0.7856
```

We used macro F1 because the dataset is not perfectly balanced. Accuracy alone can be misleading when some classes have more examples than others.

## Evaluation Results

The baseline model was very weak:

```text
Baseline Accuracy: 0.14
Baseline Macro F1: 0.0307
```

This shows that the task cannot be solved by simply predicting the most common class.

Simple model results:

```text
Extra Trees: Accuracy 0.81, Macro F1 0.8119
Random Forest: Accuracy 0.82, Macro F1 0.8113
Logistic Regression: Accuracy 0.69, Macro F1 0.6864
KNN: Accuracy 0.33, Macro F1 0.3222
Baseline: Accuracy 0.14, Macro F1 0.0307
```

Tuned model results:

```text
Tuned Logistic Regression:
Accuracy 0.76
Macro F1 0.7539
CV Macro F1 0.6970

Tuned Random Forest:
Accuracy 0.87
Macro F1 0.8700
CV Macro F1 0.7856

Tuned Extra Trees:
Accuracy 0.81
Macro F1 0.8141
CV Macro F1 0.7786
```

The final selected model was **Tuned Random Forest** because it had the highest cross-validation macro F1 among the tuned models.

## Class-Level Result

Final model classification result:

```text
Backend Developer: F1 = 0.80
Cybersecurity Analyst: F1 = 0.96
Data Analyst: F1 = 0.83
DevOps Engineer: F1 = 0.96
Frontend Developer: F1 = 0.79
Machine Learning Engineer: F1 = 0.96
Mobile Developer: F1 = 0.75
QA Engineer: F1 = 0.91
```

Some roles were easier to predict than others.

Cybersecurity Analyst, DevOps Engineer, Machine Learning Engineer, and QA Engineer had strong results.

Mobile Developer was harder because the dataset has fewer Mobile Developer examples and some mobile skills can overlap with frontend-related skills.

## Output Files

After running `main.py`, the project creates files inside the `outputs/` folder.

Generated files:

```text
job_role_model.pkl
model_results.csv
tuned_model_results.csv
cross_validation_results.csv
selected_model.csv
error_analysis.csv
model_comparison.png
confusion_matrix.png
learning_curve.png
class_distribution.png
top_features.png
```

Explanation:

- `job_role_model.pkl` — saved trained model
- `model_results.csv` — model comparison results
- `tuned_model_results.csv` — tuned model results
- `cross_validation_results.csv` — 5-fold CV results for simple models
- `selected_model.csv` — final selected model information
- `error_analysis.csv` — wrong predictions from the test set
- `model_comparison.png` — comparison of models by macro F1
- `confusion_matrix.png` — correct and wrong predictions by class
- `learning_curve.png` — model performance with different training sizes
- `class_distribution.png` — number of rows per job role
- `top_features.png` — most important features used by the model

## Example Prediction Output

Example input:

```text
Skills: Python, FastAPI, REST API, PostgreSQL, Docker, SQL
Qualification: Backend project
Experience: 1 year
Education: Bachelor
Language: English
Certification: None
Soft skills: Communication, Problem Solving
```

Possible output:

```text
Suggested role: Backend Developer
Confidence: 35.11%

Top possible roles:
1. Backend Developer — 35.11%
2. Data Analyst — 18.00%
3. Machine Learning Engineer — 11.36%
4. DevOps Engineer — 11.29%
5. Mobile Developer — 7.67%
```

The confidence is not extremely high because the model no longer uses `qualification` for training. This is intentional and makes the prediction more realistic.

## How to Run the Project

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Train and evaluate the model

```bash
python main.py
```

This will train the models, compare them, select the best one, save the final model, and generate output files.

### 3. Run the Streamlit web app

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 4. Run the desktop app

```bash
python desktop_app.py
```

## Main Features

- Predicts IT job role from candidate profile
- Shows confidence score
- Shows top possible job roles
- Uses safe features without target leakage
- Compares several ML algorithms
- Uses TF-IDF for skill text
- Uses GridSearchCV for tuning
- Uses macro F1-score for model selection
- Includes confusion matrix and learning curve
- Includes error analysis
- Includes Streamlit web interface
- Includes Tkinter desktop interface
- Saves trained model for reuse

## Project Structure

```text
project/
│
├── data/
│   └── job_dataset.csv
│
├── outputs/
│   └── generated files after running main.py
│
├── main.py
├── model.py
├── train_utils.py
├── visualization.py
├── streamlit_app.py
├── desktop_app.py
├── requirements.txt
└── README.md
```

## Team Members

- Bayazit Madina
- Zhetesuly Jihangir
- Kuanyshova Fariza

## Course

INF375 Final Project  
Skill-Based Job Role Predictor
