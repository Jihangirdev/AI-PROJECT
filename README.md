# Job Role Finder

This project predicts an IT job role from a candidate skill profile.
It uses supervised machine learning with a clean sklearn pipeline.

## How to install

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python main.py
```

This creates output files in `outputs/`:

- `job_role_model.pkl`
- `model_results.csv`
- `tuned_model_results.csv`
- `cross_validation_results.csv`
- `selected_model.csv`
- `error_analysis.csv`
- `model_comparison.png`
- `confusion_matrix.png`
- `learning_curve.png`
- `class_distribution.png`
- `top_features.png`

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Run desktop UI

```bash
python desktop_app.py
```

## Models used

- Dummy Classifier baseline
- Logistic Regression
- KNN
- Random Forest
- Extra Trees

The final model is selected by cross-validation macro F1 among tuned models.
