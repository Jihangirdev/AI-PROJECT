from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from model import (
    RANDOM_STATE,
    clean_data,
    split_xy,
    tune_models_and_choose_best,
    save_model,
    load_model,
)


DATA = "data/job_dataset.csv"
OUT = "outputs"
MODEL_FILE = "outputs/job_role_model.pkl"


def load_project_data():
    #Load and clean the final dataset
    df = pd.read_csv(DATA)
    df = clean_data(df)
    return df


def get_train_test(df):
    #Create train/test split stratified by job role
    X, y = split_xy(df)

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_best_model():
    #Train tuned models and save the best one
    Path(OUT).mkdir(exist_ok=True)

    df = load_project_data()
    X_train, X_test, y_train, y_test = get_train_test(df)

    best_name, best_model, tuned_results = tune_models_and_choose_best(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_model(best_model, MODEL_FILE)

    pd.DataFrame([{
        "selected_model": best_name,
        "data_file": DATA,
        "feature_note": "qualification, source, and language were removed from training",
    }]).to_csv(f"{OUT}/selected_model.csv", index=False)

    return best_name, best_model, tuned_results


def get_saved_or_train_model():
    #Load saved model if it exists, otherwise train it
    Path(OUT).mkdir(exist_ok=True)

    if Path(MODEL_FILE).exists():
        return load_model(MODEL_FILE)

    _, model, _ = train_best_model()
    return model
