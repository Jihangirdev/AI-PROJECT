import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

# These columns are not used for training.
# qualification is removed because real API rows may contain the job title
# which can leak the answer into the input features
DROP_FROM_FEATURES = ["job_role", "source", "qualification", "language"]


def clean_data(df):
    #Clean dataset values before training or prediction
    df = df.copy()

    df["job_role"] = df["job_role"].replace({
        "CyberSecurity Analyst": "Cybersecurity Analyst"
    })

    text_cols = [
        "skills",
        "qualification",
        "education_level",
        "language",
        "certification",
        "soft_skills",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
    df["experience_years"] = df["experience_years"].fillna(df["experience_years"].median())

    return df


def split_xy(df):
    #Split data into features and target without leaking job titles
    drop_cols = [col for col in DROP_FROM_FEATURES if col in df.columns]
    X = df.drop(drop_cols, axis=1)
    y = df["job_role"]
    return X, y


def prepare_candidate(row):
    #Prepare one input row for prediction using the same safe features
    drop_cols = [col for col in ["source", "qualification", "language"] if col in row.columns]
    return row.drop(drop_cols, axis=1)


def preprocessor():
    #Create preprocessing for safe features only
    return ColumnTransformer([
        ("skills", TfidfVectorizer(ngram_range=(1, 2), min_df=1), "skills"),
        ("soft", TfidfVectorizer(ngram_range=(1, 2), min_df=1), "soft_skills"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["education_level", "certification"]),
        ("num", StandardScaler(), ["experience_years"]),
    ])


def make_model(clf):
    #Combine preprocessing and classifier into one sklearn pipeline
    return Pipeline([
        ("prep", preprocessor()),
        ("clf", clf),
    ])


def evaluate(name, model, X_test, y_test):
    #Print and return hold-out test metrics for one model
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(f1, 4))
    print(classification_report(y_test, pred, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": f1,
    }


def cv_score(name, model, X, y):
    #Calculate 5-fold cross-validation macro F1 for a model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)

    return {
        "model": name,
        "cv_macro_f1_mean": scores.mean(),
        "cv_macro_f1_std": scores.std(),
    }


def train_simple_models(X_train, y_train, X_test, y_test):
    #Train default models and compare them on the test set
    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    trained = {}

    for name, clf in models.items():
        model = make_model(clf)
        model.fit(X_train, y_train)

        results.append(evaluate(name, model, X_test, y_test))
        trained[name] = model

    results = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    return results, trained


def cross_validate_simple_models(X, y):
    #Run the same 5-fold CV metric for all simple models
    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    for name, clf in models.items():
        rows.append(cv_score(name, make_model(clf), X, y))

    return pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)


def tune_logistic_regression(X_train, y_train):
    #Tune Logistic Regression with GridSearchCV
    model = make_model(LogisticRegression(max_iter=1500, random_state=RANDOM_STATE))

    params = {
        "clf__C": [0.05, 0.1, 0.5, 1, 3, 5, 10],
        "clf__solver": ["lbfgs"],
    }

    grid = GridSearchCV(model, params, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_random_forest(X_train, y_train):
    #Tune Random Forest with a small but useful parameter grid
    model = make_model(RandomForestClassifier(random_state=RANDOM_STATE))

    params = {
        "clf__n_estimators": [150, 250, 350],
        "clf__max_depth": [8, 12, None],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(model, params, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_extra_trees(X_train, y_train):
    #Tune Extra Trees with the same style of grid as Random Forest
    model = make_model(ExtraTreesClassifier(random_state=RANDOM_STATE))

    params = {
        "clf__n_estimators": [150, 250, 350],
        "clf__max_depth": [8, 12, None],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features": ["sqrt", "log2"],
    }

    grid = GridSearchCV(model, params, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_models_and_choose_best(X_train, y_train, X_test, y_test):
    #Tune models and choose the final one by CV macro F1, not by leakage
    tuned = {}

    lr, lr_params, lr_cv = tune_logistic_regression(X_train, y_train)
    tuned["Tuned Logistic Regression"] = {
        "model": lr,
        "params": lr_params,
        "cv_macro_f1": lr_cv,
    }

    rf, rf_params, rf_cv = tune_random_forest(X_train, y_train)
    tuned["Tuned Random Forest"] = {
        "model": rf,
        "params": rf_params,
        "cv_macro_f1": rf_cv,
    }

    et, et_params, et_cv = tune_extra_trees(X_train, y_train)
    tuned["Tuned Extra Trees"] = {
        "model": et,
        "params": et_params,
        "cv_macro_f1": et_cv,
    }

    rows = []

    print("\n" + "=" * 60)
    print("TUNED MODEL COMPARISON")
    print("=" * 60)

    for name, item in tuned.items():
        model = item["model"]
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        rows.append({
            "model": name,
            "accuracy": acc,
            "macro_f1": f1,
            "cv_macro_f1": item["cv_macro_f1"],
            "params": str(item["params"]),
        })

        print("\n" + name)
        print("Best params:", item["params"])
        print("CV macro F1:", round(item["cv_macro_f1"], 4))
        print("Test accuracy:", round(acc, 4))
        print("Test macro F1:", round(f1, 4))

    results = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)

    best_name = results.iloc[0]["model"]
    best_model = tuned[best_name]["model"]

    print("\nFinal selected model:", best_name)
    print("Selection rule: highest cross-validation macro F1.")

    return best_name, best_model, results


def predict_one(model, row):
    #Predict one candidate row and return probabilities for all roles
    row = prepare_candidate(row)

    pred = model.predict(row)[0]
    probs = model.predict_proba(row)[0]

    probs = pd.DataFrame({
        "job_role": model.classes_,
        "probability": probs,
    }).sort_values("probability", ascending=False)

    return pred, probs


def save_model(model, path):
    #Save trained model.
    joblib.dump(model, path)


def load_model(path):
    #Load saved model.
    return joblib.load(path)
