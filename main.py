from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score

from model import split_xy, train_simple_models, cross_validate_simple_models
from train_utils import OUT, load_project_data, get_train_test, train_best_model
from visualization import (
    class_distribution_chart,
    confusion_chart,
    learning_chart,
    model_chart,
    top_features_chart,
)


def save_errors(model, X_test, y_test):
    #Save wrong predictions for error analysis
    pred = model.predict(X_test)

    errors = X_test.copy()
    errors["true_role"] = y_test.values
    errors["predicted_role"] = pred

    wrong = errors[errors["true_role"] != errors["predicted_role"]]
    wrong.to_csv(f"{OUT}/error_analysis.csv", index=False)

    print("\nWrong predictions:", len(wrong), "out of", len(X_test))


def main():
    Path(OUT).mkdir(exist_ok=True)

    df = load_project_data()

    print("Rows and columns:", df.shape)
    print("\nClasses:")
    print(df["job_role"].value_counts())

    if "source" in df.columns:
        print("\nSource distribution:")
        print(df["source"].value_counts(normalize=True).round(3))

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = get_train_test(df)

    print("\nUsed features:")
    print(list(X.columns))
    print("\nRemoved from training: qualification, source, language")

    simple_results, simple_models = train_simple_models(X_train, y_train, X_test, y_test)

    print("\nSimple model hold-out results:")
    print(simple_results)

    cv_results = cross_validate_simple_models(X, y)
    cv_results.to_csv(f"{OUT}/cross_validation_results.csv", index=False)

    print("\nSimple model 5-fold CV results:")
    print(cv_results)

    best_name, best_model, tuned_results = train_best_model()
    pred = best_model.predict(X_test)

    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)
    print("Selected:", best_name)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Macro F1:", round(f1_score(y_test, pred, average="macro"), 4))
    print(classification_report(y_test, pred, zero_division=0))

    all_results = pd.concat([simple_results, tuned_results], ignore_index=True)
    all_results.to_csv(f"{OUT}/model_results.csv", index=False)
    tuned_results.to_csv(f"{OUT}/tuned_model_results.csv", index=False)

    save_errors(best_model, X_test, y_test)

    model_chart(all_results, f"{OUT}/model_comparison.png")
    confusion_chart(best_model, X_test, y_test, f"{OUT}/confusion_matrix.png")
    learning_chart(best_model, X, y, f"{OUT}/learning_curve.png")
    class_distribution_chart(y, f"{OUT}/class_distribution.png")
    top_features_chart(best_model, f"{OUT}/top_features.png")

    print("\nDone. Files saved in outputs folder.")


if __name__ == "__main__":
    main()
