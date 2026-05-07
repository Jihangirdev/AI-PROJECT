import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve


def model_chart(results, path):
    results = results.sort_values("macro_f1", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(results["model"], results["macro_f1"])
    plt.title("Model Comparison by Macro F1")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def confusion_chart(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(14, 10))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        xticks_rotation=30,
        ax=ax,
    )

    ax.set_title("Confusion Matrix", fontsize=14)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def learning_chart(model, X, y, path):
    sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="f1_macro",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="Train")
    plt.plot(sizes, test_scores.mean(axis=1), marker="o", label="Validation")
    plt.title("Learning Curve")
    plt.xlabel("Training size")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def class_distribution_chart(y, path):
    counts = y.value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Job role")
    plt.ylabel("Number of rows")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def source_distribution_chart(df, path):
    if "source" not in df.columns:
        return False

    counts = df["source"].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Dataset Source Distribution")
    plt.xlabel("Source")
    plt.ylabel("Rows")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return True


def top_features_chart(model, path, top_n=20):
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    feature_names = prep.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = np.abs(clf.coef_).mean(axis=0)
    else:
        return False

    idx = np.argsort(scores)[-top_n:]
    names = feature_names[idx]
    values = scores[idx]

    cleaned_names = []

    for name in names:
        for prefix in ["skills__", "qualification__", "soft__", "cat__", "num__"]:
            name = name.replace(prefix, "")
        cleaned_names.append(name)

    plt.figure(figsize=(10, 7))
    plt.barh(cleaned_names, values)
    plt.title("Top Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return True
