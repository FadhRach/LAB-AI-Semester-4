"""Script CLI untuk melatih dan menyimpan pipeline deteksi AML terbaik."""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

# Precision bisa undefined di beberapa fold grid search (model prediksi satu kelas saja) — itu wajar
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Agar src/ dapat mengimpor data_loader saat dijalankan dari root proyek
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_data, to_binary

RANDOM_STATE = 42
TEST_SIZE = 0.25
VARIANCE_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latih pipeline deteksi AML dan simpan model terbaik."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path ke file CSV dataset (mis. dataset/Leukemia_GSE9476.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Direktori tujuan untuk menyimpan artefak (default: models)",
    )
    return parser.parse_args()


def muat_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Muat dataset dan konversi label ke biner."""
    X, y_raw = load_data(path)
    y = to_binary(y_raw)
    print(f"Dataset dimuat: {X.shape[0]} sampel, {X.shape[1]} fitur")
    print(f"  AML (positif): {y.sum()} | Non-AML (negatif): {(y == 0).sum()}")
    return X, y


def bagi_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset menjadi train dan test (75/25, stratified)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nSplit data: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def buat_pipeline_rf() -> tuple[Pipeline, dict]:
    """Buat pipeline Random Forest beserta param_grid-nya."""
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE, class_weight="balanced"
                ),
            ),
        ]
    )
    # PCA n_components dibatasi 35 agar aman pada fold terkecil (~38 sampel)
    param_grid = {
        "pca__n_components": [20, 30, 35],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    return pipeline, param_grid


def buat_pipeline_svm() -> tuple[Pipeline, dict]:
    """Buat pipeline SVM RBF beserta param_grid-nya."""
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    param_grid = {
        "pca__n_components": [20, 30, 35],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
    }
    return pipeline, param_grid


def jalankan_gridsearch(
    pipe: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> GridSearchCV:
    """Jalankan GridSearchCV dengan multi-scoring; refit berdasarkan recall."""
    scoring = {
        "recall": "recall",
        "f1": "f1",
        "precision": "precision",
        "accuracy": "accuracy",
    }
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)
    return gs


def ambil_cv_metrik(gs: GridSearchCV) -> dict:
    """Ambil metrik CV terbaik dari hasil GridSearch."""
    idx = gs.best_index_
    cvr = gs.cv_results_
    return {
        "recall_mean": round(float(cvr["mean_test_recall"][idx]), 4),
        "recall_std": round(float(cvr["std_test_recall"][idx]), 4),
        "f1_mean": round(float(cvr["mean_test_f1"][idx]), 4),
        "f1_std": round(float(cvr["std_test_f1"][idx]), 4),
        "precision_mean": round(float(cvr["mean_test_precision"][idx]), 4),
        "accuracy_mean": round(float(cvr["mean_test_accuracy"][idx]), 4),
        "best_params": gs.best_params_,
    }


def pilih_model_terbaik(
    gs_rf: GridSearchCV, gs_svm: GridSearchCV
) -> tuple[str, object]:
    """Pilih model terbaik berdasarkan recall CV tertinggi; F1 sebagai tiebreaker."""
    cv_rf = ambil_cv_metrik(gs_rf)
    cv_svm = ambil_cv_metrik(gs_svm)

    print("\n=== Perbandingan CV ===")
    print(
        f"  Random Forest: Recall={cv_rf['recall_mean']:.4f} (+/-{cv_rf['recall_std']:.4f}), "
        f"F1={cv_rf['f1_mean']:.4f}"
    )
    print(
        f"  SVM RBF      : Recall={cv_svm['recall_mean']:.4f} (+/-{cv_svm['recall_std']:.4f}), "
        f"F1={cv_svm['f1_mean']:.4f}"
    )

    if gs_rf.best_score_ > gs_svm.best_score_:
        nama = "Random Forest"
        model = gs_rf.best_estimator_
    elif gs_svm.best_score_ > gs_rf.best_score_:
        nama = "SVM RBF"
        model = gs_svm.best_estimator_
    else:
        # Recall sama — bandingkan F1
        if cv_rf["f1_mean"] >= cv_svm["f1_mean"]:
            nama = "Random Forest"
            model = gs_rf.best_estimator_
        else:
            nama = "SVM RBF"
            model = gs_svm.best_estimator_

    print(f"\nModel terpilih: {nama}")
    return nama, model


def evaluasi_test(
    model: object, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict, pd.Series, pd.Series]:
    """Evaluasi model pada test set; return metrik, prediksi, dan probabilitas."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrik = {
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    return metrik, y_pred, y_prob


def simpan_artefak(
    model: object,
    nama_model: str,
    cv_rf: dict,
    cv_svm: dict,
    metrik_test: dict,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Simpan pipeline, metadata JSON, dan sampel test ke direktori output."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline utuh (preprocessing + model)
    joblib.dump(model, dir_path / "aml_model.pkl")
    print(f"Pipeline tersimpan: {dir_path / 'aml_model.pkl'}")

    # 2. Metadata JSON
    metadata = {
        "best_model": nama_model,
        "variance_threshold": VARIANCE_THRESHOLD,
        "cv_results": {"rf": cv_rf, "svm": cv_svm},
        "test_metrics": metrik_test,
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
    }
    with open(dir_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata tersimpan: {dir_path / 'metadata.json'}")

    # 3. Sampel test untuk demo Streamlit
    df_test = X_test.copy()
    df_test["sample_id"] = X_test.index.astype(str)
    df_test["true_label"] = y_test.map({1: "AML", 0: "Non-AML"}).values
    df_test.to_csv(dir_path / "test_samples.csv", index=False)
    print(f"Sampel test tersimpan: {dir_path / 'test_samples.csv'}")


def main() -> None:
    args = parse_args()

    X, y = muat_data(args.data)
    X_train, X_test, y_train, y_test = bagi_data(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Latih Random Forest
    print("\n--- GridSearch: Random Forest ---")
    pipe_rf, grid_rf = buat_pipeline_rf()
    gs_rf = jalankan_gridsearch(pipe_rf, grid_rf, X_train, y_train, cv)

    # Latih SVM
    print("\n--- GridSearch: SVM RBF ---")
    pipe_svm, grid_svm = buat_pipeline_svm()
    gs_svm = jalankan_gridsearch(pipe_svm, grid_svm, X_train, y_train, cv)

    # Pilih model terbaik
    nama_model, model_terbaik = pilih_model_terbaik(gs_rf, gs_svm)

    # Evaluasi final pada test set yang ditahan
    print("\n=== Evaluasi Final (Test Set) ===")
    metrik_test, _, _ = evaluasi_test(model_terbaik, X_test, y_test)
    for k, v in metrik_test.items():
        print(f"  {k}: {v}")

    # Simpan semua artefak
    cv_rf = ambil_cv_metrik(gs_rf)
    cv_svm = ambil_cv_metrik(gs_svm)
    simpan_artefak(
        model_terbaik, nama_model, cv_rf, cv_svm,
        metrik_test, X, X_test, y_test, args.output_dir
    )

    print("\nSelesai. Jalankan demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
