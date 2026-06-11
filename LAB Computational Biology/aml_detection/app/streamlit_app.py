"""Demo aplikasi Streamlit untuk deteksi AML dari ekspresi gen."""

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

# Path artefak relatif terhadap root proyek
MODEL_PATH = "models/aml_model.pkl"
META_PATH = "models/metadata.json"
TEST_SAMPLES_PATH = "models/test_samples.csv"


@st.cache_resource
def _muat_model():
    """Muat pipeline dari disk; di-cache agar tidak dibaca ulang tiap render."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def _muat_metadata() -> dict:
    """Muat metadata JSON dari disk."""
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def _muat_test_samples() -> pd.DataFrame:
    """Muat sampel test yang disimpan saat training."""
    return pd.read_csv(TEST_SAMPLES_PATH)


def muat_artefak():
    """Cek keberadaan semua file artefak, muat jika ada; kembalikan None jika belum."""
    for path in [MODEL_PATH, META_PATH, TEST_SAMPLES_PATH]:
        if not os.path.exists(path):
            return None, None, None
    model = _muat_model()
    metadata = _muat_metadata()
    df_test = _muat_test_samples()
    return model, metadata, df_test


def prediksi_sampel(model, X_row: pd.DataFrame) -> tuple[str, float]:
    """Prediksi satu baris data gen; return (label, confidence persen)."""
    label_idx = model.predict(X_row)[0]
    prob = model.predict_proba(X_row)[0]
    label = "AML" if label_idx == 1 else "Non-AML"
    confidence = round(float(prob.max()) * 100, 1)
    return label, confidence


def validasi_upload(df: pd.DataFrame, feature_columns: list[str]) -> tuple[bool, str]:
    """Periksa apakah semua kolom fitur yang diperlukan ada di CSV yang diunggah."""
    kolom_hilang = [k for k in feature_columns if k not in df.columns]
    if kolom_hilang:
        jumlah = len(kolom_hilang)
        contoh = ", ".join(kolom_hilang[:5])
        return False, f"{jumlah} kolom tidak ditemukan (mis. {contoh}...)"
    return True, ""


def render_sidebar(metadata: dict) -> None:
    """Tampilkan info model dan disclaimer di sidebar."""
    st.sidebar.title("Info Proyek")
    st.sidebar.markdown(
        "**Early Detection of AML** menggunakan machine learning pada data ekspresi gen "
        "(GSE9476, 64 sampel, 22.283 fitur)."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Terpilih")
    st.sidebar.metric("Algoritma", metadata["best_model"])

    tm = metadata["test_metrics"]
    st.sidebar.metric("Recall (Test)", tm["recall"])
    st.sidebar.metric("F1 Score (Test)", tm["f1"])
    st.sidebar.metric("ROC-AUC (Test)", tm["roc_auc"])

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "**Disclaimer:** Ini adalah proof-of-concept akademis (Kelompok 7, BINUS). "
        "BUKAN alat diagnosis klinis dan tidak boleh digunakan untuk pengambilan keputusan medis."
    )


def render_tab_sampel_test(model, df_test: pd.DataFrame) -> None:
    """Tab pertama: pilih sampel dari test set, tampilkan prediksi vs label asli."""
    st.subheader("Prediksi pada Sampel Test")
    st.caption("Pilih satu sampel dari test set yang digunakan saat evaluasi final.")

    feature_cols = [c for c in df_test.columns if c not in ("sample_id", "true_label")]

    # Buat opsi dropdown
    opsi = [
        f"Sampel {row.sample_id}  (Label Asli: {row.true_label})"
        for row in df_test.itertuples()
    ]
    pilihan = st.selectbox("Pilih sampel:", opsi)
    idx_terpilih = opsi.index(pilihan)
    baris = df_test.iloc[[idx_terpilih]]

    # Preview beberapa fitur pertama
    with st.expander("Lihat nilai ekspresi gen (5 fitur pertama)"):
        st.dataframe(baris[feature_cols[:5]], use_container_width=True)

    if st.button("Prediksi", key="btn_sampel"):
        X_row = baris[feature_cols]
        label, confidence = prediksi_sampel(model, X_row)
        true_label = baris["true_label"].values[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Prediksi Model")
            if label == "AML":
                st.error(f"**{label}**  ({confidence}% keyakinan)")
            else:
                st.success(f"**{label}**  ({confidence}% keyakinan)")
            st.progress(confidence / 100)

        with col2:
            st.markdown("#### Label Asli")
            if true_label == "AML":
                st.error(f"**{true_label}**")
            else:
                st.success(f"**{true_label}**")

        # Tampilkan apakah prediksi benar
        if label == true_label:
            st.info("Prediksi BENAR")
        else:
            st.warning("Prediksi SALAH (False Negative atau False Positive)")


def render_tab_upload(model, feature_columns: list[str]) -> None:
    """Tab kedua: upload CSV baru, prediksi semua barisnya."""
    st.subheader("Prediksi dari File CSV")
    st.caption(
        f"Upload CSV dengan header kolom gen ({len(feature_columns)} kolom). "
        "Kolom 'samples' dan 'type' boleh ada atau tidak."
    )
    st.info(
        "Format: satu baris = satu sampel. Header kolom harus sesuai dengan "
        "dataset training (nama gen yang sama)."
    )

    uploaded = st.file_uploader("Unggah file CSV:", type=["csv"])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

        # Hapus kolom metadata jika ada
        for col in ["samples", "type"]:
            if col in df_upload.columns:
                df_upload = df_upload.drop(columns=[col])

        valid, pesan_error = validasi_upload(df_upload, feature_columns)
        if not valid:
            st.error(f"Format CSV tidak valid: {pesan_error}")
            return

        X_upload = df_upload[feature_columns]

        with st.spinner("Memprediksi..."):
            prediksi = model.predict(X_upload)
            prob = model.predict_proba(X_upload)

        hasil = pd.DataFrame(
            {
                "No. Sampel": range(1, len(X_upload) + 1),
                "Prediksi": ["AML" if p == 1 else "Non-AML" for p in prediksi],
                "Keyakinan (%)": [round(float(p.max()) * 100, 1) for p in prob],
            }
        )

        # Warna baris berdasarkan prediksi
        def warnai_baris(row):
            warna = "background-color: #ffd6d6" if row["Prediksi"] == "AML" else "background-color: #d6f5d6"
            return [warna] * len(row)

        st.dataframe(
            hasil.style.apply(warnai_baris, axis=1),
            use_container_width=True,
        )

        # Ringkasan
        n_aml = int((prediksi == 1).sum())
        n_nonaml = int((prediksi == 0).sum())
        col1, col2 = st.columns(2)
        col1.metric("AML Terdeteksi", n_aml)
        col2.metric("Non-AML", n_nonaml)

        # Tombol unduh hasil
        csv_hasil = hasil.to_csv(index=False)
        st.download_button(
            "Unduh Hasil Prediksi (CSV)",
            data=csv_hasil,
            file_name="hasil_prediksi_aml.csv",
            mime="text/csv",
        )


def render_confusion_matrix(metadata: dict) -> None:
    """Buat heatmap confusion matrix dari metadata test set."""
    tm = metadata["test_metrics"]

    # Rekonstruksi confusion matrix dari nilai metrik
    # Gunakan test_samples jika tersedia untuk confusion matrix yang akurat
    try:
        df_test = _muat_test_samples()
        model = _muat_model()
        feature_cols = [c for c in df_test.columns if c not in ("sample_id", "true_label")]
        X_test = df_test[feature_cols]
        y_true = df_test["true_label"].map({"AML": 1, "Non-AML": 0})
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        # Fallback: estimasi kasar dari metrik
        cm = np.array([[0, 0], [0, 0]])

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-AML", "AML"],
        yticklabels=["Non-AML", "AML"],
        ax=ax,
    )
    ax.set_ylabel("Label Asli")
    ax.set_xlabel("Prediksi Model")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_tab_metrik(metadata: dict) -> None:
    """Tab ketiga: perbandingan CV, metrik test, confusion matrix, best params."""
    st.subheader("Performa Model")

    # Kartu metrik test set
    tm = metadata["test_metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Recall", tm["recall"])
    col2.metric("Precision", tm["precision"])
    col3.metric("F1 Score", tm["f1"])
    col4.metric("Accuracy", tm["accuracy"])
    col5.metric("ROC-AUC", tm["roc_auc"])

    st.markdown("---")

    # Tabel perbandingan RF vs SVM
    st.markdown("**Perbandingan CV (K=5, refit berdasarkan Recall)**")
    cv = metadata["cv_results"]
    tabel_cv = pd.DataFrame(
        [
            {
                "Model": "Random Forest",
                "Recall (CV)": cv["rf"]["recall_mean"],
                "Recall Std": cv["rf"]["recall_std"],
                "F1 (CV)": cv["rf"]["f1_mean"],
                "Precision (CV)": cv["rf"]["precision_mean"],
                "Accuracy (CV)": cv["rf"]["accuracy_mean"],
            },
            {
                "Model": "SVM RBF",
                "Recall (CV)": cv["svm"]["recall_mean"],
                "Recall Std": cv["svm"]["recall_std"],
                "F1 (CV)": cv["svm"]["f1_mean"],
                "Precision (CV)": cv["svm"]["precision_mean"],
                "Accuracy (CV)": cv["svm"]["accuracy_mean"],
            },
        ]
    )
    st.dataframe(tabel_cv, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_cm, col_params = st.columns([1, 1])

    with col_cm:
        st.markdown("**Confusion Matrix (Test Set)**")
        render_confusion_matrix(metadata)

    with col_params:
        st.markdown(f"**Hyperparameter Terbaik — {metadata['best_model']}**")
        nama_model_lower = "rf" if "Forest" in metadata["best_model"] else "svm"
        best_params = cv[nama_model_lower]["best_params"]
        st.json(best_params)

        st.markdown("**Info Dataset**")
        st.write(f"Jumlah fitur gen: {metadata['n_features']:,}")
        st.write(f"Variance threshold: {metadata['variance_threshold']}")


def main() -> None:
    st.set_page_config(
        page_title="Deteksi AML",
        page_icon=":microscope:",
        layout="wide",
    )
    st.title("Early Detection of Acute Myeloid Leukemia")
    st.caption(
        "Klasifikasi AML vs Non-AML dari profil ekspresi gen menggunakan Machine Learning | "
        "Kelompok 7 — Computational Biology, BINUS"
    )

    model, metadata, df_test = muat_artefak()

    # Tampilkan instruksi jika model belum ada
    if model is None:
        st.error("Model belum ditemukan di folder `models/`.")
        st.markdown(
            "**Langkah untuk generate model:**\n"
            "```bash\n"
            "python src/train.py --data dataset/Leukemia_GSE9476.csv\n"
            "```\n"
            "Setelah selesai, refresh halaman ini."
        )
        st.stop()

    render_sidebar(metadata)

    tab1, tab2, tab3 = st.tabs(["Sampel Test", "Upload CSV", "Metrik Model"])

    with tab1:
        render_tab_sampel_test(model, df_test)

    with tab2:
        render_tab_upload(model, metadata["feature_columns"])

    with tab3:
        render_tab_metrik(metadata)


if __name__ == "__main__":
    main()
