# Early Detection of Acute Myeloid Leukemia (AML)

Proyek machine learning untuk mendeteksi AML vs Non-AML dari data ekspresi gen (GSE9476).
**Ini adalah proof-of-concept akademis (Kelompok 7, BINUS) dan bukan alat diagnosis klinis.**

---

## Prasyarat

- Python 3.10 atau lebih baru
- Dataset: `Leukemia_GSE9476.csv` (unduh dari Kaggle — lihat bagian Persiapan Dataset)

---

## Instalasi

```bash
# Buat virtual environment (opsional tapi disarankan)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

---

## Persiapan Dataset

1. Unduh dataset dari Kaggle:
   `https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida`
2. Pilih file `Leukemia_GSE9476.csv`
3. Letakkan file tersebut di folder `dataset/`:
   ```
   aml_detection/
   └── dataset/
       └── Leukemia_GSE9476.csv
   ```

---

## Cara Menjalankan

### 1. Training Model (wajib sebelum demo)

```bash
python src/train.py --data dataset/Leukemia_GSE9476.csv
```

Opsional: tentukan direktori output model (default: `models/`):

```bash
python src/train.py --data dataset/Leukemia_GSE9476.csv --output-dir models
```

Artefak yang dihasilkan:
- `models/aml_model.pkl` — pipeline sklearn lengkap (preprocessing + model)
- `models/metadata.json` — metrik, parameter terbaik, nama kolom fitur
- `models/test_samples.csv` — sampel test untuk demo Streamlit

### 2. Demo Aplikasi Streamlit

```bash
streamlit run app/streamlit_app.py
```

Buka browser di `http://localhost:8501`.

Fitur:
- **Tab Sampel Test**: pilih sampel dari test set, lihat prediksi vs label asli
- **Tab Upload CSV**: upload file ekspresi gen baru, dapatkan prediksi batch
- **Tab Metrik Model**: confusion matrix, perbandingan RF vs SVM, hyperparameter terbaik

### 3. Notebook Eksperimen

```bash
jupyter notebook notebook/aml_experiment.ipynb
```

Jalankan semua sel dari atas ke bawah (Kernel > Restart & Run All).

---

## Struktur File

```
aml_detection/
├── CLAUDE.md                    # Konteks proyek untuk Claude Code
├── README.md                    # Panduan ini
├── requirements.txt             # Dependensi Python
├── dataset/
│   └── Leukemia_GSE9476.csv     # Dataset (diletakkan manual, tidak di-commit)
├── notebook/
│   └── aml_experiment.ipynb    # Notebook eksperimen lengkap
├── src/
│   ├── data_loader.py           # Fungsi load dan validasi dataset
│   └── train.py                 # Script CLI training
├── models/
│   ├── aml_model.pkl            # Pipeline tersimpan (hasil training)
│   ├── metadata.json            # Metrik dan konfigurasi model
│   └── test_samples.csv         # Sampel test untuk demo
└── app/
    └── streamlit_app.py         # Aplikasi demo
```

---

## Metodologi Singkat

| Langkah | Detail |
|---------|--------|
| Dataset | 64 sampel × 22.283 fitur gen; 5 kelas → biner (AML=1, lainnya=0) |
| Split | 75/25 train/test, stratified, random_state=42 |
| Pipeline | VarianceThreshold → StandardScaler → PCA → Classifier |
| CV | StratifiedKFold K=5 di dalam GridSearchCV (anti data leakage) |
| Model | Random Forest vs SVM RBF; pilih berdasarkan Recall tertinggi |
| Metrik utama | Recall (prioritas karena False Negative berbahaya di konteks medis) |
| Simpan | Seluruh pipeline (bukan model saja) disimpan dengan joblib |

---

## Disclaimer

Proyek ini dibuat untuk keperluan akademis mata kuliah Computational Biology di BINUS University.
Hasil klasifikasi **tidak boleh** digunakan sebagai dasar diagnosis medis.
