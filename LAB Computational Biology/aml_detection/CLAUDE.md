# CLAUDE.md — Proyek Deteksi AML (Computational Biology, Kelompok 7)

Dokumen ini adalah konteks persisten untuk Claude Code. Baca seluruhnya sebelum mengerjakan tugas apa pun.

## 1. Tentang Proyek

**Judul:** Early Detection of Acute Myeloid Leukemia using Machine Learning on Gene Expression Data.

**Tujuan:** Membangun model machine learning yang mengklasifikasikan sampel sebagai **AML** atau
**Non-AML** berdasarkan profil ekspresi gen. Ini **proof-of-concept akademik**, BUKAN alat diagnosis klinis.
Jangan pernah menulis klaim bahwa sistem ini siap dipakai di rumah sakit.

**Rumusan masalah (spesifik & terukur):** Dapatkah Random Forest / SVM mengklasifikasikan AML vs Non-AML
dari ekspresi gen dengan **Recall ≥ 0,90** dan **F1 ≥ 0,90** pada validasi K-Fold?

**Mengapa Recall diprioritaskan:** Di konteks medis, *False Negative* (pasien AML diprediksi normal)
paling berbahaya. Recall mengukur kemampuan menangkap semua kasus AML, jadi itu metrik utama.

## 2. Dataset

- **Sumber:** CuMiDa, kode **GSE9476**, dari Kaggle:
  `https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida`
- **Ukuran:** 64 sampel (baris) × 22.283 fitur gen (kolom) + kolom metadata.
- **Struktur kolom CSV** (perkiraan; nama bisa sedikit beda — buat loader yang robust):
  - `samples` — ID sampel (buang sebelum training).
  - `type` — label kelas (target).
  - sisanya — nilai ekspresi gen (fitur numerik).
- **5 kelas pada `type`:**

  | Label | Jumlah | Arti | Status |
  |-------|--------|------|--------|
  | `AML` | 26 | Leukemia mieloid akut | **Sakit (1)** |
  | `Bone_Marrow` | 10 | Sumsum tulang sehat | Normal (0) |
  | `Bone_Marrow_CD34` | 8 | Sel induk sumsum sehat | Normal (0) |
  | `PB` | 10 | Darah tepi normal | Normal (0) |
  | `PBSC_CD34` | 10 | Sel induk dari darah | Normal (0) |

- **Pelabelan biner:** `AML → 1`, semua kelas lain `→ 0`. Hasil: 26 positif vs 38 negatif (ada *class imbalance* ringan).

## 3. Aturan Metodologi (WAJIB DIPATUHI)

Ini bagian terpenting. Pelanggaran membuat hasil eksperimen tidak valid secara ilmiah.

1. **CEGAH DATA LEAKAGE.** Jangan pernah `fit` scaler/PCA pada seluruh data lalu baru di-split atau di-CV.
   Bungkus `VarianceThreshold → StandardScaler → PCA → classifier` dalam **satu `sklearn.pipeline.Pipeline`**,
   lalu lakukan CV / GridSearch pada pipeline itu. Dengan begitu preprocessing hanya di-`fit` pada fold latih.
2. **Gunakan `StratifiedKFold`** (bukan KFold biasa) karena data kecil dan tidak seimbang. K = 5.
3. **Tahan test set.** Pakai `train_test_split(..., stratify=y, test_size=0.25, random_state=42)`.
   Evaluasi final HANYA pada test set ini, sekali, di akhir.
4. **Reproducibility.** Set `random_state=42` di semua tempat (split, RF, SVM, PCA, KFold, GridSearch).
5. **Curse of dimensionality.** 22.283 fitur vs 64 sampel → feature reduction WAJIB. Target PCA `n_components`
   sekitar 30–50 (jadikan ini parameter yang dituning, dibatasi `min(n_samples, n_features)`).
6. **Metrik.** Selalu laporkan Recall, F1, Precision, Accuracy + confusion matrix. Jangan andalkan Accuracy saja
   (menyesatkan saat imbalance). Tambahkan ROC-AUC pada evaluasi final.
7. **Simpan pipeline utuh.** Saat menyimpan model untuk demo, simpan SELURUH pipeline (preprocessing + model)
   dengan `joblib`, supaya aplikasi Streamlit tidak perlu mengulang preprocessing secara manual.

## 4. Tech Stack & Konvensi

- **Python 3.10+**. Library: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.
- **Model:** Random Forest dan SVM (RBF). Bandingkan keduanya; pilih yang terbaik berdasarkan Recall lalu F1.
- **Gaya kode:** PEP 8, fungsi kecil dan jelas, type hints jika memungkinkan.
- **Komentar & teks:** Bahasa Indonesia untuk penjelasan; istilah teknis (pipeline, recall, dll.) boleh Inggris.
- **Angka tampilan:** selalu `round()` / `.toFixed`-equivalent agar tidak muncul float panjang.
- **Jangan** hardcode path absolut; pakai path relatif terhadap root proyek atau `argparse`.

## 5. Struktur File yang Diharapkan

```
aml-detection/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/            # dataset CSV diletakkan manual oleh user (jangan commit dataset besar)
├── notebooks/aml_experiment.ipynb
├── src/data_loader.py
├── src/train.py
├── models/          # aml_model.pkl + metadata.json (hasil training)
└── app/streamlit_app.py
```

## 6. Kontrak Antar-Komponen

- `src/data_loader.py` mengekspor `load_data(path) -> (X: DataFrame, y: Series)` yang:
  membuang kolom non-fitur (`samples`, `type`), mengembalikan fitur numerik + label mentah.
  Sediakan juga `to_binary(y) -> Series` (`AML→1`, lainnya→0). Validasi: error jelas jika kolom `type` tak ada.
- `src/train.py` memakai `data_loader`, melatih pipeline terbaik pada seluruh data, menyimpan
  `models/aml_model.pkl` (pipeline) dan `models/metadata.json` (kelas, metrik CV, daftar contoh sampel test untuk demo).
- `app/streamlit_app.py` hanya **memuat** `aml_model.pkl` + `metadata.json`; tidak melatih ulang.
  Jika model belum ada, tampilkan instruksi menjalankan `python src/train.py` dulu.

## 7. Hal yang HARUS Dihindari

- Jangan klaim akurasi/hasil tanpa benar-benar menjalankan kode.
- Jangan menulis preprocessing di luar pipeline saat melakukan CV (data leakage).
- Jangan menyimpan dataset mentah ke git; cukup `.gitkeep` di folder `data/` dan `models/`.
- Jangan membuat UI Streamlit yang meminta user mengetik 22.283 angka manual — gunakan pilih-sampel atau upload CSV.
- Jangan over-engineer: ini proyek kuliah, utamakan kejelasan dan kebenaran metodologi di atas kompleksitas.

## 8. Definisi Selesai (Acceptance Criteria)

- [ ] Notebook berjalan top-to-bottom tanpa error.
- [ ] CV memakai pipeline (tidak ada leakage), StratifiedKFold K=5, random_state konsisten.
- [ ] RF dan SVM dibandingkan; confusion matrix + Recall/F1 ditampilkan.
- [ ] Evaluasi final di test set yang ditahan, lengkap dengan ROC-AUC.
- [ ] Pipeline terbaik tersimpan di `models/aml_model.pkl` + metadata.
- [ ] `src/train.py` bisa meregenerasi model dari CLI.
- [ ] Streamlit jalan dengan `streamlit run app/streamlit_app.py`, bisa prediksi & menampilkan keyakinan.
- [ ] README menjelaskan setup, dataset, dan cara menjalankan semuanya.

## 9. Code Style Requirements

Semua notebook dan code mengikuti konvensi berikut:

- **Clean code**: setiap fungsi punya satu tanggung jawab yang jelas
- **Reusable**: helper atau function didefinisikan sekali, dipanggil berkali-kali — bukan inline copy-paste
- **Minimalis**: tidak ada code yang tidak dipakai; tidak ada abstraksi prematur
- **Beginner-friendly**: nama variabel deskriptif; alur code linear, mudah diikuti dari atas ke bawah
- **Komentar seperlunya**: tulis komentar hanya untuk sesuatu yang tidak obvious dari nama variabel/fungsi itu sendiri
- **Tanpa emotikon di code**: jangan gunakan emoji di dalam code cell maupun string output