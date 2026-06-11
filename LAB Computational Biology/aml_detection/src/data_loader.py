"""Modul untuk memuat dan memvalidasi dataset ekspresi gen leukemia."""

import pandas as pd


def _validasi_kolom(df: pd.DataFrame) -> None:
    """Pastikan kolom wajib ada; raise ValueError dengan pesan jelas jika tidak."""
    if "type" not in df.columns:
        raise ValueError(
            "Kolom 'type' tidak ditemukan dalam dataset. "
            "Pastikan file CSV menggunakan format yang benar (kolom: samples, type, gen1, gen2, ...)."
        )


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Muat dataset dari CSV dan pisahkan fitur dari label.

    Parameter
    ---------
    path : str
        Path ke file CSV dataset (relatif terhadap root proyek).

    Returns
    -------
    X : pd.DataFrame
        DataFrame fitur ekspresi gen (numerik murni, tanpa kolom metadata).
    y_raw : pd.Series
        Series label kelas asli 5-kategori (belum dikonversi ke biner).
    """
    df = pd.read_csv(path)
    _validasi_kolom(df)

    # Buang kolom metadata; 'samples' bersifat opsional (mungkin tidak ada di CSV upload)
    kolom_buang = [k for k in ["samples", "type"] if k in df.columns]
    X = df.drop(columns=kolom_buang)
    y_raw = df["type"]

    return X, y_raw


def to_binary(y: pd.Series) -> pd.Series:
    """Konversi label 5-kelas ke label biner AML vs Non-AML.

    Mapping: AML -> 1, semua kelas lain (Bone_Marrow, PB, dll.) -> 0.

    Parameter
    ---------
    y : pd.Series
        Label kelas asli dari kolom 'type'.

    Returns
    -------
    pd.Series
        Series integer: 1 untuk AML, 0 untuk Non-AML.
    """
    return (y == "AML").astype(int)
