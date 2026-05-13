"""Tests for feature engineering pipeline."""
import numpy as np
import polars as pl
import pytest

from src.features.engineering import FeatureEngineer


def _make_df(n: int = 20, with_target: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data: dict = {
        "id": list(range(n)),
        "is_train": [1] * (n // 2) + [0] * (n - n // 2),
        "stres_skoru": rng.uniform(1, 9, n).tolist(),
        "rem_yuzdesi": rng.uniform(10, 30, n).tolist(),
        "derin_uyku_yuzdesi": rng.uniform(5, 25, n).tolist(),
        "gunluk_calisma_saati": rng.uniform(0, 12, n).tolist(),
        "gecelik_uyanma_sayisi": rng.integers(0, 5, n).tolist(),
        "gunluk_adim_sayisi": rng.integers(2000, 15000, n).tolist(),
        "uykuya_dalma_suresi_dk": rng.integers(5, 60, n).tolist(),
        "uyku_oncesi_kafein_mg": rng.uniform(0, 200, n).tolist(),
        "yas": rng.integers(20, 60, n).tolist(),
        "sekerleme_suresi_dk": rng.integers(0, 60, n).tolist(),
        "dinlenik_nabiz_bpm": rng.integers(50, 90, n).tolist(),
        "oda_sicakligi_celsius": rng.uniform(18, 26, n).tolist(),
        "hafta_sonu_uyku_farki_saat": rng.uniform(-2, 2, n).tolist(),
        "vucut_kitle_indeksi": rng.uniform(18, 35, n).tolist(),
        "uyku_oncesi_ekran_suresi_dk": rng.integers(0, 120, n).tolist(),
        "meslek": (["Ogretmen", "Muhendis", "Doktor"] * n)[:n],
        "ruh_sagligi_durumu": (["Saglikli", "Anksiyete"] * n)[:n],
        "gun_tipi": (["Hafta ici", "Hafta sonu"] * n)[:n],
        "kronotip": (["Sabahci", "Aksamci"] * n)[:n],
        "mevsim": (["Yaz", "Kis"] * n)[:n],
        "cinsiyet": (["Erkek", "Kadin"] * n)[:n],
        "ulke": ["TR"] * n,
    }
    if with_target:
        data["bilissel_performans_skoru"] = rng.uniform(3, 9, n).tolist()
    return pl.DataFrame(data)


class TestFeatureEngineer:
    def test_fit_transform_increases_columns(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        assert result.shape[1] > df.shape[1]

    def test_row_count_preserved(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        assert result.shape[0] == df.shape[0]

    def test_no_nulls_in_domain_features(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        domain_cols = ["sleep_efficiency", "sleep_fragmentation", "social_jetlag", "stres_squared"]
        for col in domain_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].null_count() == 0, f"Nulls in {col}"

    def test_sleep_efficiency_range(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        col = result["sleep_efficiency"].to_numpy()
        assert np.all(col >= 0)

    def test_age_zscore_train_mean_near_zero(self) -> None:
        df = _make_df(n=100)
        result = FeatureEngineer.fit_transform(df)
        train_mask = result["is_train"] == 1
        zscore = result.filter(train_mask)["stres_skoru_age_zscore"].to_numpy()
        assert abs(zscore.mean()) < 0.5

    def test_smoothed_te_present(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        assert "meslek_ruh_te" in result.columns
        assert "all_cat_te" in result.columns

    def test_sample_weights_mean_one(self) -> None:
        y = np.random.default_rng(1).uniform(0, 10, 200)
        w = FeatureEngineer.calculate_sample_weights(y)
        assert abs(w.mean() - 1.0) < 0.01

    def test_sample_weights_outliers_downweighted(self) -> None:
        y = np.zeros(100)
        y[0] = 100.0
        w = FeatureEngineer.calculate_sample_weights(y)
        assert w[0] < w[1:-1].mean()

    def test_no_target_in_test_split_needed(self) -> None:
        df = _make_df(with_target=True)
        result = FeatureEngineer.fit_transform(df)
        assert result.shape[0] == 20

    def test_pca_columns_created(self) -> None:
        df = _make_df(n=40)
        result = FeatureEngineer.fit_transform(df)
        for i in range(1, 6):
            assert f"pca_{i}" in result.columns

    def test_frequency_encoding_nonnegative(self) -> None:
        df = _make_df()
        result = FeatureEngineer.fit_transform(df)
        assert "freq_meslek" in result.columns
        assert (result["freq_meslek"] >= 0).all()
