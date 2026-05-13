"""Tests for data loading, schema validation and config."""
import pytest
import polars as pl
import numpy as np

from src.config import settings
from src.data.schema import DatathonSchema


_VALID_ROW: dict = {
    "id": 1,
    "yas": 30,
    "cinsiyet": "Erkek",
    "ulke": "TR",
    "rem_yuzdesi": 20.0,
    "derin_uyku_yuzdesi": 15.0,
    "uykuya_dalma_suresi_dk": 10,
    "gecelik_uyanma_sayisi": 1,
    "uyku_oncesi_ekran_suresi_dk": 30,
    "gunluk_adim_sayisi": 8000,
    "sekerleme_suresi_dk": 0,
    "gunluk_calisma_saati": 8.0,
    "dinlenik_nabiz_bpm": 65,
    "oda_sicakligi_celsius": 21.0,
    "hafta_sonu_uyku_farki_saat": 1.0,
    "mevsim": "Yaz",
    "gun_tipi": "Hafta_ici",
}


class TestSettings:
    def test_data_dir_name(self) -> None:
        assert settings.DATA_DIR.name == "data"

    def test_project_dir_exists(self) -> None:
        assert settings.PROJECT_DIR.exists()

    def test_seeds_nonempty(self) -> None:
        assert len(settings.SEEDS) > 0

    def test_n_folds_positive(self) -> None:
        assert settings.N_FOLDS > 1

    def test_target_col_defined(self) -> None:
        assert settings.TARGET_COL != ""

    def test_id_col_defined(self) -> None:
        assert settings.ID_COL != ""


class TestSchema:
    def test_valid_row(self) -> None:
        s = DatathonSchema(**_VALID_ROW)
        assert s.yas == 30

    def test_age_below_minimum_rejected(self) -> None:
        bad = {**_VALID_ROW, "yas": 17}
        with pytest.raises(Exception):
            DatathonSchema(**bad)

    def test_age_above_maximum_rejected(self) -> None:
        bad = {**_VALID_ROW, "yas": 101}
        with pytest.raises(Exception):
            DatathonSchema(**bad)

    def test_rem_above_100_rejected(self) -> None:
        bad = {**_VALID_ROW, "rem_yuzdesi": 101.0}
        with pytest.raises(Exception):
            DatathonSchema(**bad)

    def test_negative_steps_rejected(self) -> None:
        bad = {**_VALID_ROW, "gunluk_adim_sayisi": -1}
        with pytest.raises(Exception):
            DatathonSchema(**bad)

    def test_optional_fields_none(self) -> None:
        s = DatathonSchema(**_VALID_ROW)
        assert s.stres_skoru is None
        assert s.kronotip is None

    def test_target_optional(self) -> None:
        s = DatathonSchema(**{**_VALID_ROW, "bilissel_performans_skoru": 7.5})
        assert s.bilissel_performans_skoru == pytest.approx(7.5)
