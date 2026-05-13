from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    MODELS_DIR: Path = PROJECT_DIR / "models"

    DEVICE: str = "cpu"
    SEEDS: list[int] = [42, 2026]
    N_FOLDS: int = 5
    TARGET_COL: str = "bilissel_performans_skoru"
    ID_COL: str = "id"

    WINSOR_LIMITS: tuple[float, float] = (1.0, 99.0)

    CAT_FEATURES: list[str] = [
        "cinsiyet", "meslek", "ulke", "kronotip",
        "ruh_sagligi_durumu", "mevsim", "gun_tipi",
    ]
    CAT_INTERACTION_FEATURES: list[str] = [
        "chronotype_daytype", "meslek_ruh_sagligi",
        "meslek_gun_tipi", "ruh_sagligi_gun_tipi",
    ]
    NUM_FEATURES_BASE: list[str] = [
        "yas", "vucut_kitle_indeksi", "rem_yuzdesi",
        "derin_uyku_yuzdesi", "uykuya_dalma_suresi_dk",
        "gecelik_uyanma_sayisi", "uyku_oncesi_kafein_mg",
        "uyku_oncesi_ekran_suresi_dk", "gunluk_adim_sayisi",
        "sekerleme_suresi_dk", "stres_skoru",
        "gunluk_calisma_saati", "dinlenik_nabiz_bpm",
        "oda_sicakligi_celsius", "hafta_sonu_uyku_farki_saat",
    ]

    # Optuna HPO
    HPO_TRIALS: int = 60
    HPO_FOLDS: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
for d in [settings.DATA_DIR, settings.RAW_DATA_DIR, settings.MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
