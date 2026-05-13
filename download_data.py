"""Kaggle'dan yarışma verisini indir ve data/raw/ altına çıkart."""

import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

COMPETITION = "yzta-2026-datathon"
DATA_DIR = Path("data/raw")


def main() -> None:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise SystemExit(
            "Hata: .env dosyasında KAGGLE_USERNAME ve KAGGLE_KEY tanımlı olmalı.\n"
            "Örnek için .env.example dosyasına bak."
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    import kaggle  # noqa: PLC0415  (import after env vars set)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"İndiriliyor: {COMPETITION}")
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition=COMPETITION,
        path=str(DATA_DIR),
        quiet=False,
    )

    for zf in DATA_DIR.glob("*.zip"):
        print(f"Çıkartılıyor: {zf.name}")
        with zipfile.ZipFile(zf) as z:
            z.extractall(DATA_DIR)
        zf.unlink()

    files = sorted(DATA_DIR.iterdir())
    print(f"\ndata/raw/ içeriği ({len(files)} dosya):")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
