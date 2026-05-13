"""EDA — Exploratory Data Analysis modülü.

Datathon jüri kriterlerinden biri olan EDA görselleştirmelerini üretir.
Tüm grafikler `outputs/eda/` dizinine kaydedilir.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger

from src.config import settings

matplotlib.use("Agg")  # Headless rendering


class EDAReport:
    """Veri seti keşif ve görselleştirme raporu."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or (settings.PROJECT_DIR / "outputs" / "eda")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: matplotlib.figure.Figure, name: str) -> None:
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"  📊 {path}")

    def plot_target_distribution(self, y: np.ndarray) -> None:
        """Target değişkenin histogram + KDE dağılımı."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(y, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].axvline(np.mean(y), color="red", ls="--", label=f"mean={np.mean(y):.2f}")
        axes[0].axvline(np.median(y), color="orange", ls="--", label=f"median={np.median(y):.2f}")
        axes[0].set_title("Target Dağılımı", fontsize=14, fontweight="bold")
        axes[0].set_xlabel(settings.TARGET_COL)
        axes[0].legend()

        # Box plot
        bp = axes[1].boxplot(y, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("#4C72B0")
        axes[1].set_title("Target Box Plot", fontsize=14, fontweight="bold")
        axes[1].set_ylabel(settings.TARGET_COL)

        p1, p99 = np.percentile(y, [1, 99])
        axes[1].axhline(p1, color="red", ls=":", alpha=0.7, label=f"p1={p1:.2f}")
        axes[1].axhline(p99, color="red", ls=":", alpha=0.7, label=f"p99={p99:.2f}")
        axes[1].legend()

        fig.suptitle("Hedef Değişken Analizi", fontsize=16, fontweight="bold", y=1.02)
        self._save(fig, "01_target_distribution")

    def plot_missing_values(self, df: pl.DataFrame) -> None:
        """Eksik veri oranları bar chart."""
        null_pcts = []
        for col in df.columns:
            pct = 100 * df[col].null_count() / len(df)
            if pct > 0:
                null_pcts.append((col, pct))

        if not null_pcts:
            logger.info("  Eksik veri yok — grafik atlanıyor")
            return

        null_pcts.sort(key=lambda x: -x[1])
        names, pcts = zip(*null_pcts, strict=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
        bars = ax.barh(range(len(names)), pcts, color="#E74C3C", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Eksik Oran (%)")
        ax.set_title("Eksik Veri Haritası", fontsize=14, fontweight="bold")
        for bar, pct in zip(bars, pcts, strict=True):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=10)
        ax.invert_yaxis()
        self._save(fig, "02_missing_values")

    def plot_correlation_heatmap(self, df: pl.DataFrame) -> None:
        """Sayısal değişkenlerin korelasyon matrisi."""
        num_cols = [
            c for c in df.columns
            if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and c not in ["id", "is_train"]
        ]
        if len(num_cols) < 3:
            return

        if settings.TARGET_COL in num_cols:
            # Target ile korelasyona göre sırala
            corrs = []
            for c in num_cols:
                if c != settings.TARGET_COL:
                    r = float(df.select(pl.corr(c, settings.TARGET_COL)).item())
                    corrs.append((c, abs(r)))
            corrs.sort(key=lambda x: -x[1])
            top_cols = [c for c, _ in corrs[:19]] + [settings.TARGET_COL]
        else:
            top_cols = num_cols[:20]

        corr_df = df.select(top_cols).to_pandas().corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        sns.heatmap(
            corr_df, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, square=True, linewidths=0.5,
            annot_kws={"size": 7},
        )
        ax.set_title(
            "Korelasyon Haritası (Top 20 Feature)",
            fontsize=14, fontweight="bold",
        )
        self._save(fig, "03_correlation_heatmap")

    def plot_feature_vs_target(self, df: pl.DataFrame) -> None:
        """En güçlü 6 sayısal feature'ın target ile ilişkisi."""
        num_cols = [
            c for c in df.columns
            if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and c not in ["id", "is_train", settings.TARGET_COL]
        ]
        if settings.TARGET_COL not in df.columns:
            return

        corrs = []
        for c in num_cols:
            r = float(df.select(pl.corr(c, settings.TARGET_COL)).item())
            corrs.append((c, abs(r), r))
        corrs.sort(key=lambda x: -x[1])
        top6 = corrs[:6]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        pdf = df.select([settings.TARGET_COL] + [c for c, _, _ in top6]).to_pandas()

        for idx, (col, _abs_r, r) in enumerate(top6):
            ax = axes[idx // 3][idx % 3]
            ax.scatter(
                pdf[col], pdf[settings.TARGET_COL],
                alpha=0.05, s=3, color="#4C72B0",
            )
            # Trend line
            mask = pdf[col].notna() & pdf[settings.TARGET_COL].notna()
            if mask.sum() > 2:
                z = np.polyfit(pdf.loc[mask, col], pdf.loc[mask, settings.TARGET_COL], 1)
                p = np.poly1d(z)
                x_range = np.linspace(pdf[col].min(), pdf[col].max(), 100)
                ax.plot(x_range, p(x_range), "r--", linewidth=2)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel(settings.TARGET_COL, fontsize=10)
            ax.set_title(f"{col}\nr={r:+.3f}", fontsize=11, fontweight="bold")

        fig.suptitle(
            "En Güçlü 6 Feature → Target İlişkisi",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        self._save(fig, "04_feature_vs_target")

    def plot_categorical_vs_target(self, df: pl.DataFrame) -> None:
        """Güçlü kategorik değişkenlerin target dağılımı (violin)."""
        cats = ["meslek", "ruh_sagligi_durumu", "gun_tipi", "kronotip"]
        cats = [c for c in cats if c in df.columns and settings.TARGET_COL in df.columns]

        if not cats:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        pdf = df.select(cats + [settings.TARGET_COL]).to_pandas()

        for idx, col in enumerate(cats):
            ax = axes[idx // 2][idx % 2]
            order = (
                pdf.groupby(col)[settings.TARGET_COL]
                .mean()
                .sort_values()
                .index.tolist()
            )
            sns.violinplot(
                data=pdf, x=col, y=settings.TARGET_COL,
                order=order, ax=ax, palette="viridis", inner="box",
            )
            ax.set_title(f"{col} → Target", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

        fig.suptitle(
            "Kategorik Değişkenler → Target Dağılımı",
            fontsize=16, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        self._save(fig, "05_categorical_vs_target")

    def plot_outlier_analysis(self, y: np.ndarray) -> None:
        """Z-score ve IQR tabanlı outlier tespiti."""
        z_scores = np.abs((y - np.mean(y)) / (np.std(y) + 1e-9))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Z-score dağılımı
        axes[0].hist(z_scores, bins=50, color="#2ECC71", edgecolor="white", alpha=0.8)
        axes[0].axvline(2, color="orange", ls="--", label="z=2 threshold")
        axes[0].axvline(3, color="red", ls="--", label="z=3 threshold")
        n_z2 = int((z_scores > 2).sum())
        n_z3 = int((z_scores > 3).sum())
        axes[0].set_title(
            f"Z-Score Dağılımı\n|z|>2: {n_z2} ({100 * n_z2 / len(y):.1f}%)  |z|>3: {n_z3}",
            fontsize=12, fontweight="bold",
        )
        axes[0].legend()

        # Winsorization etkisi
        p1, p99 = np.percentile(y, settings.WINSOR_LIMITS)
        y_winsor = np.clip(y, p1, p99)
        axes[1].hist(y, bins=50, alpha=0.5, color="blue", label="Original")
        axes[1].hist(y_winsor, bins=50, alpha=0.5, color="green", label="Winsorized")
        axes[1].set_title(
            f"Winsorization (p{settings.WINSOR_LIMITS[0]:.0f}"
            f"-p{settings.WINSOR_LIMITS[1]:.0f})",
            fontsize=12, fontweight="bold",
        )
        axes[1].legend()

        fig.suptitle("Aykırı Değer Analizi", fontsize=16, fontweight="bold", y=1.02)
        self._save(fig, "06_outlier_analysis")

    def run_full_eda(self, df: pl.DataFrame) -> None:
        """Tüm EDA grafiklerini sırasıyla üretir."""
        logger.info("EDA raporu üretiliyor...")

        train_df = df.filter(pl.col("is_train") == 1)
        y = train_df.get_column(settings.TARGET_COL).to_numpy()

        self.plot_target_distribution(y)
        self.plot_missing_values(train_df)
        self.plot_correlation_heatmap(train_df)
        self.plot_feature_vs_target(train_df)
        self.plot_categorical_vs_target(train_df)
        self.plot_outlier_analysis(y)

        logger.success(
            f"EDA tamamlandı — {len(list(self.output_dir.glob('*.png')))} "
            f"grafik → {self.output_dir}"
        )
