"""Feature engineering pipeline — domain features, encodings, and transforms."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from src.config import settings


class FeatureEngineer:
    """Feature factory using Polars lazy expressions."""

    @staticmethod
    def _add_domain_features(lf: pl.LazyFrame) -> pl.LazyFrame:
        eps = 1e-6
        return lf.with_columns(
            (pl.col("derin_uyku_yuzdesi") + pl.col("rem_yuzdesi")).alias("sleep_efficiency"),

            (100 - pl.col("rem_yuzdesi") - pl.col("derin_uyku_yuzdesi"))
            .clip(lower_bound=0).alias("non_rem_estimate"),

            (pl.col("gecelik_uyanma_sayisi") / (pl.col("derin_uyku_yuzdesi") + 1))
            .alias("sleep_fragmentation"),

            (1.0 / (pl.col("uykuya_dalma_suresi_dk") + 1)).alias("sleep_onset_speed"),

            (pl.col("rem_yuzdesi") / (pl.col("derin_uyku_yuzdesi") + eps)).alias("rem_to_deep_ratio"),

            (
                (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi"))
                / (pl.col("uykuya_dalma_suresi_dk") + 5 * pl.col("gecelik_uyanma_sayisi") + 1)
            ).alias("sleep_quality_proxy"),

            pl.col("hafta_sonu_uyku_farki_saat").abs().alias("social_jetlag"),

            (pl.col("uyku_oncesi_kafein_mg") * pl.col("uyku_oncesi_ekran_suresi_dk") / 1000)
            .alias("stimulant_load"),

            (pl.col("stres_skoru") * pl.col("uyku_oncesi_kafein_mg").log1p()).alias("stress_caffeine"),

            (pl.col("sekerleme_suresi_dk") / (pl.col("stres_skoru") + 1)).alias("stress_recovery"),

            (
                pl.col("uyku_oncesi_kafein_mg") / 100
                + pl.col("uyku_oncesi_ekran_suresi_dk") / 60
                + pl.col("stres_skoru") / 10
            ).alias("pre_sleep_arousal"),

            (pl.col("gunluk_adim_sayisi") / (pl.col("dinlenik_nabiz_bpm") + 1)).alias("activity_recovery"),

            (pl.col("vucut_kitle_indeksi") * pl.col("dinlenik_nabiz_bpm") / 100).alias("metabolic_proxy"),

            (pl.col("uyku_oncesi_ekran_suresi_dk") / (pl.col("gunluk_calisma_saati") + 1))
            .alias("screen_per_work_hour"),

            (pl.col("yas") // 10 * 10).cast(pl.Int32).alias("age_decade"),

            pl.col("uyku_oncesi_kafein_mg").log1p().alias("log_kafein"),
            pl.col("uyku_oncesi_ekran_suresi_dk").log1p().alias("log_ekran"),
            pl.col("gunluk_adim_sayisi").log1p().alias("log_adim"),

            (
                (pl.col("rem_yuzdesi") * 0.6 + pl.col("derin_uyku_yuzdesi") * 0.4)
                / (pl.col("gecelik_uyanma_sayisi") + 1)
            ).alias("wakeful_efficiency"),

            (
                (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi"))
                * (10 - pl.col("stres_skoru")) / 10
            ).alias("stress_adj_sleep"),

            (
                pl.col("gun_tipi").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("kronotip").cast(pl.Utf8).fill_null("NA")
                + pl.lit("_")
                + pl.col("meslek").cast(pl.Utf8).fill_null("NA")
            ).alias("triple_day_chrono_job"),

            (pl.col("stres_skoru") ** 2).alias("stres_squared"),
            (pl.col("stres_skoru") ** 3).alias("stres_cubed"),
            (pl.col("stres_skoru").sqrt()).alias("stres_sqrt"),

            (pl.col("rem_yuzdesi") ** 2).alias("rem_squared"),
            (pl.col("gunluk_calisma_saati") ** 2).alias("work_squared"),
            (pl.col("derin_uyku_yuzdesi") ** 2).alias("deep_sleep_sq"),
            (pl.col("gecelik_uyanma_sayisi") ** 2).alias("wakeups_sq"),

            (pl.col("rem_yuzdesi") / (pl.col("stres_skoru") + 0.1)).alias("rem_per_stres"),
            (pl.col("derin_uyku_yuzdesi") / (pl.col("stres_skoru") + 0.1)).alias("deep_per_stres"),
            (pl.col("rem_yuzdesi") / (pl.col("gunluk_calisma_saati") + 0.1)).alias("rem_per_work"),
            (
                (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi"))
                / (pl.col("gecelik_uyanma_sayisi") + 1)
            ).alias("total_quality_sleep"),
            (pl.col("gunluk_adim_sayisi") / (pl.col("gunluk_calisma_saati") + 0.1)).alias("activity_per_work"),
            (
                (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi"))
                / (pl.col("stres_skoru") + 1.0)
            ).alias("sleep_stress_ratio"),
            (pl.col("uyku_oncesi_ekran_suresi_dk") * pl.col("stres_skoru")).alias("screen_stress_load"),

            (pl.col("stres_skoru") * pl.col("rem_yuzdesi")).alias("stres_x_rem"),
            (pl.col("stres_skoru") * pl.col("gunluk_calisma_saati")).alias("stres_x_work"),
            (pl.col("stres_skoru") * pl.col("gecelik_uyanma_sayisi")).alias("stres_x_wakeups"),
            (pl.col("rem_yuzdesi") * pl.col("gunluk_calisma_saati")).alias("rem_x_work"),
            (pl.col("rem_yuzdesi") * pl.col("gecelik_uyanma_sayisi")).alias("rem_x_wakeups"),
            (pl.col("derin_uyku_yuzdesi") * pl.col("gunluk_calisma_saati")).alias("deep_x_work"),

            (pl.col("stres_skoru") * 10).cast(pl.Int32).alias("stres_bin10"),
            (pl.col("rem_yuzdesi") * 5).cast(pl.Int32).alias("rem_bin5"),
            (pl.col("yas") // 5 * 5).alias("age_bin5"),

            (
                pl.col("meslek").cast(pl.Utf8).fill_null("NA")
                + pl.lit("_")
                + pl.col("ruh_sagligi_durumu").cast(pl.Utf8).fill_null("NA")
            ).alias("meslek_ruh_sagligi"),

            (
                pl.col("meslek").cast(pl.Utf8).fill_null("NA")
                + pl.lit("_")
                + pl.col("gun_tipi").cast(pl.Utf8)
            ).alias("meslek_gun_tipi"),

            (
                pl.col("ruh_sagligi_durumu").cast(pl.Utf8).fill_null("NA")
                + pl.lit("_")
                + pl.col("gun_tipi").cast(pl.Utf8)
            ).alias("ruh_sagligi_gun_tipi"),

            (
                pl.col("kronotip").cast(pl.Utf8).fill_null("NA")
                + pl.lit("_")
                + pl.col("gun_tipi").cast(pl.Utf8)
            ).alias("chronotype_daytype"),

            (100.0 - pl.col("rem_yuzdesi") - pl.col("derin_uyku_yuzdesi"))
            .clip(lower_bound=0).alias("light_sleep_pct"),

            (
                pl.col("stres_skoru")
                / (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi") + 0.1)
            ).alias("stres_per_total_sleep"),

            (pl.col("stres_skoru") / (pl.col("derin_uyku_yuzdesi") + 0.1)).alias("stres_per_deep"),

            (
                pl.col("rem_yuzdesi")
                / (pl.col("derin_uyku_yuzdesi") + pl.col("rem_yuzdesi") + 0.1)
            ).alias("rem_architecture_ratio"),

            (pl.col("stres_skoru") ** 0.5 * pl.col("rem_yuzdesi")).alias("sqrt_stres_x_rem"),
            (pl.col("stres_skoru").log1p() * pl.col("derin_uyku_yuzdesi")).alias("log_stres_x_deep"),

            (pl.col("oda_sicakligi_celsius") * pl.col("stres_skoru")).alias("temp_x_stres"),
            (pl.col("oda_sicakligi_celsius") * pl.col("gunluk_calisma_saati")).alias("temp_x_work"),

            (
                pl.col("rem_yuzdesi") * 0.35
                + pl.col("derin_uyku_yuzdesi") * 0.30
                - pl.col("gecelik_uyanma_sayisi") * 2.0
                - pl.col("uykuya_dalma_suresi_dk") * 0.15
                + pl.col("gunluk_adim_sayisi") / 5000
            ).alias("composite_sleep_quality"),

            (pl.col("stres_skoru") * pl.col("uykuya_dalma_suresi_dk")).alias("stres_x_latency"),
            (pl.col("rem_yuzdesi") * pl.col("derin_uyku_yuzdesi")).alias("rem_x_deep_synergy"),

            (
                (pl.col("uyku_oncesi_kafein_mg") / 100 + 1)
                * (pl.col("stres_skoru") / 5 + 1)
            ).alias("arousal_synergy"),
        )

    @staticmethod
    def _add_age_relative_features(df: pl.DataFrame) -> pl.DataFrame:
        train_only = df.filter(pl.col("is_train") == 1)
        rel_cols = [
            "stres_skoru", "gunluk_calisma_saati", "uyku_oncesi_kafein_mg",
            "gunluk_adim_sayisi", "uyku_oncesi_ekran_suresi_dk",
            "dinlenik_nabiz_bpm", "gecelik_uyanma_sayisi",
        ]
        for col in rel_cols:
            age_stats = (
                train_only.group_by("age_decade")
                .agg(
                    pl.col(col).mean().alias(f"{col}_age_mean"),
                    pl.col(col).std().alias(f"{col}_age_std"),
                )
                .with_columns(
                    pl.col(f"{col}_age_std").fill_null(1.0).replace(0.0, 1.0)
                )
            )
            df = df.join(age_stats, on="age_decade", how="left")
            df = df.with_columns(
                ((pl.col(col) - pl.col(f"{col}_age_mean")) / pl.col(f"{col}_age_std"))
                .alias(f"{col}_age_zscore")
            )
            df = df.drop(f"{col}_age_mean", f"{col}_age_std")
        return df

    @staticmethod
    def _add_group_stats(df: pl.DataFrame) -> pl.DataFrame:
        group_cols = ["meslek", "ulke", "kronotip"]
        target_proxies = ["stres_skoru", "vucut_kitle_indeksi", "dinlenik_nabiz_bpm"]
        train_only = df.filter(pl.col("is_train") == 1)
        for g in group_cols:
            for t in target_proxies:
                stats = train_only.group_by(g).agg([
                    pl.col(t).mean().alias(f"feat_avg_{t}_per_{g}"),
                    pl.col(t).std().alias(f"feat_std_{t}_per_{g}"),
                ])
                df = df.join(stats, on=g, how="left")
                df = df.with_columns(
                    (pl.col(t) - pl.col(f"feat_avg_{t}_per_{g}")).alias(f"feat_diff_avg_{t}_per_{g}")
                )
        return df

    @staticmethod
    def _add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col("rem_yuzdesi") * pl.col("derin_uyku_yuzdesi")).alias("feat_deep_sleep_synergy"),
            (pl.col("gunluk_adim_sayisi") / (pl.col("vucut_kitle_indeksi") + 1)).alias("feat_activity_bmi_ratio"),
            (pl.col("stres_skoru") * pl.col("oda_sicakligi_celsius")).alias("feat_temp_stress_impact"),
            (pl.col("uyku_oncesi_kafein_mg") / (pl.col("yas") + 1)).alias("feat_caffeine_age_sensitivity"),
            (pl.col("gunluk_calisma_saati") - (pl.col("derin_uyku_yuzdesi") / 10)).alias("feat_work_sleep_debt"),
            (pl.col("sekerleme_suresi_dk") * pl.col("stres_skoru")).alias("feat_nap_stress_interaction"),
            (
                (pl.col("gunluk_adim_sayisi") * (pl.col("rem_yuzdesi") + pl.col("derin_uyku_yuzdesi")))
                / (pl.col("stres_skoru") + 1.0)
            ).alias("feat_health_balance_index"),
        )

    @staticmethod
    def calculate_sample_weights(y: np.ndarray) -> np.ndarray:
        z = np.abs((y - np.mean(y)) / (np.std(y) + 1e-9))
        weights = 1.0 / (1.0 + np.maximum(0, z - 2.0))
        result: np.ndarray = weights / np.mean(weights)
        return result

    @staticmethod
    def _add_frequency_encoding(df: pl.DataFrame) -> pl.DataFrame:
        freq_cols = ["meslek", "ruh_sagligi_durumu", "kronotip"]
        train_only = df.filter(pl.col("is_train") == 1)
        for col in freq_cols:
            if col in df.columns:
                stats = train_only.group_by(col).agg(pl.len().alias(f"freq_{col}"))
                df = df.join(stats, on=col, how="left")
                df = df.with_columns(pl.col(f"freq_{col}").fill_null(0))
        return df

    @staticmethod
    def _encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
        cat_cols = (
            settings.CAT_FEATURES
            + ["chronotype_daytype", "meslek_ruh_sagligi", "meslek_gun_tipi",
               "ruh_sagligi_gun_tipi", "triple_day_chrono_job"]
        )
        for col in cat_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).fill_null("NA")
                    .cast(pl.Categorical).to_physical().alias(col)
                )
        return df

    @staticmethod
    def _add_smoothed_target_encoding(df: pl.DataFrame) -> pl.DataFrame:
        """Smoothed target encoding — computed from train only, applied to both splits."""
        target_col = settings.TARGET_COL
        if target_col not in df.columns:
            return df

        train_only = df.filter(pl.col("is_train") == 1)
        global_mean: float = train_only.get_column(target_col).drop_nulls().cast(pl.Float64).mean() or 0.0  # type: ignore[assignment]
        weight = 10

        encoding_configs = [
            (["meslek", "ruh_sagligi_durumu"], "meslek_ruh_te"),
            (["meslek", "gun_tipi"], "meslek_guntipi_te"),
            (["ruh_sagligi_durumu", "gun_tipi"], "ruh_guntipi_te"),
            (["meslek", "ruh_sagligi_durumu", "gun_tipi", "kronotip"], "all_cat_te"),
            (["gun_tipi", "kronotip", "meslek"], "triple_day_chrono_job_te"),
        ]

        for cat_cols, alias in encoding_configs:
            combo_expr = pl.col(cat_cols[0]).cast(pl.Utf8).fill_null("NA")
            for c in cat_cols[1:]:
                combo_expr = combo_expr + pl.lit("_") + pl.col(c).cast(pl.Utf8).fill_null("NA")
            df = df.with_columns(combo_expr.alias("_tmp_combo"))
            train_combo = df.filter(pl.col("is_train") == 1)
            group_stats = (
                train_combo.group_by("_tmp_combo")
                .agg(
                    pl.col(target_col).mean().alias("_grp_mean"),
                    pl.col(target_col).count().alias("_grp_count"),
                )
                .with_columns(
                    (
                        (pl.col("_grp_count") * pl.col("_grp_mean") + weight * global_mean)
                        / (pl.col("_grp_count") + weight)
                    ).alias(alias)
                )
                .select("_tmp_combo", alias)
            )
            df = df.join(group_stats, on="_tmp_combo", how="left")
            df = df.with_columns(pl.col(alias).fill_null(global_mean))
            df = df.drop("_tmp_combo")
        return df

    @staticmethod
    def _add_quant_features(df: pl.DataFrame) -> pl.DataFrame:
        from sklearn.decomposition import PCA, FastICA
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        quant_cols = [
            "stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi",
            "gunluk_calisma_saati", "gecelik_uyanma_sayisi",
            "uykuya_dalma_suresi_dk", "gunluk_adim_sayisi",
            "uyku_oncesi_kafein_mg", "yas", "oda_sicakligi_celsius",
        ]
        pdf = df.to_pandas()
        train_median = pdf.loc[pdf["is_train"] == 1, quant_cols].median()
        X = pdf[quant_cols].fillna(train_median)
        train_mask = pdf["is_train"] == 1
        X_train = X[train_mask].values
        X_all = X.values

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_sc = scaler.transform(X_train)
        X_all_sc = scaler.transform(X_all)

        pca = PCA(n_components=5, random_state=42)
        pca.fit(X_train_sc)
        pca_all = pca.transform(X_all_sc)
        for i in range(5):
            df = df.with_columns(pl.Series(f"pca_{i+1}", pca_all[:, i]))

        ica = FastICA(n_components=5, random_state=42, max_iter=500)
        ica.fit(X_train_sc)
        ica_all = ica.transform(X_all_sc)
        for i in range(5):
            df = df.with_columns(pl.Series(f"ica_{i+1}", ica_all[:, i]))

        for n_comp in [3, 5, 7]:
            gmm = GaussianMixture(n_components=n_comp, random_state=42, covariance_type="diag", max_iter=200)
            gmm.fit(X_train_sc)
            probs = gmm.predict_proba(X_all_sc)
            for i in range(n_comp):
                df = df.with_columns(pl.Series(f"gmm{n_comp}_prob_{i}", probs[:, i]))

        train_only_df = df.filter(pl.col("is_train") == 1)
        for cat_col in ["gun_tipi", "ruh_sagligi_durumu"]:
            for num_col in ["stres_skoru", "rem_yuzdesi"]:
                stats = train_only_df.group_by(cat_col).agg([
                    pl.col(num_col).mean().alias(f"_mean_{num_col}_{cat_col}"),
                    pl.col(num_col).std().alias(f"_std_{num_col}_{cat_col}"),
                ])
                df = df.join(stats, on=cat_col, how="left")
                df = df.with_columns(
                    pl.col(f"_std_{num_col}_{cat_col}").fill_null(1.0).clip(lower_bound=0.01)
                )
                df = df.with_columns(
                    ((pl.col(num_col) - pl.col(f"_mean_{num_col}_{cat_col}")) / pl.col(f"_std_{num_col}_{cat_col}"))
                    .alias(f"crossz_{cat_col}_{num_col}")
                )
                df = df.drop([f"_mean_{num_col}_{cat_col}", f"_std_{num_col}_{cat_col}"])

        w = np.array([-0.98, 0.75, 0.41, -0.20, -0.36])
        mark_cols = [
            "stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi",
            "gunluk_calisma_saati", "gecelik_uyanma_sayisi",
        ]
        mark_vals = pdf[mark_cols].fillna(0).values
        mark_sc = StandardScaler().fit(mark_vals[train_mask]).transform(mark_vals)
        df = df.with_columns(pl.Series("markowitz_score", mark_sc @ w))
        return df

    @classmethod
    def fit_transform(cls, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Feature engineering started. Initial columns: {len(df.columns)}")
        df = cls._add_domain_features(df.lazy()).collect()
        df = cls._add_age_relative_features(df)
        df = cls._add_group_stats(df)
        df = cls._add_interaction_features(df)
        df = cls._add_frequency_encoding(df)
        df = cls._add_smoothed_target_encoding(df)
        df = cls._add_quant_features(df)
        df = cls._encode_categoricals(df)
        logger.success(f"Feature engineering done. Final columns: {len(df.columns)}")
        return df
