"""Adversarial validation: measures train/test distribution gap via LightGBM classifier."""

from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class AdversarialValidator:
    """Train/test dağılım farkı kontrolü (LightGBM classifier ile)."""

    @staticmethod
    def run(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        cat_features: list[str] | None = None,
        n_splits: int = 5,
        seed: int = 42,
    ) -> tuple[float, np.ndarray]:
        """
        Returns
        -------
        adv_auc : float
            Ortalama OOF AUC skoru.
        test_likeness : np.ndarray
            Her train örneğinin test'e benzerlik skoru (sample weighting için).
        """
        cat_features = cat_features or []

        adv_X = pd.concat(
            [train_df[features].assign(_src=0), test_df[features].assign(_src=1)],
            axis=0,
        ).reset_index(drop=True)
        adv_y = adv_X.pop("_src")

        for c in cat_features:
            if c in adv_X.columns:
                adv_X[c] = adv_X[c].astype("category")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        auc_scores: list[float] = []
        adv_oof = np.zeros(len(adv_X))

        for _fold, (tr, vl) in enumerate(skf.split(adv_X, adv_y)):
            m = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                verbose=-1,
            )
            m.fit(
                adv_X.iloc[tr], adv_y.iloc[tr],
                eval_set=[(adv_X.iloc[vl], adv_y.iloc[vl])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            adv_oof[vl] = m.predict_proba(adv_X.iloc[vl])[:, 1]
            auc_scores.append(roc_auc_score(adv_y.iloc[vl], adv_oof[vl]))

        adv_auc = float(np.mean(auc_scores))

        if adv_auc < 0.55:
            logger.success(
                f"Adversarial AUC: {adv_auc:.4f} — "
                f"Distribution shift YOK ✓"
            )
        elif adv_auc < 0.65:
            logger.warning(
                f"Adversarial AUC: {adv_auc:.4f} — "
                f"Hafif shift. Sample weight kullanılabilir."
            )
        elif adv_auc < 0.80:
            logger.warning(
                f"Adversarial AUC: {adv_auc:.4f} — "
                f"Belirgin shift! Sample weighting TAVSİYE."
            )
        else:
            logger.error(
                f"Adversarial AUC: {adv_auc:.4f} — "
                f"Ciddi shift! Feature mühendisliği gerekli."
            )

        m_full = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=seed, verbose=-1,
        )
        m_full.fit(adv_X, adv_y)
        shift_imp = (
            pd.Series(m_full.feature_importances_, index=adv_X.columns)
            .sort_values(ascending=False)
        )
        logger.info(f"En shifted 5 feature: {shift_imp.head(5).to_dict()}")

        test_likeness = adv_oof[: len(train_df)]

        del adv_X, adv_y, adv_oof, m_full
        gc.collect()

        return adv_auc, test_likeness
