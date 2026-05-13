"""
FINAL PUSH — CB + LGBM DART blend, zero leakage
Hedef: 1.20356 altı
"""
import time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

t0 = time.time()
def log(msg): print(f"[{int(time.time()-t0):4d}s] {msg}", flush=True)

TARGET = "bilissel_performans_skoru"
train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test_x.csv")
y     = train[TARGET].values

# ── Features ──────────────────────────────────────────────────────────
NUM = ["stres_skoru","rem_yuzdesi","derin_uyku_yuzdesi","gunluk_calisma_saati",
       "gecelik_uyanma_sayisi","gunluk_adim_sayisi","uyku_oncesi_kafein_mg",
       "yas","sekerleme_suresi_dk","dinlenik_nabiz_bpm","uykuya_dalma_suresi_dk",
       "oda_sicakligi_celsius","vucut_kitle_indeksi","hafta_sonu_uyku_farki_saat",
       "uyku_oncesi_ekran_suresi_dk"]
CAT = ["meslek","ruh_sagligi_durumu","gun_tipi","kronotip","mevsim","cinsiyet","ulke"]
med = train[NUM].median()
eps = 0.01

def fe(df):
    d = df[NUM + CAT].copy()
    for c in NUM:
        d[c] = d[c].fillna(med[c])
    for c in CAT:
        d[c] = d[c].fillna("Unknown").astype(str)

    S  = d["stres_skoru"].values
    R  = d["rem_yuzdesi"].values
    D  = d["derin_uyku_yuzdesi"].values
    C  = d["gunluk_calisma_saati"].values
    U  = d["gecelik_uyanma_sayisi"].values
    A  = d["gunluk_adim_sayisi"].values / 1000
    KAF= d["uyku_oncesi_kafein_mg"].values
    SEK= d["sekerleme_suresi_dk"].values
    YAS= d["yas"].values
    NAB= d["dinlenik_nabiz_bpm"].values
    LAT= d["uykuya_dalma_suresi_dk"].values
    R_ = np.clip(R, eps, None)
    D_ = np.clip(D, eps, None)
    S_ = np.clip(S, eps, None)
    RD = R + D
    RD_= np.clip(RD, eps, None)
    iy = 1 / np.clip(YAS, 0.1, None)
    s15= S**1.5
    s3 = s15**2 * 0.16159**2
    R2 = R_ + 1.6081

    d["fe_adim_C_D"]    = (A + C) / D_
    d["fe_log_rd"]      = np.log(RD_)
    d["fe_log_s"]       = np.log(S_)
    d["fe_log_ratio"]   = np.log(RD_) - np.log(S_)
    d["fe_s3_R"]        = s3 / R2
    d["fe_U_R"]         = U  / R2
    d["fe_knafs_iy_R"]  = (NAB + KAF*0.74 + SEK) * iy / R2
    d["fe_formula"]     = (
        d["fe_adim_C_D"] + np.log(RD_)*4.4416 - 16.231
        - (U*2.7337 + s3*2.7337 + (NAB+KAF*0.74+SEK)*iy) / R2
    )
    d["fe_rdivs"]       = R_ / S_
    d["fe_RDdivs"]      = RD / S_
    d["fe_sqrt_u"]      = np.sqrt(np.clip(U, 0, None))
    d["fe_sqrt_lat"]    = np.sqrt(np.clip(LAT, 0, None))
    d["fe_s15_D"]       = s15 / D_
    d["fe_iy"]          = iy
    d["fe_s3"]          = s3
    d["fe_s15"]         = s15
    d["fe_RD"]          = RD
    d["fe_U_S"]         = U * S
    d["fe_C_S"]         = C * S
    d["fe_sleep_eff"]   = RD
    d["fe_non_rem"]     = np.clip(100 - RD, 0, None)
    d["fe_frag"]        = U / (D_ + 1)
    d["fe_stress_adj"]  = RD * (10 - S) / 10
    d["fe_stres2"]      = S**2
    d["fe_stres3"]      = S**3
    d["fe_stres_sqrt"]  = np.sqrt(S_)
    d["fe_rem2"]        = R**2
    d["fe_work2"]       = C**2
    d["fe_stim"]        = KAF * LAT / 1000
    d["fe_act_rec"]     = A*1000 / (NAB + 1)
    d["fe_met"]         = d["vucut_kitle_indeksi"].values * NAB / 100
    d["fe_screen_s"]    = d["uyku_oncesi_ekran_suresi_dk"].values * S
    d["fe_age_kaf"]     = KAF / (YAS + 1)
    d["fe_work_deep"]   = C - D_ / 10
    d["fe_nap_stress"]  = SEK * S

    # Oda sıcaklığı
    ODA = d["oda_sicakligi_celsius"].values
    d["fe_temp_s"]      = ODA * S
    d["fe_temp_work"]   = ODA * C

    # HFW
    HFW = d["hafta_sonu_uyku_farki_saat"].values
    d["fe_hfw_abs"]     = np.abs(HFW)

    return d

log("Feature engineering...")
tr = fe(train)
te = fe(test)

feat_cols = [c for c in tr.columns if c not in [TARGET, "id"] + list(train.columns[:1])]
feat_cols = [c for c in tr.columns if c != TARGET and c != "id" and c in te.columns]
cat_idx_names = CAT

X_tr = tr[feat_cols]
X_te = te[feat_cols]

# Stratified folds (bin target)
y_bins = pd.cut(y, bins=10, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X_tr, y_bins))
log(f"Features: {len(feat_cols)}  Folds: 5")

# ── MODEL 1: CatBoost ─────────────────────────────────────────────────
log("\n[CB] depth=8, lr=0.04, 4000 iter...")
from catboost import CatBoostRegressor, Pool

cat_idx = [X_tr.columns.get_loc(c) for c in CAT if c in X_tr.columns]
cb_params = dict(
    iterations=4000, learning_rate=0.04, depth=8,
    l2_leaf_reg=4, loss_function="RMSE", eval_metric="RMSE",
    random_seed=42, verbose=False, early_stopping_rounds=250,
)

oof_cb = np.zeros(len(y))
preds_cb = []
for fold, (tr_i, vl_i) in enumerate(folds):
    pool_tr = Pool(X_tr.iloc[tr_i], y[tr_i], cat_features=cat_idx)
    pool_vl = Pool(X_tr.iloc[vl_i], y[vl_i], cat_features=cat_idx)
    m = CatBoostRegressor(**cb_params)
    m.fit(pool_tr, eval_set=pool_vl)
    oof_cb[vl_i] = np.clip(m.predict(pool_vl), 0, 10)
    preds_cb.append(np.clip(m.predict(Pool(X_te, cat_features=cat_idx)), 0, 10))
    log(f"  CB Fold {fold+1}: {np.sqrt(mean_squared_error(y[vl_i], oof_cb[vl_i])):.5f}  "
        f"(best_iter={m.get_best_iteration()})")

cv_cb = np.sqrt(mean_squared_error(y, oof_cb))
pred_cb = np.mean(preds_cb, axis=0)
log(f"  CB CV RMSE: {cv_cb:.5f}")
np.save("outputs/oof_fp_cb.npy", oof_cb)
np.save("outputs/test_fp_cb.npy", pred_cb)

# ── MODEL 2: LightGBM DART ────────────────────────────────────────────
log("\n[LGB DART] n_est=3000, lr=0.03...")
import lightgbm as lgb

# Kategorikleri int encode et (LGBM için)
X_tr_lgb = X_tr.copy()
X_te_lgb = X_te.copy()
cat_enc = {}
for c in CAT:
    if c in X_tr_lgb.columns:
        cats = X_tr_lgb[c].astype("category")
        X_tr_lgb[c] = cats.cat.codes
        cat_enc[c] = dict(enumerate(cats.cat.categories))
        X_te_lgb[c] = X_te_lgb[c].map({v: k for k, v in cat_enc[c].items()}).fillna(-1).astype(int)

lgb_params = dict(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    boosting_type="dart",
    drop_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

oof_lgb = np.zeros(len(y))
preds_lgb = []
for fold, (tr_i, vl_i) in enumerate(folds):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(
        X_tr_lgb.iloc[tr_i], y[tr_i],
        eval_set=[(X_tr_lgb.iloc[vl_i], y[vl_i])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[vl_i] = np.clip(m.predict(X_tr_lgb.iloc[vl_i]), 0, 10)
    preds_lgb.append(np.clip(m.predict(X_te_lgb), 0, 10))
    log(f"  LGB Fold {fold+1}: {np.sqrt(mean_squared_error(y[vl_i], oof_lgb[vl_i])):.5f}  "
        f"(best_iter={m.best_iteration_})")

cv_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
pred_lgb = np.mean(preds_lgb, axis=0)
log(f"  LGB CV RMSE: {cv_lgb:.5f}")
np.save("outputs/oof_fp_lgb.npy", oof_lgb)
np.save("outputs/test_fp_lgb.npy", pred_lgb)

# ── MODEL 3: XGBoost ──────────────────────────────────────────────────
log("\n[XGB] n_est=2000, lr=0.05...")
import xgboost as xgb

X_tr_xgb = X_tr_lgb.copy()
X_te_xgb = X_te_lgb.copy()

xgb_params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

oof_xgb = np.zeros(len(y))
preds_xgb = []
for fold, (tr_i, vl_i) in enumerate(folds):
    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=150, eval_metric="rmse")
    m.fit(
        X_tr_xgb.iloc[tr_i], y[tr_i],
        eval_set=[(X_tr_xgb.iloc[vl_i], y[vl_i])],
        verbose=False,
    )
    oof_xgb[vl_i] = np.clip(m.predict(X_tr_xgb.iloc[vl_i]), 0, 10)
    preds_xgb.append(np.clip(m.predict(X_te_xgb), 0, 10))
    log(f"  XGB Fold {fold+1}: {np.sqrt(mean_squared_error(y[vl_i], oof_xgb[vl_i])):.5f}  "
        f"(best_iter={m.best_iteration})")

cv_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
pred_xgb = np.mean(preds_xgb, axis=0)
log(f"  XGB CV RMSE: {cv_xgb:.5f}")
np.save("outputs/oof_fp_xgb.npy", oof_xgb)
np.save("outputs/test_fp_xgb.npy", pred_xgb)

# ── Optimal Blend ─────────────────────────────────────────────────────
log("\n[BLEND] Optuna ile optimal ağırlık arama...")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

oofs  = {"cb": oof_cb,  "lgb": oof_lgb,  "xgb": oof_xgb}
preds = {"cb": pred_cb, "lgb": pred_lgb, "xgb": pred_xgb}

def objective(trial):
    w = {k: trial.suggest_float(k, 0, 1) for k in oofs}
    total = sum(w.values())
    if total < 1e-6: return 99.0
    blend = sum(w[k] * oofs[k] for k in oofs) / total
    return float(np.sqrt(mean_squared_error(y, np.clip(blend, 0, 10))))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500, show_progress_bar=False)

best = study.best_params
total = sum(best.values())
blend_oof  = sum(best[k] * oofs[k]  for k in oofs) / total
blend_pred = sum(best[k] * preds[k] for k in oofs) / total
cv_blend = np.sqrt(mean_squared_error(y, np.clip(blend_oof, 0, 10)))

log(f"\n{'='*60}")
log(f"Sonuçlar:")
log(f"  CB   CV: {cv_cb:.5f}  (w={best['cb']/total:.3f})")
log(f"  LGB  CV: {cv_lgb:.5f}  (w={best['lgb']/total:.3f})")
log(f"  XGB  CV: {cv_xgb:.5f}  (w={best['xgb']/total:.3f})")
log(f"  Blend CV: {cv_blend:.5f}")
log(f"  Hedef:    1.20356 ({'✓ GEÇTIK' if cv_blend < 1.20356 else '✗ geçemedik'})")

blend_pred_clip = np.clip(blend_pred, 0, 10)
pd.DataFrame({"id": test["id"], TARGET: blend_pred_clip}).to_csv(
    "submission_final_push.csv", index=False)
log(f"\nsubmission_final_push.csv  mean={blend_pred_clip.mean():.4f}")
log(f"Toplam süre: {int(time.time()-t0)}s")

# Bireysel modelleri de kaydet
for name, p in [("cb", pred_cb), ("lgb", pred_lgb), ("xgb", pred_xgb)]:
    pd.DataFrame({"id": test["id"], TARGET: np.clip(p, 0, 10)}).to_csv(
        f"submission_fp_{name}.csv", index=False)
    log(f"submission_fp_{name}.csv")
