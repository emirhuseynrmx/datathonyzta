"""Generate EDA, SHAP, and EBM visuals → assets/"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("assets")
OUT.mkdir(exist_ok=True)

TARGET = "bilissel_performans_skoru"
train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test_x.csv")
y = train[TARGET].values

NUM = ["stres_skoru","rem_yuzdesi","derin_uyku_yuzdesi","gunluk_calisma_saati",
       "gecelik_uyanma_sayisi","gunluk_adim_sayisi","uyku_oncesi_kafein_mg",
       "yas","sekerleme_suresi_dk","dinlenik_nabiz_bpm","uykuya_dalma_suresi_dk",
       "oda_sicakligi_celsius","vucut_kitle_indeksi","hafta_sonu_uyku_farki_saat",
       "uyku_oncesi_ekran_suresi_dk"]
CAT = ["meslek","ruh_sagligi_durumu","gun_tipi","kronotip","mevsim","cinsiyet","ulke"]

print("1/6  EDA plots...")

# ── 1. Target distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(y.mean(), color="red", ls="--", lw=1.5, label=f"mean={y.mean():.2f}")
axes[0].axvline(np.median(y), color="orange", ls="--", lw=1.5, label=f"median={np.median(y):.2f}")
axes[0].set_title("Hedef Dağılımı", fontsize=14, fontweight="bold")
axes[0].set_xlabel(TARGET); axes[0].legend()
bp = axes[1].boxplot(y, vert=True, patch_artist=True)
bp["boxes"][0].set_facecolor("#4C72B0")
p1,p99 = np.percentile(y,[1,99])
axes[1].axhline(p1, color="red", ls=":", label=f"p1={p1:.2f}")
axes[1].axhline(p99, color="red", ls="-.", label=f"p99={p99:.2f}")
axes[1].set_title("Box Plot", fontsize=14, fontweight="bold")
axes[1].legend()
fig.suptitle("Bilişsel Performans Skoru — Dağılım Analizi", fontsize=15, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"eda_target.png", dpi=150, bbox_inches="tight"); plt.close()

# ── 2. Correlation heatmap ────────────────────────────────────────────────────
corr_cols = NUM + [TARGET]
corr_df = train[corr_cols].corr()
target_corr = corr_df[TARGET].drop(TARGET).abs().sort_values(ascending=False)
top15 = target_corr.head(15).index.tolist()
sub = train[top15 + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
sns.heatmap(sub, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":8})
ax.set_title("Korelasyon Haritası (Top 15 Feature + Target)", fontsize=14, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"eda_correlation.png", dpi=150, bbox_inches="tight"); plt.close()

# ── 3. Feature vs target (top 6) ─────────────────────────────────────────────
top6 = target_corr.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, col in enumerate(top6):
    ax = axes[idx//3][idx%3]
    r = float(train[[col, TARGET]].corr().iloc[0, 1])
    ax.scatter(train[col], train[TARGET], alpha=0.04, s=3, color="#4C72B0")
    m = train[[col, TARGET]].dropna()
    z = np.polyfit(m[col], m[TARGET], 1)
    xr = np.linspace(m[col].min(), m[col].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), "r--", lw=2)
    ax.set_xlabel(col, fontsize=10); ax.set_ylabel(TARGET, fontsize=9)
    ax.set_title(f"{col}  r={r:+.3f}", fontsize=11, fontweight="bold")
fig.suptitle("En Güçlü 6 Feature → Target İlişkisi", fontsize=15, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"eda_feature_vs_target.png", dpi=150, bbox_inches="tight"); plt.close()

# ── 4. Categorical violin ─────────────────────────────────────────────────────
cats4 = ["ruh_sagligi_durumu","kronotip","meslek","gun_tipi"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, col in enumerate(cats4):
    ax = axes[idx//2][idx%2]
    order = train.groupby(col)[TARGET].mean().sort_values().index.tolist()
    sns.violinplot(data=train, x=col, y=TARGET, order=order,
                   ax=ax, palette="viridis", inner="box")
    ax.set_title(f"{col} → {TARGET}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Kategorik Değişkenler → Target Dağılımı", fontsize=15, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"eda_categorical.png", dpi=150, bbox_inches="tight"); plt.close()

# ── 5. Missing value & outlier ────────────────────────────────────────────────
null_pcts = {c: 100*train[c].isnull().mean() for c in train.columns if train[c].isnull().any()}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if null_pcts:
    names = list(null_pcts.keys()); pcts = list(null_pcts.values())
    axes[0].barh(names, pcts, color="#E74C3C", alpha=0.8)
    axes[0].set_title("Eksik Veri Oranları", fontsize=13, fontweight="bold")
else:
    axes[0].text(0.5, 0.5, "Eksik veri yok", ha="center", va="center", fontsize=14)
    axes[0].set_title("Eksik Veri", fontsize=13, fontweight="bold")
z = np.abs((y - y.mean()) / y.std())
axes[1].hist(z, bins=50, color="#2ECC71", edgecolor="white", alpha=0.8)
axes[1].axvline(3, color="red", ls="--", label=f"|z|>3: {(z>3).sum()} örnek")
axes[1].set_title("Z-Score Dağılımı (Outlier Analizi)", fontsize=13, fontweight="bold")
axes[1].legend()
fig.suptitle("Veri Kalitesi", fontsize=15, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"eda_data_quality.png", dpi=150, bbox_inches="tight"); plt.close()

print("2/6  Training quick CatBoost for SHAP + EBM...")

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

X = train[NUM + CAT].copy()
for c in NUM: X[c] = X[c].fillna(X[c].median())
for c in CAT: X[c] = X[c].fillna("Unknown").astype(str)

# Ordinal encode for sklearn-compatible models
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc = X.copy()
X_enc[CAT] = enc.fit_transform(X[CAT])
X_enc = X_enc.astype(np.float32)

X_tr, X_val, y_tr, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

# ── CatBoost (for SHAP) ───────────────────────────────────────────────────────
from catboost import CatBoostRegressor, Pool
cat_idx = [X.columns.get_loc(c) for c in CAT]
X_cb = train[NUM + CAT].copy()
for c in NUM: X_cb[c] = X_cb[c].fillna(X_cb[c].median())
for c in CAT: X_cb[c] = X_cb[c].fillna("Unknown").astype(str)

X_cb_tr, X_cb_val, y_tr2, y_val2 = train_test_split(X_cb, y, test_size=0.2, random_state=42)
cb = CatBoostRegressor(iterations=600, learning_rate=0.08, depth=7,
                       loss_function="RMSE", verbose=0, random_seed=42)
cb.fit(Pool(X_cb_tr, y_tr2, cat_features=cat_idx),
       eval_set=Pool(X_cb_val, y_val2, cat_features=cat_idx),
       use_best_model=True)
print("   CatBoost trained.")

# ── 6. Raw SHAP ───────────────────────────────────────────────────────────────
print("3/6  SHAP values...")
import shap

shap_vals = cb.get_feature_importance(Pool(X_cb_val, cat_features=cat_idx),
                                       type="ShapValues")[:, :-1]
feature_names = NUM + CAT

fig, ax = plt.subplots(figsize=(12, 9))
shap.summary_plot(shap_vals, X_cb_val.values if hasattr(X_cb_val, "values") else np.array(X_cb_val),
                  feature_names=feature_names, show=False, plot_type="dot",
                  max_display=20, alpha=0.4)
plt.title("SHAP Beeswarm — CatBoost (600 iter)", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout(); plt.savefig(OUT/"shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
print("   SHAP beeswarm saved.")

# ── SHAP bar (mean |SHAP|) ────────────────────────────────────────────────────
mean_abs = np.abs(shap_vals).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:20]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh([feature_names[i] for i in order[::-1]],
        mean_abs[order[::-1]], color="#4C72B0", alpha=0.85)
ax.set_xlabel("mean(|SHAP value|)", fontsize=12)
ax.set_title("Feature Importance — Mean |SHAP| (CatBoost)", fontsize=14, fontweight="bold")
plt.tight_layout(); fig.savefig(OUT/"shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# ── 7. EBM ────────────────────────────────────────────────────────────────────
print("4/6  EBM training...")
from interpret.glassbox import ExplainableBoostingRegressor

ebm = ExplainableBoostingRegressor(
    interactions=10, max_bins=256, learning_rate=0.01,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    feature_names=NUM + CAT,
)
ebm.fit(X_tr.values, y_tr)
print("   EBM trained.")

print("5/6  EBM global explanation plot...")
ebm_global = ebm.explain_global(name="EBM Global")

# Feature importance from EBM
scores = ebm_global.data()
feat_names_ebm = scores["names"]
feat_scores_ebm = scores["scores"]

order_ebm = np.argsort(feat_scores_ebm)[::-1][:20]
fig, ax = plt.subplots(figsize=(11, 8))
colors = ["#E74C3C" if feat_scores_ebm[i] > np.mean(feat_scores_ebm) else "#4C72B0"
          for i in order_ebm[::-1]]
ax.barh([feat_names_ebm[i] for i in order_ebm[::-1]],
        [feat_scores_ebm[i] for i in order_ebm[::-1]],
        color=colors, alpha=0.85)
ax.set_xlabel("Mean Absolute Score", fontsize=12)
ax.set_title("EBM Global Feature Importance (Top 20)", fontsize=14, fontweight="bold")
ax.axvline(np.mean(feat_scores_ebm), color="gray", ls="--", lw=1, label="ortalama")
ax.legend(fontsize=10)
plt.tight_layout(); fig.savefig(OUT/"ebm_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# EBM shape function for top-2 features
print("6/6  EBM shape functions...")
top2 = [feat_names_ebm[i] for i in order_ebm[:2]]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, fname in zip(axes, top2):
    if fname in ebm_global.data()["names"]:
        fidx = list(ebm_global.data()["names"]).index(fname)
        fd = ebm_global.data(fidx)
        if "names" in fd and "scores" in fd:
            xv = np.array(fd["names"])
            yv = np.array(fd["scores"])
            # EBM bin edges are n+1 while scores are n — use midpoints
            if len(xv) == len(yv) + 1:
                xv = (xv[:-1] + xv[1:]) / 2
            elif len(xv) > len(yv):
                xv = xv[:len(yv)]
            try:
                xv = xv.astype(float)
                ax.plot(xv, yv, color="#4C72B0", lw=2)
                ax.fill_between(xv, yv, alpha=0.2, color="#4C72B0")
            except (ValueError, TypeError):
                labels = [str(x) for x in xv]
                ax.bar(range(len(labels)), yv, color="#4C72B0", alpha=0.8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_title(f"EBM Shape: {fname}", fontsize=12, fontweight="bold")
            ax.set_xlabel(fname); ax.set_ylabel("contribution")
plt.tight_layout(); fig.savefig(OUT/"ebm_shape.png", dpi=150, bbox_inches="tight"); plt.close()

print("\nTüm görseller assets/ altına kaydedildi:")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f}")
