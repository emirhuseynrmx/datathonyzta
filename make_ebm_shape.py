"""Re-run only EBM shape function plot (other assets already exist)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

OUT = Path("assets")
TARGET = "bilissel_performans_skoru"
NUM = ["stres_skoru","rem_yuzdesi","derin_uyku_yuzdesi","gunluk_calisma_saati",
       "gecelik_uyanma_sayisi","gunluk_adim_sayisi","uyku_oncesi_kafein_mg",
       "yas","sekerleme_suresi_dk","dinlenik_nabiz_bpm","uykuya_dalma_suresi_dk",
       "oda_sicakligi_celsius","vucut_kitle_indeksi","hafta_sonu_uyku_farki_saat",
       "uyku_oncesi_ekran_suresi_dk"]
CAT = ["meslek","ruh_sagligi_durumu","gun_tipi","kronotip","mevsim","cinsiyet","ulke"]

train = pd.read_csv("data/raw/train.csv")
y = train[TARGET].values

X = train[NUM + CAT].copy()
for c in NUM: X[c] = X[c].fillna(X[c].median())
for c in CAT: X[c] = X[c].fillna("Unknown").astype(str)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc = X.copy(); X_enc[CAT] = enc.fit_transform(X[CAT])
X_enc = X_enc.astype(np.float32)
X_tr, X_val, y_tr, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

print("Training EBM...")
from interpret.glassbox import ExplainableBoostingRegressor
ebm = ExplainableBoostingRegressor(interactions=10, max_bins=256, learning_rate=0.01,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1,
                                    feature_names=NUM + CAT)
ebm.fit(X_tr.values, y_tr)
print("EBM trained.")

ebm_global = ebm.explain_global(name="EBM Global")
scores = ebm_global.data()
feat_names_ebm = scores["names"]
feat_scores_ebm = scores["scores"]
order_ebm = np.argsort(feat_scores_ebm)[::-1]

top2 = [feat_names_ebm[i] for i in order_ebm[:2]]
print(f"Top-2 features: {top2}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, fname in zip(axes, top2):
    if fname not in feat_names_ebm:
        continue
    fidx = list(feat_names_ebm).index(fname)
    fd = ebm_global.data(fidx)
    if "names" not in fd or "scores" not in fd:
        continue
    xv = np.array(fd["names"])
    yv = np.array(fd["scores"])
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

plt.tight_layout()
fig.savefig(OUT / "ebm_shape.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: assets/ebm_shape.png")
