import csv, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Volumes/G-Drive Pro/DementiaBank"
d = np.load(f"{D}/ad_llm_states.npz", allow_pickle=True)
X, y, spk = d["feats"].astype(np.float32), d["label"], d["spk"]
print("llm states", X.shape, "speakers", len(set(spk)))

gkf = GroupKFold(n_splits=5)
def curve(Xall):
    out = []
    for L in range(Xall.shape[1]):
        XL, aucs = Xall[:, L, :], []
        for tr, te in gkf.split(XL, y, groups=spk):
            sc = StandardScaler().fit(XL[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(XL[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(XL[te]))[:, 1]))
        out.append((float(np.mean(aucs)), float(np.std(aucs))))
        print(f"stage {L:2d} auc {out[-1][0]:.3f} ± {out[-1][1]:.3f}", flush=True)
    return out

llm = curve(X)
with open(f"{D}/ad_llm_perlayer.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["stage", "auc_mean", "auc_std"])
    for i, (m, s) in enumerate(llm): w.writerow([i, m, s])

enc = [(float(r["auc_mean"]), float(r["auc_std"])) for r in csv.DictReader(open(f"{D}/ad_encoder_perlayer.csv"))]

fig, ax = plt.subplots(figsize=(11.5, 5.2))
ex = np.arange(len(enc)); lx = np.arange(len(llm)) + len(enc)
em, es = np.array([v[0] for v in enc]), np.array([v[1] for v in enc])
lm, ls = np.array([v[0] for v in llm]), np.array([v[1] for v in llm])
ax.plot(ex, em, "-o", ms=3.2, lw=1.6, color="#E64A19", label="audio encoder layers")
ax.fill_between(ex, em-es, em+es, color="#E64A19", alpha=0.12, lw=0)
ax.plot(lx, lm, "-s", ms=3.2, lw=1.6, color="#6A1B9A", label="projector, then language model layers (audio token positions)")
ax.fill_between(lx, lm-ls, lm+ls, color="#6A1B9A", alpha=0.12, lw=0)
ax.axvline(len(enc)-0.5, color="gray", lw=1, ls="-", alpha=0.6)
ax.text(len(enc)/2, 0.965, "audio encoder", ha="center", fontsize=9, color="#E64A19")
ax.text(len(enc)+len(llm)/2, 0.965, "language model", ha="center", fontsize=9, color="#6A1B9A")
ax.axhline(0.633, color="black", ls="--", lw=1.1, alpha=0.7)
ax.text(64.5, 0.641, "model's own answer 0.63", fontsize=8, ha="right")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.text(0.3, 0.505, "chance", color="gray", fontsize=8)
ax.set_xlabel("stage  (encoder layer 0-32, then projector output, then each LLM layer)")
ax.set_ylabel("probe AUC (speaker-disjoint)")
ax.set_title("Alzheimer's (DementiaBank Pitt): disease information from the audio encoder through the language model\n"
             "Qwen2-Audio, per-stage linear probe, 5-fold GroupKFold by speaker, 468 cookie-task segments, 228 speakers",
             fontsize=10.5)
ax.set_ylim(0.42, 1.0); ax.set_xlim(-1, 66.5)
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{D}/ad_encoder_to_llm.{ext}", dpi=300)
print("figure saved", flush=True)
