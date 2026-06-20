"""
3DPC Compressive Strength – GP_ARD_RBF vs 7 ML Baselines
30 × 80/20 random splits | Fixed default params for ML | MLL for GP
Saves per-split predictions and full-model checkpoint for downstream analysis.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
N_SPLITS   = 30
TEST_SIZE  = 0.2
N_ITER_GP  = 300
N_RESTARTS = 5
LR_GP      = 0.1

FEATURES  = ['f(OPC)', 'n(OPC)', 'f(SAC)', 'n(SAC)', 'n(FA)', 'n(GGBS)', 'n(SF)',
             'n(W/B)', 'n(B/S)', 'n(WRA)', 'n(Fb-E)', 'n(Fb-L)', 'n(Fb)']
FEAT_COLS = ['f_OPC', 'n_OPC', 'f_SAC', 'n_SAC', 'n_FA', 'n_GGBS', 'n_SF',
             'n_WB', 'n_BS', 'n_WRA', 'n_FbE', 'n_FbL', 'n_Fb']
TARGET    = 'CS'
N_FEAT    = len(FEATURES)

DATA_PATH  = r'D:\PythonProjects\UHPC\3D_printed_concrete\dataset.xlsx'
RESULT_DIR = Path('results')
RESULT_DIR.mkdir(exist_ok=True)

# ── Fixed ML hyperparameters ────────────────────────────────────────────────────
ML_PARAMS = {
    'ElasticNet':   dict(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    'SVR':          dict(kernel='rbf', C=10.0, epsilon=0.5, gamma='scale'),
    'RandomForest': dict(n_estimators=300, max_depth=10, min_samples_leaf=2,
                         random_state=42, n_jobs=-1),
    'ExtraTrees':   dict(n_estimators=300, max_depth=10, min_samples_leaf=2,
                         random_state=42, n_jobs=-1),
    'XGBoost':      dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         verbosity=0, random_state=42),
    'LightGBM':     dict(n_estimators=300, max_depth=5, num_leaves=31,
                         learning_rate=0.05, subsample=0.8,
                         verbose=-1, random_state=42),
    'CatBoost':     dict(iterations=300, depth=5, learning_rate=0.05,
                         verbose=0, random_state=42),
}

# ── Style ──────────────────────────────────────────────────────────────────────
C_GP   = '#00A087'; C_RED  = '#E64B35'; C_BLUE = '#3C5488'
C_CYAN = '#4DBBD5'; C_BG   = '#FAFAFA'
NATURE_PAL = ['#00A087','#E64B35','#3C5488','#4DBBD5','#8491B4','#F39B7F',
              '#E6A023','#B07AA1','#7E6148','#91D1C2','#1B5E20','#880E4F','#1A237E']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.facecolor': C_BG, 'figure.facecolor': 'white',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'legend.frameon': False, 'legend.fontsize': 9,
    'grid.linestyle': '--', 'grid.alpha': 0.25,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.unicode_minus': False,
})

# ── GP model (ARD-RBF) ─────────────────────────────────────────────────────────
class ARDRBF_GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=N_FEAT))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def train_gp(Xtr, ytr, seed, n_restarts=N_RESTARTS, n_iter=N_ITER_GP):
    best_loss, best_model_sd, best_like_sd = float('inf'), None, None
    for r in range(n_restarts):
        torch.manual_seed(seed * 100 + r)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        model      = ARDRBF_GP(Xtr, ytr, likelihood)
        model.train(); likelihood.train()
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = (
                torch.rand(1, N_FEAT, dtype=torch.float64) * 1.5 + 0.3)
            model.covar_module.outputscale = torch.tensor(1.0, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_GP)
        mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        with gpytorch.settings.cholesky_jitter(1e-3):
            for _ in range(n_iter):
                optimizer.zero_grad()
                loss = -mll(model(Xtr), ytr)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final_loss = -mll(model(Xtr), ytr).item()
        if final_loss < best_loss:
            best_loss     = final_loss
            best_model_sd = {k: v.clone() for k, v in model.state_dict().items()}
            best_like_sd  = {k: v.clone() for k, v in likelihood.state_dict().items()}
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model      = ARDRBF_GP(Xtr, ytr, likelihood)
    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_like_sd)
    return model, likelihood


def predict_gp_full(model, likelihood, Xte):
    """Returns mu, sig_total, sig_epistemic, sig_aleatoric in scaled space."""
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3), \
         gpytorch.settings.fast_pred_var():
        f_dist = model(Xte)
        y_dist = likelihood(model(Xte))
    mu        = y_dist.mean.numpy()
    sig_total = y_dist.stddev.numpy()
    sig_epi   = f_dist.stddev.numpy()
    noise_var = max(float(likelihood.noise.item()), 1e-8)
    sig_ale   = np.full_like(sig_total, np.sqrt(noise_var))
    return mu, sig_total, sig_epi, sig_ale


# ── Metrics ────────────────────────────────────────────────────────────────────
def point_metrics(y_true, y_pred):
    return dict(
        R2   = float(r2_score(y_true, y_pred)),
        RMSE = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE  = float(mean_absolute_error(y_true, y_pred)),
    )


def uq_metrics(y_true, y_pred, y_std):
    def _picp(z):
        lo = y_pred - z * y_std
        hi = y_pred + z * y_std
        return float(np.mean((y_true >= lo) & (y_true <= hi)))
    p90    = _picp(1.6449)
    p95    = _picp(1.9600)
    mpiw90 = float(np.mean(2 * 1.6449 * y_std))
    levels = np.arange(0.10, 1.00, 0.05)
    ece    = float(np.mean([
        abs(_picp(scipy.stats.norm.ppf((1 + lv) / 2)) - lv)
        for lv in levels]))
    nll = float(-np.mean(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=y_std)))
    return dict(PICP90=p90, PICP95=p95, MPIW90=mpiw90, ECE=ece, NLL=nll)


def build_ml(name, seed):
    p = ML_PARAMS[name].copy()
    if name not in ('ElasticNet', 'SVR'):
        p['random_state'] = seed
    return {
        'ElasticNet':   ElasticNet,
        'SVR':          SVR,
        'RandomForest': RandomForestRegressor,
        'ExtraTrees':   ExtraTreesRegressor,
        'XGBoost':      xgb.XGBRegressor,
        'LightGBM':     lgb.LGBMRegressor,
        'CatBoost':     CatBoostRegressor,
    }[name](**p)


# ── Load data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_excel(DATA_PATH, sheet_name='CS')
df     = df_raw[FEATURES + [TARGET]].apply(pd.to_numeric, errors='coerce').dropna()
X      = df[FEATURES].values.astype(np.float64)
y      = df[TARGET].values.astype(np.float64)
print(f"Dataset: {len(df)} samples | {N_FEAT} features | target: {TARGET}")

ML_NAMES   = ['ElasticNet', 'SVR', 'RandomForest', 'ExtraTrees',
              'XGBoost', 'LightGBM', 'CatBoost']
ALL_NAMES  = ['GP_ARD_RBF'] + ML_NAMES
NEED_SCALE = {'ElasticNet', 'SVR'}

# ── 30 × 80/20 evaluation ─────────────────────────────────────────────────────
print(f"\n30 x 80/20 evaluation  (GP: {N_RESTARTS} restarts x {N_ITER_GP} iter) ...")

all_rows    = []
ard_splits  = []
split_preds = []

for sp in range(N_SPLITS):
    seed = sp + 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed)

    row = {'split': seed}

    # ── GP_ARD_RBF ──────────────────────────────────────────────────────────
    sc_X = StandardScaler().fit(X_tr)
    sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    Xtr_t = torch.tensor(sc_X.transform(X_tr),                      dtype=torch.float64)
    ytr_t = torch.tensor(sc_y.transform(y_tr.reshape(-1,1)).ravel(), dtype=torch.float64)
    Xte_t = torch.tensor(sc_X.transform(X_te),                      dtype=torch.float64)

    model, lk = train_gp(Xtr_t, ytr_t, seed)
    mu_s, sig_s, epi_s, ale_s = predict_gp_full(model, lk, Xte_t)

    scale_y = float(sc_y.scale_[0])
    mu  = sc_y.inverse_transform(mu_s.reshape(-1, 1)).ravel()
    sig = sig_s * scale_y
    epi = epi_s * scale_y
    ale = ale_s * scale_y

    for k, v in {**point_metrics(y_te, mu), **uq_metrics(y_te, mu, sig)}.items():
        row[f'GP_ARD_RBF_{k}'] = v

    ard_splits.append(
        model.covar_module.base_kernel.lengthscale.detach().numpy().ravel().copy())

    # Collect per-test-point predictions (with feature values for subgroup analysis)
    for i in range(len(y_te)):
        d = dict(split=seed, y_true=float(y_te[i]), y_mu=float(mu[i]),
                 y_sig=float(sig[i]), y_epi=float(epi[i]), y_ale=float(ale[i]))
        for fi, fc in enumerate(FEAT_COLS):
            d[fc] = float(X_te[i, fi])
        split_preds.append(d)

    # ── ML baselines ──────────────────────────────────────────────────────────
    sc_X_ml = StandardScaler().fit(X_tr)
    Xtr_s   = sc_X_ml.transform(X_tr)
    Xte_s   = sc_X_ml.transform(X_te)
    for name in ML_NAMES:
        Xtr_use = Xtr_s if name in NEED_SCALE else X_tr
        Xte_use = Xte_s if name in NEED_SCALE else X_te
        m_fit   = build_ml(name, seed)
        m_fit.fit(Xtr_use, y_tr)
        for k, v in point_metrics(y_te, m_fit.predict(Xte_use)).items():
            row[f'{name}_{k}'] = v

    all_rows.append(row)
    print(f"  Split {seed:2d}/{N_SPLITS} | "
          f"GP R2={row['GP_ARD_RBF_R2']:.3f} PICP90={row['GP_ARD_RBF_PICP90']:.2f} | "
          f"CatBoost R2={row['CatBoost_R2']:.3f}")

# ── Save detailed results ──────────────────────────────────────────────────────
detail_df = pd.DataFrame(all_rows)
detail_df.to_csv(RESULT_DIR / 'gp_results_detail.csv', index=False)

pred_df = pd.DataFrame(split_preds)
pred_df.to_csv(RESULT_DIR / 'gp_per_split_preds.csv', index=False)
print(f"\nSaved: gp_per_split_preds.csv  ({len(pred_df)} rows)")

# ── Summary table ──────────────────────────────────────────────────────────────
summary_rows = []
for name in ALL_NAMES:
    r2v   = detail_df[f'{name}_R2'].values
    rmsev = detail_df[f'{name}_RMSE'].values
    maev  = detail_df[f'{name}_MAE'].values
    r = dict(Model=name,
             R2_mean=r2v.mean(),     R2_std=r2v.std(),
             RMSE_mean=rmsev.mean(), RMSE_std=rmsev.std(),
             MAE_mean=maev.mean(),   MAE_std=maev.std())
    if name == 'GP_ARD_RBF':
        for m in ['PICP90', 'PICP95', 'MPIW90', 'ECE', 'NLL']:
            v = detail_df[f'GP_ARD_RBF_{m}'].values
            r[f'{m}_mean'] = v.mean(); r[f'{m}_std'] = v.std()
    summary_rows.append(r)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULT_DIR / 'gp_results_summary.csv', index=False)

# ── ARD length-scales ──────────────────────────────────────────────────────────
ard_arr = np.array(ard_splits)
ard_df  = pd.DataFrame({
    'Feature': FEATURES,
    'LS_mean': ard_arr.mean(axis=0),
    'LS_std':  ard_arr.std(axis=0),
}).sort_values('LS_mean').reset_index(drop=True)
ard_df.to_csv(RESULT_DIR / 'gp_ard_lengthscales.csv', index=False)

# ── Train full model on all 254 samples (for downstream analysis) ──────────────
print("\nTraining full model on all 254 samples ...")
sc_X_full = StandardScaler().fit(X)
sc_y_full = StandardScaler().fit(y.reshape(-1, 1))
Xfull_t   = torch.tensor(sc_X_full.transform(X),                      dtype=torch.float64)
yfull_t   = torch.tensor(sc_y_full.transform(y.reshape(-1,1)).ravel(), dtype=torch.float64)
model_full, lk_full = train_gp(Xfull_t, yfull_t, seed=99)
torch.save({
    'model_state': model_full.state_dict(),
    'lk_state':    lk_full.state_dict(),
    'Xfull_t':     Xfull_t,
    'yfull_t':     yfull_t,
    'sc_X_mean':   sc_X_full.mean_,
    'sc_X_std':    sc_X_full.scale_,
    'sc_y_mean':   sc_y_full.mean_[0],
    'sc_y_std':    sc_y_full.scale_[0],
}, RESULT_DIR / 'gp_full_model.pt')
print("Saved: gp_full_model.pt")

# ── Print summary ──────────────────────────────────────────────────────────────
SEP = '=' * 70
print(f'\n{SEP}')
print('SUMMARY  (mean +/- std,  30 splits)')
print(SEP)
print(f"{'Model':15s}  {'R2':>13s}  {'RMSE (MPa)':>13s}  {'MAE (MPa)':>13s}")
print('-' * 60)
for _, r in summary_df.sort_values('R2_mean', ascending=False).iterrows():
    print(f"{r['Model']:15s}  "
          f"{r['R2_mean']:.3f}+/-{r['R2_std']:.3f}  "
          f"{r['RMSE_mean']:.2f}+/-{r['RMSE_std']:.2f}      "
          f"{r['MAE_mean']:.2f}+/-{r['MAE_std']:.2f}")

gp = summary_df[summary_df['Model'] == 'GP_ARD_RBF'].iloc[0]
print(f'\nGP Uncertainty Metrics:')
print(f"  PICP90 = {gp['PICP90_mean']:.3f} +/- {gp['PICP90_std']:.3f}   (target 0.90)")
print(f"  PICP95 = {gp['PICP95_mean']:.3f} +/- {gp['PICP95_std']:.3f}   (target 0.95)")
print(f"  MPIW90 = {gp['MPIW90_mean']:.2f} +/- {gp['MPIW90_std']:.2f} MPa")
print(f"  ECE    = {gp['ECE_mean']:.4f} +/- {gp['ECE_std']:.4f}")
print(f"  NLL    = {gp['NLL_mean']:.3f} +/- {gp['NLL_std']:.3f}")

print(f'\nARD Length-scales (ascending = more important):')
for _, r in ard_df.iterrows():
    bar = '#' * max(1, int(15 * r['LS_mean'] / ard_df['LS_mean'].max()))
    print(f"  {r['Feature']:10s}  {r['LS_mean']:.3f}+/-{r['LS_std']:.3f}  {bar}")

# ── Figure 4: R2 / RMSE comparison bar chart ───────────────────────────────────
ML_COLORS = {'ElasticNet': '#B2B2B2', 'SVR': '#8491B4',
             'RandomForest': '#4DBBD5', 'ExtraTrees': '#3C5488',
             'XGBoost': '#F39B7F', 'LightGBM': '#E6A023', 'CatBoost': '#E64B35'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
order = summary_df.sort_values('R2_mean', ascending=False)['Model'].tolist()
colors_order = [C_GP if m == 'GP_ARD_RBF' else ML_COLORS.get(m, '#555555')
                for m in order]

ax = axes[0]
r2_m = [summary_df.loc[summary_df.Model==m, 'R2_mean'].values[0] for m in order]
r2_s = [summary_df.loc[summary_df.Model==m, 'R2_std'].values[0]  for m in order]
ax.barh(order[::-1], r2_m[::-1], xerr=r2_s[::-1],
        color=colors_order[::-1], alpha=0.85,
        error_kw=dict(elinewidth=1.2, capsize=4, ecolor='#555555'))
ax.axvline(r2_m[0], color=C_RED, lw=1.2, ls='--', alpha=0.7)
ax.set_xlabel('R²  (mean ± SD, 30 splits)')
ax.set_title('(a)  Prediction Accuracy (R²)')
ax.set_xlim(0, 1.06)
for i, (m, s) in enumerate(zip(r2_m[::-1], r2_s[::-1])):
    ax.text(m + s + 0.004, i, f'{m:.3f}', va='center', fontsize=8.5)
ax.grid(True, axis='x')

ax2 = axes[1]
rm_m = [summary_df.loc[summary_df.Model==m, 'RMSE_mean'].values[0] for m in order]
rm_s = [summary_df.loc[summary_df.Model==m, 'RMSE_std'].values[0]  for m in order]
ax2.barh(order[::-1], rm_m[::-1], xerr=rm_s[::-1],
         color=colors_order[::-1], alpha=0.85,
         error_kw=dict(elinewidth=1.2, capsize=4, ecolor='#555555'))
ax2.set_xlabel('RMSE (MPa)  (mean ± SD, 30 splits)')
ax2.set_title('(b)  Prediction Error (RMSE)')
for i, (m, s) in enumerate(zip(rm_m[::-1], rm_s[::-1])):
    ax2.text(m + s + 0.1, i, f'{m:.2f}', va='center', fontsize=8.5)
ax2.grid(True, axis='x')

plt.tight_layout(pad=1.4)
for ext in ('png', 'svg'):
    fig.savefig(RESULT_DIR / f'fig4_model_comparison.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('\nSaved: fig4_model_comparison.png/svg')

# ── Figure (quick ARD) ─────────────────────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(7, 5))
y_pos = range(len(ard_df))
ax3.barh(list(y_pos), ard_df['LS_mean'].values, xerr=ard_df['LS_std'].values,
         color=[NATURE_PAL[i % 13] for i in range(len(ard_df))],
         alpha=0.85, error_kw=dict(elinewidth=1.2, capsize=4, ecolor='#555555'))
ax3.set_yticks(list(y_pos))
ax3.set_yticklabels(ard_df['Feature'].tolist(), fontsize=9)
ax3.set_xlabel('ARD Length-scale  (shorter = more influential)')
ax3.set_title('GP-ARD Kernel Length-scales\n(mean ± SD, 30 splits)')
ax3.axvline(ard_df['LS_mean'].mean(), color='#555555', ls='--', lw=1.2, alpha=0.7)
ax3.grid(True, axis='x')
plt.tight_layout(pad=1.2)
for ext in ('png', 'svg'):
    fig2.savefig(RESULT_DIR / f'fig_ard_quick.{ext}',
                 dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print('Saved: fig_ard_quick.png/svg')

print(f'\nAll results in: {RESULT_DIR.resolve()}')
print('Done.')
