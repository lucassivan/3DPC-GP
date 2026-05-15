"""
3DPC Compressive Strength — GP Probabilistic Predictor
Model: GP_ARD_RBF  |  n=254  |  13 features  |  30-run 80/20 validation
"""
import os
import sys
import threading
import numpy as np
from scipy.stats import norm
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ── path setup: import gp_utils from sibling 算法模型 directory ────────────────
_BASE = Path(__file__).resolve().parent
_MODEL_DIR = _BASE.parent / '算法模型'
sys.path.insert(0, str(_MODEL_DIR))
RESULT_DIR = str(_MODEL_DIR / 'results')

from gp_utils import load_full_model, gp_predict, FEATURES   # noqa: E402

import customtkinter as ctk
from tkinter import messagebox

# ── constants ──────────────────────────────────────────────────────────────────
Z90 = 1.6449    # 90% PI half-width multiplier
Z95 = 1.9600    # 95% PI half-width multiplier

# Example mix: plain OPC 3DPC mortar (dataset row, actual CS = 81.47 MPa)
EXAMPLE = {
    'f(OPC)':  42.5,
    'n(OPC)':  0.80,
    'f(SAC)':  0.0,
    'n(SAC)':  0.0,
    'n(FA)':   0.0,
    'n(GGBS)': 0.0,
    'n(SF)':   0.05,
    'n(W/B)':  0.30,
    'n(B/S)':  1.0,
    'n(WRA)':  1.0,
    'n(Fb-E)': 0.0,
    'n(Fb-L)': 0.0,
    'n(Fb)':   0.0,
}

# Valid ranges derived from dataset (min, max) for out-of-range warnings
RANGES = {
    'f(OPC)':  (0.0,   52.5),
    'n(OPC)':  (0.0,   1.0),
    'f(SAC)':  (0.0,   42.5),
    'n(SAC)':  (0.0,   0.6),
    'n(FA)':   (0.0,   0.75),
    'n(GGBS)': (0.0,   0.30),
    'n(SF)':   (0.0,   0.27),
    'n(W/B)':  (0.0,   0.65),
    'n(B/S)':  (0.0,   4.5),
    'n(WRA)':  (0.0,   18.0),
    'n(Fb-E)': (0.0,  200.0),
    'n(Fb-L)': (0.0,   16.0),
    'n(Fb)':   (0.0,   4.0),
}

# ── colour palette (Nature-inspired, consistent with paper figures) ────────────
C = {
    'header':   '#1C3557',
    'g_binder': '#1C3557',
    'g_scm':    '#1B5E20',
    'g_mix':    '#4A148C',
    'g_fibre':  '#BF360C',
    'cs':       '#00A087',   # GP teal
    'unc':      '#3C5488',   # uncertainty blue
    'rel':      '#E64B35',   # reliability red
    'btn_p':    '#1C3557',   'btn_ph': '#243F6B',
    'btn_e':    '#E65100',   'btn_eh': '#BF360C',
    'btn_c':    '#546E7A',   'btn_ch': '#37474F',
    'bg':       '#F5F6FA',
    'card':     '#FFFFFF',
    'border':   '#DDE1E9',
    'secondary':'#4A6572',
    'unit':     '#546E7A',
    'auto_bg':  '#EEF1F5',
    'auto_txt': '#546E7A',
    'annot':    '#607D8B',
    'idle':     '#90A4AE',
    'pass':     '#2E7D32',
    'warn':     '#E65100',
    'fail':     '#C62828',
}


def _beta_grade(b):
    """Return (text, colour) for a reliability index."""
    if b >= 3.8:  return "Meets fib MC target (β ≥ 3.8)", C['pass']
    if b >= 3.0:  return "Below fib target — marginal",    C['warn']
    return "Fails reliability target",                      C['fail']


def _cs_grade(v):
    if v < 40:  return "Low (< 40 MPa)"
    if v < 60:  return "Moderate (40–60 MPa)"
    if v < 100: return "High (60–100 MPa)"
    return "Very high (≥ 100 MPa)"


# ── GUI application ────────────────────────────────────────────────────────────
class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("3DPC Compressive Strength — GP Probabilistic Predictor")
        self.geometry("1260x820")
        self.minsize(1100, 760)
        self.configure(fg_color=C['bg'])
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self._model = self._lk = self._sc_dict = None
        self._ready = False
        self._entries: dict[str, ctk.CTkEntry] = {}

        self._build()
        threading.Thread(target=self._load_model_bg, daemon=True).start()

    # ── background model loading ───────────────────────────────────────────────
    def _load_model_bg(self):
        self._set_status("Loading GP model from checkpoint…")
        try:
            m, lk, sc = load_full_model(RESULT_DIR)
            self._model, self._lk, self._sc_dict = m, lk, sc
            self._ready = True
            self.after(0, lambda: self._set_status(
                "GP_ARD_RBF ready  (R² = 0.887 ± 0.089, PICP₉₀ = 92.0%)"))
        except Exception as ex:
            self.after(0, lambda: self._set_status(f"Load error: {ex}"))

    # ── top-level layout ───────────────────────────────────────────────────────
    def _build(self):
        self.grid_rowconfigure(0, weight=0)   # banner
        self.grid_rowconfigure(1, weight=0)   # inputs
        self.grid_rowconfigure(2, weight=1)   # outputs
        self.grid_rowconfigure(3, weight=0)   # status bar
        self.grid_columnconfigure(0, weight=1)

        self._build_banner()
        self._build_input_card()
        self._build_output_row()
        self._build_statusbar()

    # ── banner ────────────────────────────────────────────────────────────────
    def _build_banner(self):
        bn = ctk.CTkFrame(self, fg_color=C['header'], corner_radius=0, height=82)
        bn.grid(row=0, column=0, sticky='ew')
        bn.grid_propagate(False)
        bn.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(bn,
                     text="3DPC Compressive Strength — GP Probabilistic Predictor",
                     font=ctk.CTkFont("Arial", 22, "bold"),
                     text_color="#FFFFFF").grid(row=0, column=0, pady=(18, 3))
        ctk.CTkLabel(bn,
                     text="GP_ARD_RBF  ·  n = 254  ·  13 Mix-Design Features  ·  "
                          "Calibrated 90%/95% Prediction Intervals  ·  β Reliability Index",
                     font=ctk.CTkFont("Arial", 11),
                     text_color="#90CAF9").grid(row=1, column=0, pady=(0, 12))

    # ── input card ────────────────────────────────────────────────────────────
    def _build_input_card(self):
        card = ctk.CTkFrame(self, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=1, column=0, sticky='ew', padx=18, pady=(12, 0))
        # 4 groups + 3 dividers
        for i in range(9):
            card.grid_columnconfigure(i, weight=(1 if i % 2 == 0 else 0))

        self._build_group_binder(card, col=0)
        self._divider(card, col=1)
        self._build_group_scm(card, col=2)
        self._divider(card, col=3)
        self._build_group_mix(card, col=4)
        self._divider(card, col=5)
        self._build_group_fibre(card, col=6)

        # buttons
        btn_row = ctk.CTkFrame(card, fg_color='transparent')
        btn_row.grid(row=1, column=0, columnspan=7, pady=(2, 14))

        ctk.CTkButton(btn_row, text="PREDICT", width=140, height=46,
                      font=ctk.CTkFont("Arial", 14, "bold"),
                      fg_color=C['btn_p'], hover_color=C['btn_ph'],
                      corner_radius=8, command=self._predict
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_row, text="EXAMPLE", width=115, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_e'], hover_color=C['btn_eh'],
                      corner_radius=8, command=self._example
                      ).pack(side='left', padx=8)
        ctk.CTkButton(btn_row, text="CLEAR", width=105, height=46,
                      font=ctk.CTkFont("Arial", 13),
                      fg_color=C['btn_c'], hover_color=C['btn_ch'],
                      corner_radius=8, command=self._clear
                      ).pack(side='left', padx=8)
        ctk.CTkLabel(btn_row,
                     text="  * all 13 features are active inputs  "
                          " — leave unused components at 0",
                     font=ctk.CTkFont("Arial", 11),
                     text_color=C['annot']).pack(side='left', padx=12)

    # ── group factories ────────────────────────────────────────────────────────
    def _build_group_binder(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=10, pady=10)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "BINDER COMPOSITION", C['g_binder'])
        rows = [
            ("f(OPC) *",  'f(OPC)',  "%"),
            ("n(OPC) *",  'n(OPC)',  "—"),
            ("f(SAC)",    'f(SAC)',  "%"),
            ("n(SAC)",    'n(SAC)',  "—"),
        ]
        for r, (label, feat, unit) in enumerate(rows, start=1):
            self._entry_row(grp, r, label, feat, unit)

    def _build_group_scm(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=10, pady=10)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "SUPPLEMENTARY CM", C['g_scm'])
        rows = [
            ("n(FA)",    'n(FA)',   "—"),
            ("n(GGBS)",  'n(GGBS)', "—"),
            ("n(SF)",    'n(SF)',   "—"),
        ]
        for r, (label, feat, unit) in enumerate(rows, start=1):
            self._entry_row(grp, r, label, feat, unit)

    def _build_group_mix(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=10, pady=10)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "MIX PROPORTIONS", C['g_mix'])
        rows = [
            ("n(W/B) *",  'n(W/B)',  "—"),
            ("n(B/S)",    'n(B/S)',  "—"),
            ("n(WRA) *",  'n(WRA)',  "—"),
        ]
        for r, (label, feat, unit) in enumerate(rows, start=1):
            self._entry_row(grp, r, label, feat, unit)

    def _build_group_fibre(self, parent, col):
        grp = ctk.CTkFrame(parent, fg_color='transparent')
        grp.grid(row=0, column=col, sticky='nsew', padx=10, pady=10)
        grp.grid_columnconfigure(1, weight=1)
        self._group_hdr(grp, "FIBRE REINFORCEMENT", C['g_fibre'])
        rows = [
            ("n(Fb-E)",  'n(Fb-E)', "—"),
            ("n(Fb-L)",  'n(Fb-L)', "—"),
            ("n(Fb)",    'n(Fb)',   "vol%"),
        ]
        for r, (label, feat, unit) in enumerate(rows, start=1):
            self._entry_row(grp, r, label, feat, unit)

    # ── output row ────────────────────────────────────────────────────────────
    def _build_output_row(self):
        row = ctk.CTkFrame(self, fg_color='transparent')
        row.grid(row=2, column=0, sticky='nsew', padx=18, pady=12)
        row.grid_columnconfigure(0, weight=5)
        row.grid_columnconfigure(1, weight=4)
        row.grid_columnconfigure(2, weight=5)
        row.grid_rowconfigure(0, weight=1)

        self._build_card_prediction(row, col=0)
        self._build_card_uncertainty(row, col=1)
        self._build_card_reliability(row, col=2)

    def _build_card_prediction(self, parent, col):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=0, column=col, sticky='nsew', padx=(0, 8))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(card, fg_color=C['cs'], corner_radius=8, height=40)
        hdr.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 6))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="GP Prediction",
                     font=ctk.CTkFont("Arial", 14, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.grid(row=1, column=0, sticky='nsew')
        inner.grid_columnconfigure(0, weight=1)

        self._lbl_mean = ctk.CTkLabel(inner, text="— MPa",
                                      font=ctk.CTkFont("Arial", 44, "bold"),
                                      text_color=C['cs'])
        self._lbl_mean.grid(row=0, column=0, pady=(18, 4))

        self._lbl_pi90 = ctk.CTkLabel(inner, text="90% PI:  —",
                                      font=ctk.CTkFont("Arial", 13),
                                      text_color=C['secondary'])
        self._lbl_pi90.grid(row=1, column=0, pady=2)

        self._lbl_pi95 = ctk.CTkLabel(inner, text="95% PI:  —",
                                      font=ctk.CTkFont("Arial", 13),
                                      text_color=C['secondary'])
        self._lbl_pi95.grid(row=2, column=0, pady=2)

        self._lbl_cs_grade = ctk.CTkLabel(inner, text="",
                                          font=ctk.CTkFont("Arial", 12, slant="italic"),
                                          text_color=C['idle'])
        self._lbl_cs_grade.grid(row=3, column=0, pady=(6, 20))

    def _build_card_uncertainty(self, parent, col):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=0, column=col, sticky='nsew', padx=8)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(card, fg_color=C['unc'], corner_radius=8, height=40)
        hdr.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 6))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="Uncertainty Decomposition",
                     font=ctk.CTkFont("Arial", 14, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.grid(row=1, column=0, sticky='nsew', padx=14)
        inner.grid_columnconfigure(1, weight=1)

        labels = [
            ("σ_total",      "Total predictive std"),
            ("σ_epistemic",  "Epistemic (reducible)"),
            ("σ_aleatoric",  "Aleatoric (irreducible)"),
        ]
        self._unc_vals: dict[str, ctk.CTkLabel] = {}
        for r, (key, desc) in enumerate(labels):
            ctk.CTkLabel(inner, text=desc + ":",
                         font=ctk.CTkFont("Arial", 12),
                         text_color=C['secondary'],
                         anchor='e').grid(row=r*2, column=0, columnspan=2,
                                          sticky='ew', pady=(16 if r == 0 else 8, 0))
            lbl = ctk.CTkLabel(inner, text="—",
                               font=ctk.CTkFont("Arial", 22, "bold"),
                               text_color=C['unc'])
            lbl.grid(row=r*2+1, column=0, columnspan=2, pady=(0, 4))
            self._unc_vals[key] = lbl

        # PICP reference note
        ctk.CTkLabel(inner,
                     text="σ_al ≈ 7.73 MPa (near-constant)",
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['annot']).grid(
            row=7, column=0, columnspan=2, pady=(8, 14))

    def _build_card_reliability(self, parent, col):
        card = ctk.CTkFrame(parent, fg_color=C['card'], corner_radius=12,
                            border_width=1, border_color=C['border'])
        card.grid(row=0, column=col, sticky='nsew', padx=(8, 0))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(card, fg_color=C['rel'], corner_radius=8, height=40)
        hdr.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 6))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="Reliability Index  (fib MC target: β ≥ 3.8)",
                     font=ctk.CTkFont("Arial", 13, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.grid(row=1, column=0, sticky='nsew', padx=14)
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=1)
        inner.grid_columnconfigure(2, weight=1)

        # Column headers
        for c, txt in enumerate(["f_c (MPa)", "P(CS ≥ f_c)", "β index"]):
            ctk.CTkLabel(inner, text=txt,
                         font=ctk.CTkFont("Arial", 11, "bold"),
                         text_color=C['rel']).grid(row=0, column=c, pady=(14, 4))

        self._rel_rows: list[tuple] = []   # (lbl_fc, lbl_prob, lbl_beta, lbl_verdict)
        thresholds = [40, 60, 80, 100, 120]
        for r, fc in enumerate(thresholds, start=1):
            lbl_fc = ctk.CTkLabel(inner, text=f"{fc}",
                                  font=ctk.CTkFont("Arial", 12),
                                  text_color=C['secondary'])
            lbl_fc.grid(row=r, column=0, pady=3)

            lbl_prob = ctk.CTkLabel(inner, text="—",
                                    font=ctk.CTkFont("Arial", 12),
                                    text_color=C['secondary'])
            lbl_prob.grid(row=r, column=1, pady=3)

            lbl_beta = ctk.CTkLabel(inner, text="—",
                                    font=ctk.CTkFont("Arial", 12, "bold"),
                                    text_color=C['secondary'])
            lbl_beta.grid(row=r, column=2, pady=3)

            self._rel_rows.append((fc, lbl_prob, lbl_beta))

        # Verdict label at bottom
        self._lbl_verdict = ctk.CTkLabel(inner, text="",
                                         font=ctk.CTkFont("Arial", 11, "bold"),
                                         text_color=C['idle'],
                                         wraplength=280)
        self._lbl_verdict.grid(row=6, column=0, columnspan=3, pady=(10, 14))

    # ── status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color='#E8EAF0', corner_radius=0, height=28)
        bar.grid(row=3, column=0, sticky='ew')
        bar.grid_propagate(False)
        self._status_lbl = ctk.CTkLabel(bar, text="Initialising…",
                                        font=ctk.CTkFont("Arial", 11),
                                        text_color=C['secondary'])
        self._status_lbl.place(relx=0.01, rely=0.5, anchor='w')

    def _set_status(self, msg: str):
        if hasattr(self, '_status_lbl'):
            self._status_lbl.configure(text=msg)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _group_hdr(parent, text, color):
        hdr = ctk.CTkFrame(parent, fg_color=color, corner_radius=6, height=30)
        hdr.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 6))
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text=text,
                     font=ctk.CTkFont("Arial", 11, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor='center')

    @staticmethod
    def _divider(parent, col):
        ctk.CTkFrame(parent, fg_color=C['border'], width=1).grid(
            row=0, column=col, sticky='ns', padx=2, pady=12)

    def _entry_row(self, grp, row, label, feat, unit):
        ctk.CTkLabel(grp, text=label,
                     font=ctk.CTkFont("Arial", 12, "bold"),
                     text_color=C['secondary']).grid(
            row=row, column=0, sticky='e', padx=(0, 6), pady=5)
        entry = ctk.CTkEntry(grp, width=90, height=34,
                             font=ctk.CTkFont("Arial", 12),
                             corner_radius=6, border_color=C['border'])
        entry.grid(row=row, column=1, padx=3, pady=5)
        ctk.CTkLabel(grp, text=unit,
                     font=ctk.CTkFont("Arial", 10),
                     text_color=C['unit']).grid(row=row, column=2,
                                                 sticky='w', padx=(3, 0))
        self._entries[feat] = entry

    # ── predict ───────────────────────────────────────────────────────────────
    def _predict(self):
        if not self._ready:
            messagebox.showwarning("Not Ready", "GP model is still loading, please wait.")
            return

        # Parse inputs
        try:
            vals = {}
            for feat in FEATURES:
                raw = self._entries[feat].get().strip()
                vals[feat] = float(raw) if raw else 0.0
        except ValueError:
            messagebox.showerror("Input Error",
                                 "All fields must contain numeric values (or be empty for 0).")
            return

        # Out-of-range check
        warns = []
        for feat, (lo, hi) in RANGES.items():
            v = vals[feat]
            if not (lo <= v <= hi):
                warns.append(f"  {feat} = {v} outside training range [{lo}, {hi}]")
        if warns:
            msg = "The following inputs lie outside the training range:\n\n"
            msg += "\n".join(warns)
            msg += "\n\nProceed with extrapolated prediction?"
            if not messagebox.askyesno("Out of Range", msg):
                return

        # GP prediction
        X_raw = np.array([[vals[f] for f in FEATURES]], dtype=np.float64)
        mu, sig, epi, ale = gp_predict(self._model, self._lk, X_raw, self._sc_dict)
        mu_  = float(mu[0])
        sig_ = float(sig[0])
        epi_ = float(epi[0])
        ale_ = float(ale[0])

        # Prediction intervals
        lo90, hi90 = mu_ - Z90 * sig_, mu_ + Z90 * sig_
        lo95, hi95 = mu_ - Z95 * sig_, mu_ + Z95 * sig_

        # Update prediction card
        self._lbl_mean.configure(text=f"{mu_:.1f} MPa")
        self._lbl_pi90.configure(
            text=f"90% PI:  [{lo90:.1f},  {hi90:.1f}] MPa")
        self._lbl_pi95.configure(
            text=f"95% PI:  [{lo95:.1f},  {hi95:.1f}] MPa")
        self._lbl_cs_grade.configure(text=_cs_grade(mu_), text_color=C['secondary'])

        # Update uncertainty card
        self._unc_vals['σ_total'].configure(text=f"{sig_:.2f} MPa")
        self._unc_vals['σ_epistemic'].configure(text=f"{epi_:.2f} MPa")
        self._unc_vals['σ_aleatoric'].configure(text=f"{ale_:.2f} MPa")

        # Update reliability card
        best_beta = None
        for fc, lbl_prob, lbl_beta in self._rel_rows:
            p_exc = float(norm.sf((fc - mu_) / (sig_ + 1e-9)) * 100)
            p_exc = np.clip(p_exc, 0.0, 100.0)
            p_safe = np.clip(p_exc / 100, 1e-9, 1 - 1e-9)
            beta  = float(norm.ppf(p_safe))

            lbl_prob.configure(text=f"{p_exc:.1f}%")
            lbl_beta.configure(text=f"{beta:.3f}")
            # colour by pass/fail at fib target
            col = C['pass'] if beta >= 3.8 else (C['warn'] if beta >= 0 else C['fail'])
            lbl_beta.configure(text_color=col)
            lbl_prob.configure(text_color=col)

            if best_beta is None:
                best_beta = (fc, beta)

        # Verdict based on β at fc=40 MPa (first row)
        _, b40 = best_beta if best_beta else (40, -99)
        verdict_txt, verdict_col = _beta_grade(b40)
        self._lbl_verdict.configure(text=f"At f_c = 40 MPa: {verdict_txt}",
                                    text_color=verdict_col)

        self._set_status(
            f"Predicted  —  n(OPC)={vals['n(OPC)']:.2f}  "
            f"n(W/B)={vals['n(W/B)']:.2f}  "
            f"n(WRA)={vals['n(WRA)']:.2f}  "
            f"CS = {mu_:.1f} ± {sig_:.2f} MPa (σ_ep={epi_:.2f}, σ_al={ale_:.2f})")

    # ── example / clear ───────────────────────────────────────────────────────
    def _example(self):
        self._clear()
        for feat, val in EXAMPLE.items():
            e = self._entries[feat]
            e.insert(0, str(val))
        self._set_status(
            "Example loaded: plain OPC 3DPC mortar "
            "(n(OPC)=0.80, n(W/B)=0.30, n(WRA)=1.0 — expected CS ≈ 81 MPa).")

    def _clear(self):
        for e in self._entries.values():
            e.delete(0, 'end')
        # Reset prediction card
        self._lbl_mean.configure(text="— MPa")
        self._lbl_pi90.configure(text="90% PI:  —")
        self._lbl_pi95.configure(text="95% PI:  —")
        self._lbl_cs_grade.configure(text="", text_color=C['idle'])
        # Reset uncertainty card
        for lbl in self._unc_vals.values():
            lbl.configure(text="—")
        # Reset reliability card
        for _, lbl_prob, lbl_beta in self._rel_rows:
            lbl_prob.configure(text="—", text_color=C['secondary'])
            lbl_beta.configure(text="—", text_color=C['secondary'])
        self._lbl_verdict.configure(text="", text_color=C['idle'])
        self._set_status("Cleared.")


if __name__ == '__main__':
    app = App()
    app.mainloop()
