import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
import scikit_posthocs as sp
from sktime.transformations.series.detrend import Detrender, Deseasonalizer
from sktime.forecasting.trend import PolynomialTrendForecaster, STLForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.transformations.series.filter import Filter
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TimeSeries Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1f2d45;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --text-muted: #334155;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--accent) !important;
    letter-spacing: -0.5px;
}

h2 { color: var(--text) !important; font-size: 1.1rem !important; }
h3 { color: var(--accent3) !important; font-size: 0.95rem !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stSlider > div,
.stNumberInput > div {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--accent) !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
}
.stat-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
}
.stat-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: var(--accent);
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-blue { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.7rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stExpander {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

div[data-testid="stHorizontalBlock"] { gap: 1rem; }

.upload-zone {
    border: 2px dashed var(--accent2);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: rgba(124,58,237,0.05);
}

hr { border-color: var(--border) !important; }

.stDataFrame { background: var(--surface2) !important; }
table { background: var(--surface2) !important; }

.info-box {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--text);
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.8)',
    font=dict(family='DM Sans', color='#94a3b8', size=11),
    xaxis=dict(gridcolor='#1f2d45', linecolor='#1f2d45', tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='#1f2d45', linecolor='#1f2d45', tickfont=dict(color='#64748b')),
    legend=dict(bgcolor='rgba(17,24,39,0.9)', bordercolor='#1f2d45', borderwidth=1, font=dict(color='#94a3b8')),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode='x unified',
)

COLORS = {
    'original': '#00d4ff',
    'locf':     '#7c3aed',
    'nocb':     '#a78bfa',
    'ma':       '#f59e0b',
    'linear':   '#10b981',
    'outlier':  '#ef4444',
    'cleaned':  '#10b981',
    'denoised': '#f59e0b',
    'ema':      '#a78bfa',
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_csv(file_bytes, date_col, value_col, date_fmt):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df[date_col] = pd.to_datetime(df[date_col], format=date_fmt if date_fmt else None, infer_datetime_format=True)
    df = df.sort_values(date_col).set_index(date_col)
    series = df[value_col].astype(float)
    return series


def detect_freq(series):
    if len(series) < 2:
        return None
    diffs = pd.Series(series.index).diff().dropna()
    mode_diff = diffs.mode()[0]
    days = mode_diff.days
    if days == 1:   return 'D'
    if days == 7:   return 'W'
    if 28 <= days <= 31: return 'ME'
    if 88 <= days <= 92: return 'QE'
    if 364 <= days <= 366: return 'YE'
    return None


def basic_stats(series):
    s = series.dropna()
    return {
        'Count':    len(series),
        'Missing':  series.isna().sum(),
        'Miss %':   f"{series.isna().mean()*100:.1f}%",
        'Mean':     f"{s.mean():.3f}",
        'Std':      f"{s.std():.3f}",
        'Min':      f"{s.min():.3f}",
        'Max':      f"{s.max():.3f}",
        'Median':   f"{s.median():.3f}",
        'Skew':     f"{s.skew():.3f}",
        'Kurt':     f"{s.kurtosis():.3f}",
    }


# ── Missing value imputation ──────────────────────────────────────────────────
def impute(series, method, window=6):
    s = series.copy()
    if method == 'LOCF':
        return s.ffill()
    if method == 'NOCB':
        return s.bfill()
    if method == 'Moving Average':
        filled = s.fillna(s.rolling(window, min_periods=1).mean())
        return filled
    if method == 'Linear':
        return s.interpolate(method='linear', limit_direction='both')
    if method == 'Spline':
        return s.interpolate(method='spline', order=3, limit_direction='both')
    return s


# ── Outlier detection ─────────────────────────────────────────────────────────
def hampel_filter(series, window=5, n_sigma=3, sp=12, poly_degree=2):
    """Run Hampel filter on residuals (deseasonalize → detrend), same as GESD."""
    s = series.copy()
    # build residuals the same way as residual_outliers
    resid = None
    s_freq = s.dropna().copy()
    if s_freq.index.freq is None:
        freq = pd.infer_freq(s_freq.index)
        if freq:
            s_freq.index = pd.DatetimeIndex(s_freq.index, freq=freq)
    try:
        deseason = Deseasonalizer(sp=sp, model="multiplicative")
        y_deseas = deseason.fit_transform(s_freq)
        detrend  = Detrender(forecaster=PolynomialTrendForecaster(degree=poly_degree))
        resid    = detrend.fit_transform(y_deseas)
        work     = resid
    except Exception:
        work = s.dropna()

    outlier_idx = []
    k_mad = 1.4826
    half  = window // 2
    for i in range(len(work)):
        lo = max(0, i - half)
        hi = min(len(work), i + half + 1)
        window_vals = work.iloc[lo:hi].dropna()
        if len(window_vals) < 2:
            continue
        med = window_vals.median()
        mad = k_mad * (window_vals - med).abs().median()
        if mad == 0:
            continue
        if abs(work.iloc[i] - med) > n_sigma * mad:
            outlier_idx.append(work.index[i])
    return outlier_idx, resid


def gesd_outliers(series, max_outliers=10, alpha=0.05):
    s = series.dropna()
    try:
        inliers = sp.outliers_gesd(s, outliers=max_outliers, alpha=alpha, report=False)
        outlier_vals = s[(s < inliers.min()) | (s > inliers.max())]
        return outlier_vals.index.tolist()
    except Exception:
        return []


def residual_outliers(series, sp=12, poly_degree=2, max_outliers=10, alpha=0.05):
    """Deseasonalize → detrend → GESD on residuals."""
    s = series.dropna()
    # need a PeriodIndex or DatetimeIndex with freq for sktime
    s_freq = s.copy()
    if s_freq.index.freq is None:
        freq = pd.infer_freq(s_freq.index)
        if freq:
            s_freq.index = pd.DatetimeIndex(s_freq.index, freq=freq)
    try:
        deseason = Deseasonalizer(sp=sp, model="multiplicative")
        y_deseas = deseason.fit_transform(s_freq)
        detrend = Detrender(forecaster=PolynomialTrendForecaster(degree=poly_degree))
        resid = detrend.fit_transform(y_deseas)
        out_idx = gesd_outliers(resid, max_outliers=max_outliers, alpha=alpha)
        # map residual outlier positions back to original series index
        return out_idx, resid
    except Exception as e:
        # fallback: just return GESD on raw series
        return gesd_outliers(s, max_outliers=max_outliers, alpha=alpha), None


# ── Denoising ─────────────────────────────────────────────────────────────────
def denoise_sma(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def denoise_ema(series, alpha=0.3):
    return series.ewm(alpha=alpha).mean()


def denoise_fft(series, h_freq=20):
    """
    True FFT low-pass denoising:
      1. rfft → compute magnitude spectrum
      2. Zero out all frequency bins above h_freq index
      3. irfft → reconstruct signal
    Both the denoised signal and the magnitude plots come from
    the same FFT arrays, so the frequency view is always consistent
    with what was actually removed.
    """
    s = series.dropna()
    n = len(s)

    vals = s.values.astype(float)

    # Forward FFT (real-input, so rfft gives n//2+1 bins)
    fft_vals  = np.fft.rfft(vals)
    freqs     = np.fft.rfftfreq(n)          # 0 … 0.5 (cycles/sample)
    orig_mag  = np.abs(fft_vals)

    # Low-pass: zero every bin whose index exceeds h_freq
    cutoff_idx = int(h_freq)                 # h_freq is now a bin index, 1 … n//2
    cutoff_idx = max(1, min(cutoff_idx, len(fft_vals) - 1))

    fft_filtered = fft_vals.copy()
    fft_filtered[cutoff_idx:] = 0           # hard zero above cutoff

    denoised_mag = np.abs(fft_filtered)

    # Inverse FFT → denoised signal, same length as input
    denoised_vals = np.fft.irfft(fft_filtered, n=n)

    denoised = pd.Series(
        denoised_vals[:n],                   # irfft can return n+1 on odd lengths
        index=s.index,
        name=s.name,
    )

    # Cutoff line sits at freqs[cutoff_idx] in the plot
    cutoff_freq = freqs[cutoff_idx]

    return denoised, orig_mag, denoised_mag, freqs, cutoff_freq


# ── Plotting helpers ──────────────────────────────────────────────────────────
def line_fig(traces, title='', height=320):
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=t['x'], y=t['y'],
            name=t['name'],
            line=dict(color=t.get('color', '#00d4ff'), width=t.get('width', 1.5)),
            mode='lines',
        ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=13, color='#94a3b8')), height=height)
    return fig


def hist_fig(series, title=''):
    fig = go.Figure(go.Histogram(
        x=series.dropna(), nbinsx=40,
        marker_color='#7c3aed',
        marker_line_color='#1f2d45', marker_line_width=0.5,
        opacity=0.85,
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=13, color='#94a3b8')), height=260)
    return fig


def acf_manual(series, nlags=30):
    s = series.dropna()
    n = len(s)
    mean = s.mean()
    var = ((s - mean)**2).sum() / n
    acf_vals = []
    for lag in range(nlags + 1):
        c = ((s[lag:].values - mean) * (s[:n-lag].values - mean)).sum() / n
        acf_vals.append(c / var if var != 0 else 0)
    return np.array(acf_vals)


def acf_fig(series, nlags=30, title='Autocorrelation (ACF)'):
    acf_vals = acf_manual(series, nlags)
    n = len(series.dropna())
    lags = list(range(len(acf_vals)))

    # Bartlett's formula: CI fans out as lag increases
    # var(r_k) ≈ (1 + 2*sum(r_i^2 for i<k)) / n
    ci_upper = []
    ci_lower = []
    for k in range(len(acf_vals)):
        if k == 0:
            ci_upper.append(0)
            ci_lower.append(0)
        else:
            bartlett_var = (1 + 2 * np.sum(acf_vals[1:k] ** 2)) / n
            margin = 1.96 * np.sqrt(bartlett_var)
            ci_upper.append(margin)
            ci_lower.append(-margin)

    colors = ['#ef4444' if abs(acf_vals[i]) > ci_upper[i] else '#7c3aed'
              for i in range(len(acf_vals))]

    fig = go.Figure()
    # Shaded significance band (fanning out)
    fig.add_trace(go.Scatter(
        x=lags + lags[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(0,212,255,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence',
        hoverinfo='skip',
    ))
    for i, (lag, val) in enumerate(zip(lags, acf_vals)):
        fig.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode='lines', line=dict(color=colors[i], width=2),
            showlegend=False, hoverinfo='skip',
        ))
    fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers',
        marker=dict(color=colors, size=5), name='ACF',
        hovertemplate='Lag %{x}: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_color='#334155', line_width=1)
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=13, color='#94a3b8')), height=260)
    return fig


def rolling_stats_fig(series, window=12):
    rm = series.rolling(window).mean()
    rs = series.rolling(window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name='Original',
        line=dict(color=COLORS['original'], width=1), opacity=0.4))
    fig.add_trace(go.Scatter(x=rm.index, y=rm, name=f'Rolling Mean ({window})',
        line=dict(color=COLORS['ma'], width=2)))
    fig.add_trace(go.Scatter(x=rs.index, y=rs, name=f'Rolling Std ({window})',
        line=dict(color=COLORS['outlier'], width=2), yaxis='y2'))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text='Stationarity Check — Rolling Stats', font=dict(size=13, color='#94a3b8')),
        yaxis2=dict(overlaying='y', side='right', gridcolor='#1f2d45', tickfont=dict(color='#64748b')),
        height=300,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in {
    'raw': None, 'imputed': None, 'outlier_cleaned': None, 'denoised': None,
    'outlier_indices': [], 'residuals': None,
    'decomp': None, 'pipeline_done': False,
    # forecasting
    'holt_results': None, 'hw_results': None, 'stl_results': None,
    'forecast_result': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 📈 TimeSeries<br><span style='color:#64748b;font-size:0.7rem;font-family:\"Space Mono\",monospace;letter-spacing:2px;'>LAB</span>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">01 · Data Upload</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=['csv'], label_visibility='collapsed')

    if uploaded:
        preview_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        cols = preview_df.columns.tolist()

        date_col  = st.selectbox("Date column",  cols, index=0)
        value_col = st.selectbox("Value column", [c for c in cols if c != date_col], index=0)
        date_fmt  = st.text_input("Date format (optional)", placeholder="%Y-%m-%d")

        st.markdown('<div class="section-header">02 · Missing Values</div>', unsafe_allow_html=True)
        impute_method = st.selectbox("Imputation method",
            ['Linear', 'LOCF', 'NOCB', 'Moving Average'])
        ma_window = 6
        if impute_method == 'Moving Average':
            ma_window = st.slider("MA window", 2, 24, 6)
        decomp_sp     = st.slider("Seasonality period (sp)", 2, 52, 12)
        decomp_degree = st.slider("Trend poly degree", 1, 3, 2)

        st.markdown('<div class="section-header">03 · Outlier Detection</div>', unsafe_allow_html=True)
        outlier_method = st.selectbox("Detection method", ['Hampel Filter', 'Residual (GESD)'])
        if outlier_method == 'Hampel Filter':
            h_window  = st.slider("Window length", 3, 21, 5, step=2)
            h_nsigma  = st.slider("n_sigma",       1.0, 5.0, 3.0, step=0.5)
        elif outlier_method == 'Residual (GESD)':
            gesd_max     = st.slider("Max outliers", 1, 20, 10)
            gesd_alpha   = st.slider("Alpha (significance)", 0.01, 0.10, 0.05, step=0.01)

        st.markdown('<div class="section-header">04 · Denoising</div>', unsafe_allow_html=True)
        denoise_method = st.selectbox("Denoising method", ['None', 'SMA', 'EMA', 'FFT'])
        sma_w = 3; ema_alpha = 0.3; fft_hfreq = 20
        if denoise_method == 'SMA':
            sma_w = st.slider("SMA window", 2, 24, 3)
        elif denoise_method == 'EMA':
            ema_alpha = st.slider("EMA alpha (α)", 0.05, 0.95, 0.3, step=0.05)
        elif denoise_method == 'FFT':
            fft_hfreq = st.slider("Low-pass cutoff (h_freq)", 1, 50, 20)

        st.markdown("---")
        run_btn = st.button("▶  Run Preprocessing", use_container_width=True)

        if run_btn:
            with st.spinner("Processing…"):
                raw = load_csv(uploaded.getvalue(), date_col, value_col,
                               date_fmt if date_fmt else None)
                st.session_state['raw'] = raw

                # 1. Imputation
                imputed = impute(raw, impute_method, window=ma_window)
                st.session_state['imputed'] = imputed

                # 1b. Decomposition (deseasonalize → detrend → residuals)
                decomp = None
                try:
                    s_freq = imputed.copy()
                    if s_freq.index.freq is None:
                        inferred = pd.infer_freq(s_freq.index)
                        if inferred:
                            s_freq.index = pd.DatetimeIndex(s_freq.index, freq=inferred)
                    deseason_t = Deseasonalizer(sp=decomp_sp, model="multiplicative")
                    y_deseas   = deseason_t.fit_transform(s_freq)
                    y_seasonal = s_freq / y_deseas
                    detrend_t  = Detrender(forecaster=PolynomialTrendForecaster(degree=decomp_degree))
                    y_detrended = detrend_t.fit_transform(y_deseas)
                    y_trend    = y_deseas - y_detrended
                    decomp = {
                        'deseasonalized': y_deseas,
                        'seasonal':       y_seasonal,
                        'trend':          y_trend,
                        'residuals':      y_detrended,
                    }
                except Exception:
                    pass
                st.session_state['decomp'] = decomp

                # 2. Outlier detection & replacement
                resid_series = None
                if outlier_method == 'Hampel Filter':
                    out_idx, resid_series = hampel_filter(
                        imputed, window=h_window, n_sigma=h_nsigma,
                        sp=decomp_sp, poly_degree=decomp_degree,
                    )
                elif outlier_method == 'Residual (GESD)':
                    out_idx, resid_series = residual_outliers(
                        imputed, sp=decomp_sp, poly_degree=decomp_degree,
                        max_outliers=gesd_max, alpha=gesd_alpha,
                    )
                else:
                    out_idx = []
                st.session_state['outlier_indices'] = out_idx
                st.session_state['residuals'] = resid_series
                cleaned = imputed.copy()
                cleaned[out_idx] = np.nan
                cleaned = cleaned.interpolate(method='linear', limit_direction='both')
                st.session_state['outlier_cleaned'] = cleaned

                # 3. Denoising
                if denoise_method == 'SMA':
                    denoised = denoise_sma(cleaned, window=sma_w)
                elif denoise_method == 'EMA':
                    denoised = denoise_ema(cleaned, alpha=ema_alpha)
                elif denoise_method == 'FFT':
                    denoised, _, _, _, _ = denoise_fft(cleaned, h_freq=fft_hfreq)
                else:
                    denoised = cleaned.copy()
                st.session_state['denoised'] = denoised
                st.session_state['pipeline_done'] = True
            st.success("✓ Done")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# TimeSeries Preprocessing Lab")

if not uploaded:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:2.5rem;margin-bottom:0.5rem">📂</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.9rem;color:#94a3b8;">
            Upload a univariate time series CSV via the sidebar to get started.
        </div>
        <div style="margin-top:0.8rem;font-size:0.78rem;color:#475569;">
            Expected format: one datetime column + one numeric value column
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── After upload, show column preview ─────────────────────────────────────────
if not st.session_state['pipeline_done']:
    st.markdown('<div class="info-box">⚡ Configure preprocessing settings in the sidebar, then click <b>Run Preprocessing</b>.</div>', unsafe_allow_html=True)
    preview_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    st.dataframe(preview_df.head(10), use_container_width=True)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
raw     = st.session_state['raw']
imputed = st.session_state['imputed']
cleaned = st.session_state['outlier_cleaned']
denoised= st.session_state['denoised']
out_idx = st.session_state['outlier_indices']
residuals = st.session_state['residuals']
decomp    = st.session_state['decomp']

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊  Overview",
    "🔍  Missing Values",
    "⚡  Outliers",
    "🌊  Denoising",
    "📈  Final Result",
    "⚙️  Model Params",
    "🔮  Forecast",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    freq = detect_freq(raw)

    # KPI row
    miss_count = int(raw.isna().sum())
    out_count  = len(out_idx)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data Points",   f"{len(raw):,}")
    c2.metric("Missing",       f"{miss_count:,}",  f"{miss_count/len(raw)*100:.1f}%")
    c3.metric("Outliers Found",f"{out_count:,}",   f"{out_count/len(raw)*100:.1f}%")
    c4.metric("Inferred Freq", freq or "N/A")
    c5.metric("Time Span",     f"{(raw.index[-1]-raw.index[0]).days} days")

    st.markdown("---")

    # Raw series plot with missing markers
    miss_mask = raw.isna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=raw.index, y=raw,
        name='Original', line=dict(color=COLORS['original'], width=1.5),
    ))
    if miss_count > 0:
        miss_x = raw.index[miss_mask]
        miss_y = [raw.dropna().min()] * len(miss_x)
        fig.add_trace(go.Scatter(
            x=miss_x, y=miss_y, mode='markers',
            name='Missing', marker=dict(color='#ef4444', size=6, symbol='x'),
        ))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text='Raw Time Series', font=dict(size=13, color='#94a3b8')),
        height=300)
    st.plotly_chart(fig, width='stretch')

    # Distribution + ACF side by side
    st.markdown("---")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.plotly_chart(hist_fig(raw, 'Value Distribution'), width='stretch')
    with dc2:
        st.plotly_chart(acf_fig(raw.ffill().bfill(), title='ACF — Raw Series'), width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 2 · MISSING VALUES
# ════════════════════════════════════════════════════════════════════
with tab2:
    miss_count = int(raw.isna().sum())

    if miss_count == 0:
        st.markdown('<div class="info-box">✅ No missing values detected in this dataset.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ Found <b>{miss_count}</b> missing values ({miss_count/len(raw)*100:.2f}%). Imputation method: <b>{impute_method}</b></div>', unsafe_allow_html=True)

    # Compare all imputation methods
    methods = ['LOCF', 'NOCB', 'Moving Average', 'Linear']
    method_colors = [COLORS['locf'], COLORS['nocb'], COLORS['ma'], COLORS['linear']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw, name='Original',
        line=dict(color=COLORS['original'], width=1.5), opacity=0.5))
    for m, c in zip(methods, method_colors):
        imp = impute(raw, m)
        fig.add_trace(go.Scatter(x=imp.index, y=imp, name=m,
            line=dict(color=c, width=1.5)))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text='Imputation Methods Comparison', font=dict(size=13, color='#94a3b8')),
        height=340)
    st.plotly_chart(fig, width='stretch')

    # Stats comparison
    st.markdown("---")
    st.markdown('<div class="section-header">Before vs After Imputation — Stats</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        'Original': pd.Series(basic_stats(raw)),
        f'Imputed ({impute_method})': pd.Series(basic_stats(imputed)),
    }).astype(str)
    st.dataframe(comp, use_container_width=True)

    # Decomposition: original → trend → deseasonalized → seasonal → residuals
    if decomp is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">Time Series Decomposition (sp={}, degree={})</div>'.format(
            decomp_sp, decomp_degree), unsafe_allow_html=True)
        decomp_traces = [
            ('Original (Imputed)', imputed,                    COLORS['original'], 1.5),
            ('Trend',              decomp['trend'],             COLORS['ma'],       1.8),
            ('Deseasonalized',     decomp['deseasonalized'],    COLORS['linear'],   1.5),
            ('Seasonal',           decomp['seasonal'],          '#a78bfa',          1.5),
            ('Residuals',          decomp['residuals'],         COLORS['outlier'],  1.2),
        ]
        decomp_fig = go.Figure()
        for label, series_d, color, lw in decomp_traces:
            decomp_fig.add_trace(go.Scatter(
                x=series_d.index, y=series_d,
                name=label, line=dict(color=color, width=lw),
            ))
        decomp_fig.update_layout(**PLOT_LAYOUT,
            title=dict(text='Decomposition — Original / Trend / Deseasonalized / Seasonal / Residuals',
                       font=dict(size=13, color='#94a3b8')),
            height=380)
        st.plotly_chart(decomp_fig, width='stretch')
    else:
        st.markdown('<div class="warn-box">⚠️ Decomposition failed — check that the series has enough data points for the chosen seasonality period.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 · OUTLIERS
# ════════════════════════════════════════════════════════════════════
with tab3:
    out_count = len(out_idx)
    method_label = outlier_method

    if out_count == 0:
        st.markdown('<div class="info-box">✅ No outliers detected with current settings.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ Detected <b>{out_count}</b> outliers using <b>{method_label}</b>. They are replaced via linear interpolation.</div>', unsafe_allow_html=True)

    # Plot with outlier markers
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imputed.index, y=imputed,
        name='After Imputation', line=dict(color=COLORS['original'], width=1.5)))
    fig.add_trace(go.Scatter(x=cleaned.index, y=cleaned,
        name='Outlier Cleaned', line=dict(color=COLORS['cleaned'], width=1.5, dash='dot')))
    if out_idx:
        fig.add_trace(go.Scatter(
            x=out_idx, y=imputed[out_idx],
            name='Outliers', mode='markers',
            marker=dict(color=COLORS['outlier'], size=9, symbol='x', line=dict(width=2, color='white')),
        ))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=f'Outlier Detection — {method_label}', font=dict(size=13, color='#94a3b8')),
        height=340)
    st.plotly_chart(fig, width='stretch')

    # Outlier detail table
    if out_idx:
        st.markdown('<div class="section-header">Detected Outlier Details</div>', unsafe_allow_html=True)
        out_df = pd.DataFrame({
            'Timestamp': out_idx,
            'Outlier Value': [imputed[i] for i in out_idx],
            'Replaced With': [cleaned[i] for i in out_idx],
            'Deviation': [abs(imputed[i] - cleaned[i]) for i in out_idx],
        })
        st.dataframe(out_df.set_index('Timestamp'), use_container_width=True)

    # Residual plot — shown for both methods
    if residuals is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">Residual Decomposition</div>', unsafe_allow_html=True)

        resid_fig = go.Figure()

        if outlier_method == 'Hampel Filter':
            st.markdown('<div class="info-box">ℹ️ Hampel filter applied on residuals (deseasonalize → detrend). Shaded band shows the per-window ±n_sigma·MAD threshold envelope.</div>', unsafe_allow_html=True)
            # Per-point sliding-window MAD envelope — mirrors exactly what the filter checks
            k_mad = 1.4826
            half  = h_window // 2
            upper_env, lower_env = [], []
            for i in range(len(residuals)):
                lo = max(0, i - half)
                hi = min(len(residuals), i + half + 1)
                w_vals = residuals.iloc[lo:hi].dropna()
                if len(w_vals) < 2:
                    upper_env.append(np.nan)
                    lower_env.append(np.nan)
                else:
                    med = w_vals.median()
                    mad = k_mad * (w_vals - med).abs().median()
                    upper_env.append(med + h_nsigma * mad)
                    lower_env.append(med - h_nsigma * mad)
            resid_fig.add_trace(go.Scatter(
                x=list(residuals.index) + list(residuals.index[::-1]),
                y=upper_env + lower_env[::-1],
                fill='toself', fillcolor='rgba(239,68,68,0.08)',
                line=dict(color='rgba(0,0,0,0)'),
                name=f'±{h_nsigma}σ MAD envelope', hoverinfo='skip',
            ))
        else:
            st.markdown('<div class="info-box">ℹ️ Outliers detected on residuals after deseasonalizing & detrending (GESD).</div>', unsafe_allow_html=True)
            band_hi = residuals.std() * 3
            band_lo = -band_hi
            resid_fig.add_trace(go.Scatter(
                x=list(residuals.index) + list(residuals.index[::-1]),
                y=[band_hi] * len(residuals) + [band_lo] * len(residuals),
                fill='toself', fillcolor='rgba(239,68,68,0.08)',
                line=dict(color='rgba(0,0,0,0)'),
                name='±3σ band', hoverinfo='skip',
            ))

        resid_fig.add_trace(go.Scatter(
            x=residuals.index, y=residuals,
            name='Residuals', line=dict(color='#a78bfa', width=1.5),
        ))
        if out_idx:
            valid_resid_idx = [i for i in out_idx if i in residuals.index]
            if valid_resid_idx:
                resid_fig.add_trace(go.Scatter(
                    x=valid_resid_idx, y=residuals[valid_resid_idx],
                    name='Outliers', mode='markers',
                    marker=dict(color=COLORS['outlier'], size=9, symbol='x',
                                line=dict(width=2, color='white')),
                ))
        resid_fig.add_hline(y=0, line_color='#334155', line_width=1)
        resid_fig.update_layout(**PLOT_LAYOUT,
            title=dict(text='Residuals after Deseasonalizing & Detrending', font=dict(size=13, color='#94a3b8')),
            height=300)
        st.plotly_chart(resid_fig, width='stretch')

    # ACF of cleaned series
    st.markdown("---")
    st.plotly_chart(acf_fig(cleaned, title='ACF — After Outlier Removal'), width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 4 · DENOISING
# ════════════════════════════════════════════════════════════════════
with tab4:
    if denoise_method == 'None':
        st.markdown('<div class="info-box">ℹ️ Denoising is disabled. Select SMA, EMA, or FFT in the sidebar to activate.</div>', unsafe_allow_html=True)

    # Compare denoising methods (time domain)
    sma3  = denoise_sma(cleaned, window=3)
    sma7  = denoise_sma(cleaned, window=7)
    ema03 = denoise_ema(cleaned, alpha=0.3)
    fft_denoised, orig_mag, denoised_mag, fft_freqs, cutoff_freq = denoise_fft(cleaned, h_freq=fft_hfreq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cleaned.index, y=cleaned, name='Cleaned',
        line=dict(color=COLORS['original'], width=1.5), opacity=0.45))
    fig.add_trace(go.Scatter(x=sma3.index, y=sma3, name='SMA(3)',
        line=dict(color=COLORS['ma'], width=1.8)))
    fig.add_trace(go.Scatter(x=sma7.index, y=sma7, name='SMA(7)',
        line=dict(color='#f97316', width=1.8)))
    fig.add_trace(go.Scatter(x=ema03.index, y=ema03, name='EMA(α=0.3)',
        line=dict(color=COLORS['ema'], width=1.8)))
    if fft_denoised is not None:
        fig.add_trace(go.Scatter(x=fft_denoised.index, y=fft_denoised,
            name=f'FFT LPF (h_freq={fft_hfreq})',
            line=dict(color='#10b981', width=1.8)))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text='Time Domain — Denoising Methods Comparison', font=dict(size=13, color='#94a3b8')),
        height=340)
    st.plotly_chart(fig, width='stretch')

    # FFT: frequency domain view
    if orig_mag is not None and denoised_mag is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">Frequency Domain — Magnitude Spectrum</div>', unsafe_allow_html=True)
        freq_fig = go.Figure()
        freq_fig.add_trace(go.Scatter(
            x=fft_freqs, y=orig_mag,
            name='Before LPF', line=dict(color=COLORS['original'], width=1.5), opacity=0.6,
        ))
        freq_fig.add_trace(go.Scatter(
            x=fft_freqs, y=denoised_mag,
            name=f'After LPF (h_freq={fft_hfreq})', line=dict(color='#10b981', width=2),
        ))
        freq_fig.add_vline(
            x=cutoff_freq, line_color='#f59e0b',
            line_dash='dash', line_width=1.5,
            annotation_text=f'cutoff', annotation_font_color='#f59e0b',
        )
        freq_fig.update_layout(**PLOT_LAYOUT,
            title=dict(text='Frequency Domain — Original vs LPF Filtered', font=dict(size=13, color='#94a3b8')),
            xaxis_title='Frequency (cycles/sample)', yaxis_title='Amplitude', height=300,
        )
        st.plotly_chart(freq_fig, width='stretch')
    st.markdown("---")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.plotly_chart(acf_fig(cleaned, title='ACF — Before Denoising'), width='stretch')
    with ac2:
        st.plotly_chart(acf_fig(denoised, title='ACF — After Denoising'), width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 5 · FINAL RESULT
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Preprocessing Pipeline Summary")

    # Pipeline flow badges
    steps = [
        ("Raw Data",          f"{len(raw):,} pts",          "badge-blue"),
        ("Missing Imputed",   f"{impute_method}",            "badge-yellow"),
        ("Outliers Removed",  f"{len(out_idx)} found",       "badge-red"  if len(out_idx) else "badge-green"),
        ("Denoised",          denoise_method,                "badge-green"),
        ("Final Output",      f"{denoised.isna().sum()} NaN","badge-blue"),
    ]
    cols = st.columns(5)
    for col, (title, val, badge) in zip(cols, steps):
        col.markdown(f"""
        <div class="stat-card" style="text-align:center;">
            <div style="margin-bottom:0.4rem"><span class="badge {badge}">{title}</span></div>
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#e2e8f0;">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Overlay: raw vs final
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw, name='Original (Raw)',
        line=dict(color=COLORS['original'], width=1.2), opacity=0.4))
    fig.add_trace(go.Scatter(x=imputed.index, y=imputed, name='After Imputation',
        line=dict(color=COLORS['locf'], width=1.4), opacity=0.55))
    fig.add_trace(go.Scatter(x=cleaned.index, y=cleaned, name='After Outlier Removal',
        line=dict(color=COLORS['ma'], width=1.6)))
    if denoise_method != 'None':
        fig.add_trace(go.Scatter(x=denoised.index, y=denoised, name=f'Final (Denoised: {denoise_method})',
            line=dict(color=COLORS['cleaned'], width=2.2)))
    if out_idx:
        fig.add_trace(go.Scatter(x=out_idx, y=imputed[out_idx], name='Outliers',
            mode='markers', marker=dict(color=COLORS['outlier'], size=8, symbol='x')))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text='Full Pipeline — Stage Comparison', font=dict(size=13, color='#94a3b8')),
        height=380)
    st.plotly_chart(fig, width='stretch')

    # Stats: before vs after all stages
    st.markdown("---")
    st.markdown('<div class="section-header">Statistical Changes Through Pipeline</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        'Raw':           pd.Series(basic_stats(raw)),
        'Imputed':       pd.Series(basic_stats(imputed)),
        'Outlier Clean': pd.Series(basic_stats(cleaned)),
        'Final':         pd.Series(basic_stats(denoised)),
    }).astype(str)
    st.dataframe(comp_df, use_container_width=True)

    # Download
    st.markdown("---")
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    out_df = pd.DataFrame({
        'timestamp':      denoised.index,
        'value_raw':      raw.values,
        'value_imputed':  imputed.values,
        'value_cleaned':  cleaned.values,
        'value_final':    denoised.values,
    })
    csv_bytes = out_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇  Download Preprocessed CSV",
        data=csv_bytes,
        file_name="preprocessed_timeseries.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ════════════════════════════════════════════════════════════════════
# HELPERS — forecasting
# ════════════════════════════════════════════════════════════════════

def prep_series_for_sktime(series):
    """Ensure PeriodIndex for sktime — avoids <MonthBegin>/<MonthEnd> errors."""
    s = series.dropna().copy()
    freq_map = {'MS': 'M', 'ME': 'M', 'QS': 'Q', 'QE': 'Q',
                'YS': 'Y', 'YE': 'Y', 'AS': 'Y', 'A': 'Y'}

    if isinstance(s.index, pd.PeriodIndex):
        # Already a PeriodIndex — just return as-is
        return s

    # DatetimeIndex — read freq from the index attribute first
    idx = pd.DatetimeIndex(s.index)
    if idx.freq is not None:
        raw_freq = idx.freq.freqstr
    else:
        raw_freq = pd.infer_freq(idx) or 'M'

    freq = freq_map.get(raw_freq, raw_freq)
    s.index = idx.to_period(freq)
    return s


def compute_metrics(y_true, y_pred):
    mae  = MeanAbsoluteError().evaluate(y_true, y_pred)
    mse  = MeanSquaredError().evaluate(y_true, y_pred)
    mape = MeanAbsolutePercentageError().evaluate(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'MAPE (%)': mape * 100}


def run_holt(series, trend, smoothing_level, smoothing_trend):
    s = prep_series_for_sktime(series)
    y_train, y_test = temporal_train_test_split(s, test_size=36)
    model = ExponentialSmoothing(
        trend=trend,
        seasonal=None,
        smoothing_level=smoothing_level if smoothing_level > 0 else None,
        smoothing_trend=smoothing_trend if smoothing_trend > 0 else None,
    )
    model.fit(y_train)
    fh = list(range(1, len(y_test) + 1))
    y_pred = model.predict(fh=fh)
    y_pred.index = y_test.index
    metrics = compute_metrics(y_test, y_pred)
    return {'model': model, 'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'metrics': metrics}


def run_hw(series, trend, seasonal, smoothing_level, smoothing_trend, smoothing_seasonal, sp):
    s = prep_series_for_sktime(series)
    y_train, y_test = temporal_train_test_split(s, test_size=36)
    model = ExponentialSmoothing(
        trend=trend,
        seasonal=seasonal,
        sp=sp,
        smoothing_level=smoothing_level if smoothing_level > 0 else None,
        smoothing_trend=smoothing_trend if smoothing_trend > 0 else None,
        smoothing_seasonal=smoothing_seasonal if smoothing_seasonal > 0 else None,
    )
    model.fit(y_train)
    fh = list(range(1, len(y_test) + 1))
    y_pred = model.predict(fh=fh)
    y_pred.index = y_test.index
    metrics = compute_metrics(y_test, y_pred)
    return {'model': model, 'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'metrics': metrics}


def run_stl(series, sp):
    s = prep_series_for_sktime(series)
    y_train, y_test = temporal_train_test_split(s, test_size=36)
    model = STLForecaster(sp=sp)
    model.fit(y_train)
    fh = list(range(1, len(y_test) + 1))
    y_pred = model.predict(fh=fh)
    y_pred.index = y_test.index
    metrics = compute_metrics(y_test, y_pred)
    return {'model': model, 'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred, 'metrics': metrics}


def metrics_bar_fig(all_metrics, metric_name):
    models = list(all_metrics.keys())
    vals   = [all_metrics[m][metric_name] for m in models]
    colors = ['#00d4ff', '#7c3aed', '#10b981']
    fig = go.Figure(go.Bar(
        x=models, y=vals,
        marker_color=colors[:len(models)],
        text=[f"{v:.4f}" for v in vals],
        textposition='outside',
    ))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=metric_name, font=dict(size=13, color='#94a3b8')),
        height=280, showlegend=False,
    )
    return fig


def to_plot_index(series):
    """Convert PeriodIndex → DatetimeIndex so Plotly can serialize it."""
    if isinstance(series.index, pd.PeriodIndex):
        return series.to_timestamp()
    return series


# ════════════════════════════════════════════════════════════════════
# TAB 6 · MODEL PARAMS
# ════════════════════════════════════════════════════════════════════
with tab6:
    if not st.session_state['pipeline_done']:
        st.markdown('<div class="warn-box">⚠️ Run preprocessing first.</div>', unsafe_allow_html=True)
        st.stop()

    series_for_fc = prep_series_for_sktime(st.session_state['denoised'])

    st.markdown('<div class="section-header">Model Configuration & Evaluation (test_size = 36)</div>', unsafe_allow_html=True)

    # ── Holt ──────────────────────────────────────────────────────
    with st.expander("📘 Holt — Double Exponential Smoothing", expanded=True):
        hc1, hc2 = st.columns(2)
        with hc1:
            holt_trend = st.selectbox("Trend type", ['add', 'mul'], key='holt_trend')
            holt_alpha = st.slider("smoothing_level (α)", 0.0, 1.0, 0.0, step=0.05, key='holt_alpha')
        with hc2:
            holt_beta  = st.slider("smoothing_trend (β)", 0.0, 1.0, 0.0, step=0.05, key='holt_beta')
        if st.button("▶  Run Holt", key='run_holt'):
            with st.spinner("Fitting Holt…"):
                try:
                    res = run_holt(series_for_fc, holt_trend, holt_alpha, holt_beta)
                    st.session_state['holt_results'] = res
                    st.success(f"✓ MAE={res['metrics']['MAE']:.4f}  MSE={res['metrics']['MSE']:.4f}  MAPE={res['metrics']['MAPE (%)']:.2f}%")
                except Exception as e:
                    st.error(f"Holt failed: {e}")

    # ── Holt-Winters ───────────────────────────────────────────────
    with st.expander("📗 Holt-Winters — Triple Exponential Smoothing", expanded=True):
        wc1, wc2 = st.columns(2)
        with wc1:
            hw_trend    = st.selectbox("Trend type",    ['add', 'mul'], key='hw_trend')
            hw_seasonal = st.selectbox("Seasonal type", ['add', 'mul'], key='hw_seasonal')
        with wc2:
            hw_alpha = st.slider("smoothing_level (α)",   0.0, 1.0, 0.0, step=0.05, key='hw_alpha')
            hw_beta  = st.slider("smoothing_trend (β)",   0.0, 1.0, 0.0, step=0.05, key='hw_beta')
            hw_gamma = st.slider("smoothing_seasonal (γ)", 0.0, 1.0, 0.0, step=0.05, key='hw_gamma')
        if st.button("▶  Run Holt-Winters", key='run_hw'):
            with st.spinner("Fitting Holt-Winters…"):
                try:
                    res = run_hw(series_for_fc, hw_trend, hw_seasonal, hw_alpha, hw_beta, hw_gamma, decomp_sp)
                    st.session_state['hw_results'] = res
                    st.success(f"✓ MAE={res['metrics']['MAE']:.4f}  MSE={res['metrics']['MSE']:.4f}  MAPE={res['metrics']['MAPE (%)']:.2f}%")
                except Exception as e:
                    st.error(f"Holt-Winters failed: {e}")

    # ── STL ────────────────────────────────────────────────────────
    with st.expander("📙 STL Forecaster", expanded=True):
        stl_sp = st.slider("Seasonal period (sp)", 2, 52, 12, key='stl_sp')
        if st.button("▶  Run STL", key='run_stl'):
            with st.spinner("Fitting STL…"):
                try:
                    res = run_stl(series_for_fc, stl_sp)
                    st.session_state['stl_results'] = res
                    st.success(f"✓ MAE={res['metrics']['MAE']:.4f}  MSE={res['metrics']['MSE']:.4f}  MAPE={res['metrics']['MAPE (%)']:.2f}%")
                except Exception as e:
                    st.error(f"STL failed: {e}")

    # ── Metrics dashboard (shown once at least one model ran) ──────
    available = {
        k: v for k, v in {
            'Holt':         st.session_state['holt_results'],
            'Holt-Winters': st.session_state['hw_results'],
            'STL':          st.session_state['stl_results'],
        }.items() if v is not None
    }

    if available:
        st.markdown("---")
        st.markdown('<div class="section-header">Evaluation Dashboard</div>', unsafe_allow_html=True)

        all_metrics = {name: res['metrics'] for name, res in available.items()}

        # Metric bar charts
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.plotly_chart(metrics_bar_fig(all_metrics, 'MAE'), width='stretch')
        with mc2:
            st.plotly_chart(metrics_bar_fig(all_metrics, 'MSE'), width='stretch')
        with mc3:
            st.plotly_chart(metrics_bar_fig(all_metrics, 'MAPE (%)'), width='stretch')

        # Metrics table
        metrics_df = pd.DataFrame(all_metrics).T.round(4)
        st.dataframe(metrics_df, use_container_width=True)

        # Actual vs predicted overlay
        st.markdown("---")
        st.markdown('<div class="section-header">Test Set — Actual vs Predicted</div>', unsafe_allow_html=True)
        model_colors = {'Holt': '#00d4ff', 'Holt-Winters': '#7c3aed', 'STL': '#10b981'}
        fig = go.Figure()
        first = next(iter(available.values()))
        y_test_plot = to_plot_index(first['y_test'])
        fig.add_trace(go.Scatter(
            x=y_test_plot.index, y=y_test_plot,
            name='Actual', line=dict(color='#f59e0b', width=2),
        ))
        for name, res in available.items():
            y_pred_plot = to_plot_index(res['y_pred'])
            fig.add_trace(go.Scatter(
                x=y_pred_plot.index, y=y_pred_plot,
                name=f'{name} Pred', line=dict(color=model_colors[name], width=1.8, dash='dot'),
            ))
        fig.update_layout(**PLOT_LAYOUT,
            title=dict(text='Test Set Predictions', font=dict(size=13, color='#94a3b8')),
            height=360)
        st.plotly_chart(fig, width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 7 · FORECAST
# ════════════════════════════════════════════════════════════════════
with tab7:
    if not st.session_state['pipeline_done']:
        st.markdown('<div class="warn-box">⚠️ Run preprocessing first.</div>', unsafe_allow_html=True)
        st.stop()

    available_fc = {
        k: v for k, v in {
            'Holt':         st.session_state['holt_results'],
            'Holt-Winters': st.session_state['hw_results'],
            'STL':          st.session_state['stl_results'],
        }.items() if v is not None
    }

    if not available_fc:
        st.markdown('<div class="info-box">ℹ️ Run at least one model in the Model Params tab first.</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="section-header">Future Forecast</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        selected_model = st.selectbox("Select model", list(available_fc.keys()))
    with fc2:
        horizon = st.slider("Forecast horizon (months)", 1, 120, 12)

    if st.button("▶  Generate Forecast", use_container_width=True):
        with st.spinner("Forecasting…"):
            try:
                series_full = prep_series_for_sktime(st.session_state['denoised'])
                res = available_fc[selected_model]

                # Re-fit on full series using same params stored in the fitted model
                full_model = res['model'].clone()
                full_model.fit(series_full)
                fh = list(range(1, horizon + 1))
                y_future = full_model.predict(fh=fh)
                st.session_state['forecast_result'] = {
                    'model_name': selected_model,
                    'series': series_full,
                    'raw': st.session_state['raw'],      # ← add this
                    'forecast': y_future,
                    'horizon': horizon,
                }
            except Exception as e:
                st.error(f"Forecast failed: {e}")

    fc_res = st.session_state.get('forecast_result')
    if fc_res:
        st.markdown("---")
        model_colors = {'Holt': '#00d4ff', 'Holt-Winters': '#7c3aed', 'STL': '#10b981'}
        color = model_colors.get(fc_res['model_name'], '#00d4ff')

        fig = go.Figure()
        series_plot   = to_plot_index(fc_res['series'])
        forecast_plot = to_plot_index(fc_res['forecast'])
        raw_plot = fc_res['raw']                # already DatetimeIndex, no conversion needed
        fig.add_trace(go.Scatter(
            x=raw_plot.index, y=raw_plot,
            name='Original', line=dict(color='#94a3b8', width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_plot.index, y=forecast_plot,
            name=f"{fc_res['model_name']} Forecast ({fc_res['horizon']}m)",
            line=dict(color=color, width=2.5, dash='dot'),
        ))
        # vertical line at forecast start
        split_x = series_plot.index[-1]
        fig.add_vline(x=split_x, line_color='#334155', line_dash='dash', line_width=1)
        fig.update_layout(**PLOT_LAYOUT,
            title=dict(
                text=f"{fc_res['model_name']} — {fc_res['horizon']}-month Forecast",
                font=dict(size=13, color='#94a3b8'),
            ),
            height=420,
        )
        st.plotly_chart(fig, width='stretch')

        # Forecast table
        st.markdown('<div class="section-header">Forecast Values</div>', unsafe_allow_html=True)
        fc_df = pd.DataFrame({
            'Date':     forecast_plot.index,
            'Forecast': forecast_plot.values,
        }).set_index('Date').round(4)
        st.dataframe(fc_df, use_container_width=True)

        # Download
        csv_fc = fc_df.reset_index().to_csv(index=False).encode()
        st.download_button(
            label="⬇  Download Forecast CSV",
            data=csv_fc,
            file_name=f"forecast_{fc_res['model_name'].lower().replace('-','_')}_{fc_res['horizon']}m.csv",
            mime="text/csv",
            use_container_width=True,
        )
