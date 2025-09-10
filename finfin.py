
# === IMPORTACIONES Y ESTILO ===
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({'axes.facecolor': 'white', 'figure.facecolor': 'white', 'grid.color': '#cccccc', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'text.color': 'black'})
from scipy.special import fresnel
import pandas as pd
import io
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
SUPPORTED_LANGS = {'es': 'Español', 'en': 'English'}
TXT = {'en': {'a': 'a (m)',
        'a1': 'a1 (m)',
        'b': 'b (m)',
        'b1': 'b1 (m)',
        'csv_E': 'CSV (E)',
        'csv_H': 'CSV (H)',
        'cutoff_error': "TE₁₀ below cutoff: f_c = {fc:.3f} GHz. Increase f or 'a'.",
        'derived_params': 'Derived parameters',
        'freq': 'Frequency (GHz)',
        'geom': 'Aperture geometry and lengths',
        'help_a': 'Rectangular waveguide width (broad wall) of the feed.',
        'help_a1': 'Guide length at the aperture (H-plane dimension).',
        'help_b': 'Rectangular waveguide height (narrow wall) of the feed.',
        'help_b1': 'Guide length at the aperture (E-plane dimension).',
        'help_freq': 'Operating frequency in gigahertz.',
        'help_lE': 'ρ1 (≡ lE): effective distance to the imaginary apex in the E-plane; sets the quadratic phase at '
                   'the aperture.',
        'help_lH': 'ρ2 (≡ lH): effective distance to the imaginary apex in the H-plane; sets the quadratic phase at '
                   'the aperture.',
        'hero_title': 'Rectangular Aperture Antennas',
        'horn_type': 'Horn Type',
        'hover_enable': 'Use Plotly with hover',
        'hpbw_e': 'HPBW_E (°)',
        'hpbw_h': 'HPBW_H (°)',
        'interactivity': 'Interactivity',
        'lE': 'ρ1 ≡ lE (m)',
        'lH': 'ρ2 ≡ lH (m)',
        'language': 'Language',
        'metric_s': 's = b1^2/(8 λ ρ1)',
        'metric_t': 't = a1^2/(8 λ ρ2)',
        'mode': 'Mode',
        'mode_1d': '1D',
        'mode_polar': 'Polar',
        'opt_caption': 'Optimal points: s=0.25 (E) • t=0.375 (H).',
        'opt_design': 'Optimal design',
        'page_icon': '📡',
        'page_title': 'Horn Antenna Calculator',
        'plotly_missing': 'Plotly not available. Install: pip install plotly',
        'plots_title': 'Radiation patterns',
        'png_E': 'PNG (E)',
        'png_H': 'PNG (H)',
        'prec_help': 'Number of samples used in the aperture integral. Higher precision increases fidelity but uses '
                     'more CPU.',
        'prec_high': 'High ',
        'prec_normal': 'Normal ',
        'precision': 'Integral precision',
        'results': 'Results',
        's_opt': 'Optimal s',
        's_opt_help': 'Enforces b1=√(2 λ ρ1) while active',
        'scale_depth': 'Scale (dB)',
        'scale_header': 'Plot scale',
        'sel_DE_db': 'DE (dB)',
        'sel_DE_lin': 'DE (linear)',
        'sel_DH_db': 'DH (dB)',
        'sel_DH_lin': 'DH (linear)',
        'sel_Dp_db': 'Dp (dB)',
        'sel_Dp_lin': 'Dp (linear)',
        'settings': 'Settings',
        't_opt': 'Optimal t',
        't_opt_help': 'Enforces a1=√(3 λ ρ2) while active',
        'tabs_arbitrary': 'Fields at constant φ',
        'tabs_params': 'Parameters',
        'tabs_plots': 'E/H-plane fields',
        'tabs_univ': 'E/H-plane universal curves',
        'title_1dE': '1D Pattern — E-plane (dB)',
        'title_1dH': '1D Pattern — H-plane (dB)',
        'title_polE': 'Polar Pattern — E-plane (dB)',
        'title_polH': 'Polar Pattern — H-plane (dB)',
        'type_E': 'Sectoral (E-Plane)',
        'type_H': 'Sectoral (H-Plane)',
        'type_P': 'Pyramidal',
        'univ_E': 'E-plane (s)',
        'univ_H': 'H-plane (t)',
        'univ_kind': 'Curve',
        'univ_s_current': 'current s = {s:.5f}',
        'univ_t_current': 'current t = {t:.5f}',
        'univ_title': 'Universal curves',
        'univ_values_s': 's values (comma-separated)',
        'univ_values_t': 't values (comma-separated)',
        'univ_xlabel_E': 'x = (b1/λ)·sinθ',
        'univ_xlabel_H': 'x = (a1/λ)·sinθ',
        'univ_xmax': 'X-axis scale',
        'update_btn': 'Update plots',
        'warn_noD': 'Directivity could not be computed with the current parameters.',
        'wg_params': 'Waveguide lengths',
        'which_plane': 'Pattern plane',
        'xlabel_deg': 'θ (degrees)',
        'ylabel_norm': 'Level (dB)',
        'direct_lin': 'Directivity (linear)',
        'direct_db': 'Directivity (dB)'},
 'es': {'a': 'a (m)',
        'a1': 'a1 (m)',
        'b': 'b (m)',
        'b1': 'b1 (m)',
        'csv_E': 'CSV (E)',
        'csv_H': 'CSV (H)',
        'cutoff_error': "TE₁₀ por debajo de corte: f_c = {fc:.3f} GHz. Aumenta f o 'a'.",
        'derived_params': 'Parámetros derivados',
        'freq': 'Frecuencia (GHz)',
        'geom': 'Geometría y longitudes en la apertura',
        'help_a': 'Ancho de la guía rectangular (pared ancha) del alimentador.',
        'help_a1': 'Longitud de la guía en la apertura (plano H).',
        'help_b': 'Alto de la guía rectangular (pared estrecha) del alimentador.',
        'help_b1': 'Longitud de la guía en la apertura (plano E).',
        'help_freq': 'Frecuencia de operación en gigahercios.',
        'help_lE': 'ρ1 (≡ lE): distancia efectiva al ápice imaginario en el plano E; controla la fase cuadrática en la '
                   'apertura.',
        'help_lH': 'ρ2 (≡ lH): distancia efectiva al ápice imaginario en el plano H; controla la fase cuadrática en la '
                   'apertura.',
        'hero_title': 'Antenas de apertura rectangular',
        'horn_type': 'Tipo de Bocina',
        'hover_enable': 'Usar Plotly con hover',
        'hpbw_e': 'HPBW_E (°)',
        'hpbw_h': 'HPBW_H (°)',
        'interactivity': 'Interactividad',
        'lE': 'ρ1 ≡ lE (m)',
        'lH': 'ρ2 ≡ lH (m)',
        'language': 'Idioma',
        'metric_s': 's = b1^2/(8 λ ρ1)',
        'metric_t': 't = a1^2/(8 λ ρ2)',
        'mode': 'Modo',
        'mode_1d': '1D',
        'mode_polar': 'Polar',
        'opt_caption': 'Óptimos: s=0.25 (E) • t=0.375 (H).',
        'opt_design': 'Diseño óptimo',
        'page_icon': '📡',
        'page_title': 'Calculadora de Antenas de Bocina',
        'plotly_missing': 'Plotly no está disponible. Instálalo con: pip install plotly',
        'plots_title': 'Diagramas de radiación',
        'png_E': 'PNG (E)',
        'png_H': 'PNG (H)',
        'prec_help': 'Número de muestras usadas en la integral de apertura. Alta precisión mejora fidelidad pero '
                     'consume más CPU.',
        'prec_high': 'Alta ',
        'prec_normal': 'Normal ',
        'precision': 'Precisión de la integral',
        'results': 'Resultados',
        's_opt': 's óptimo',
        's_opt_help': 'Impone b1=√(2 λ ρ1) mientras esté activo',
        'scale_depth': 'Escala (dB)',
        'scale_header': 'Escala de los diagramas',
        'sel_DE_db': 'DE (dB)',
        'sel_DE_lin': 'DE (lineal)',
        'sel_DH_db': 'DH (dB)',
        'sel_DH_lin': 'DH (lineal)',
        'sel_Dp_db': 'Dp (dB)',
        'sel_Dp_lin': 'Dp (lineal)',
        'settings': 'Ajustes',
        't_opt': 't óptimo',
        't_opt_help': 'Impone a1=√(3 λ ρ2) mientras esté activo',
        'tabs_arbitrary': 'Campos para φ constante',
        'tabs_params': 'Parámetros',
        'tabs_plots': 'Campos plano E/H',
        'tabs_univ': 'Curvas universales plano E/H',
        'title_1dE': 'Patrón 1D — Plano E (dB)',
        'title_1dH': 'Patrón 1D — Plano H (dB)',
        'title_polE': 'Patrón polar — Plano E (dB)',
        'title_polH': 'Patrón polar — Plano H (dB)',
        'type_E': 'Sectorial (Plano E)',
        'type_H': 'Sectorial (Plano H)',
        'type_P': 'Piramidal',
        'univ_E': 'Plano E (s)',
        'univ_H': 'Plano H (t)',
        'univ_kind': 'Curva',
        'univ_s_current': 's actual = {s:.5f}',
        'univ_t_current': 't actual = {t:.5f}',
        'univ_title': 'Curvas universales',
        'univ_values_s': 'Valores de s (coma-separados)',
        'univ_values_t': 'Valores de t (coma-separados)',
        'univ_xlabel_E': 'x = (b1/λ)·sinθ',
        'univ_xlabel_H': 'x = (a1/λ)·sinθ',
        'univ_xmax': 'Escala eje X',
        'update_btn': 'Actualizar gráficas',
        'warn_noD': 'No se pudo calcular la directividad con los parámetros actuales.',
        'wg_params': 'Longitudes de la guía',
        'which_plane': 'Plano para patrón',
        'xlabel_deg': 'θ (grados)',
        'ylabel_norm': 'Nivel (dB)',
        'direct_lin': 'Directividad (lineal)',
        'direct_db': 'Directividad (dB)'}}


# === UTILIDAD DE TEXTOS (TRADUCCIONES) ===
def _T(lang, key):
    return TXT.get(lang, TXT['en']).get(key, TXT['en'].get(key, key))

# === CONFIGURACIÓN DE LA PÁGINA ===
st.set_page_config(page_title=TXT['es']['page_title'], page_icon=TXT['es']['page_icon'], layout='wide')

# === ESTILOS GLOBALES (CSS) ===
st.markdown('\n    <style>\n    html, body, .stApp { background: #0b1220; color:#e5e7eb; }\n    .hero{background: linear-gradient(135deg,#0f172a,#0b1220 60%);border:1px solid #1f2937;border-radius:20px;padding:18px 22px;box-shadow:0 1px 2px rgba(0,0,0,.25);margin-bottom:0.75rem;}\n    .card{background:#111827;border-radius:16px;padding:1rem 1.25rem;border:1px solid #1f2937;box-shadow:0 1px 2px rgba(0,0,0,0.25);margin-bottom:0.75rem;}\n    .metric-big{font-size:2.0rem; font-weight:800; letter-spacing:-0.01em}\n    </style>\n    ', unsafe_allow_html=True)


# === GESTIÓN DE ESTADO POR DEFECTO ===
def _set_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v
try:
    qp = st.query_params
    lang_from_url = qp.get('lang', None)
except Exception:
    try:
        qp = st.experimental_get_query_params()
        lang_from_url = qp.get('lang', [None])[0]
    except Exception:
        lang_from_url = None
_set_default('lang', lang_from_url if lang_from_url in SUPPORTED_LANGS else 'es')
_set_default('tipo', 'Sectorial (Plano E)')
_set_default('fGHz', 10.0)
_set_default('a1', 0.36)
_set_default('b1', 0.083)
_set_default('lE', 0.18)
_set_default('lH', 0.18)
_set_default('a', 0.015)
_set_default('b', 0.0075)
_set_default('s_opt_on', False)
_set_default('t_opt_on', False)

# === PANEL LATERAL — ENTRADA DE PARÁMETROS ===
with st.sidebar:
    st.header(_T(st.session_state.lang, 'settings'))
    lang_name_to_code = {v: k for k, v in SUPPORTED_LANGS.items()}
    lang_choice = st.selectbox(_T(st.session_state.lang, 'language') + ' / ' + _T('en', 'language'), list(SUPPORTED_LANGS.values()), index=list(SUPPORTED_LANGS.keys()).index(st.session_state.lang))
    st.session_state.lang = lang_name_to_code.get(lang_choice, 'es')
    try:
        st.query_params['lang'] = st.session_state.lang
    except Exception:
        try:
            st.experimental_set_query_params(lang=st.session_state.lang)
        except Exception:
            pass
    tipo_map = {'Sectorial (Plano E)': _T(st.session_state.lang, 'type_E'), 'Sectorial (Plano H)': _T(st.session_state.lang, 'type_H'), 'Piramidal': _T(st.session_state.lang, 'type_P')}
    tipo_inv = {v: k for k, v in tipo_map.items()}
    tipo_label = st.selectbox(_T(st.session_state.lang, 'horn_type'), list(tipo_map.values()), index=list(tipo_map.keys()).index(st.session_state.tipo))
    st.session_state.tipo = tipo_inv[tipo_label]
    st.number_input(_T(st.session_state.lang, 'freq'), min_value=10.0, step=0.1, format='%.3f', key='fGHz', help=_T(st.session_state.lang, 'help_freq'))
    st.subheader(_T(st.session_state.lang, 'geom'))
    if st.session_state.tipo == 'Sectorial (Plano E)':
        st.number_input(_T(st.session_state.lang, 'b1'), min_value=0.0825, step=0.001, format='%.4f', key='b1', disabled=st.session_state.get('s_opt_on', False), help=_T(st.session_state.lang, 'help_b1'))
    elif st.session_state.tipo == 'Sectorial (Plano H)':
        st.number_input(_T(st.session_state.lang, 'a1'), min_value=0.0, step=0.001, format='%.6f', key='a1', disabled=st.session_state.get('t_opt_on', False), help=_T(st.session_state.lang, 'help_a1'))
    else:
        st.number_input(_T(st.session_state.lang, 'a1'), min_value=0.0, step=0.001, format='%.6f', key='a1', disabled=st.session_state.get('t_opt_on', False), help=_T(st.session_state.lang, 'help_a1'))
        st.number_input(_T(st.session_state.lang, 'b1'), min_value=0.0825, step=0.001, format='%.4f', key='b1', disabled=st.session_state.get('s_opt_on', False), help=_T(st.session_state.lang, 'help_b1'))
    st.number_input(_T(st.session_state.lang, 'lE'), min_value=0.18, step=0.001, format='%.4f', key='lE', help=_T(st.session_state.lang, 'help_lE'))
    st.number_input(_T(st.session_state.lang, 'lH'), min_value=0.0, step=0.001, format='%.6f', key='lH', help=_T(st.session_state.lang, 'help_lH'))
    st.subheader(_T(st.session_state.lang, 'wg_params'))
    st.number_input(_T(st.session_state.lang, 'a'), min_value=0.015, step=0.001, format='%.4f', key='a', help=_T(st.session_state.lang, 'help_a'))
    st.number_input(_T(st.session_state.lang, 'b'), min_value=0.0075, step=0.001, format='%.4f', key='b', help=_T(st.session_state.lang, 'help_b'))
    st.subheader(_T(st.session_state.lang, 'opt_design'))
    c1, c2 = st.columns(2)
    with c1:
        st.toggle(_T(st.session_state.lang, 's_opt'), key='s_opt_on', help=_T(st.session_state.lang, 's_opt_help'))
    with c2:
        st.toggle(_T(st.session_state.lang, 't_opt'), key='t_opt_on', help=_T(st.session_state.lang, 't_opt_help'))

# === CONSTANTES FÍSICAS Y PARÁMETROS DERIVADOS ===
c0 = 299792458.0
lam = c0 / (st.session_state.fGHz * 1000000000.0)
k = 2.0 * np.pi / lam if lam > 0 else 0.0
b1_eff = float(np.sqrt(2.0 * lam * st.session_state.lE)) if st.session_state.get('s_opt_on', False) and st.session_state.lE > 0 and (lam > 0) else st.session_state.b1
a1_eff = float(np.sqrt(3.0 * lam * st.session_state.lH)) if st.session_state.get('t_opt_on', False) and st.session_state.lH > 0 and (lam > 0) else st.session_state.a1
fc = c0 / (2.0 * st.session_state.a) / 1000000000.0 if st.session_state.a > 0 else np.inf
if st.session_state.fGHz <= fc:
    st.error(_T(st.session_state.lang, 'cutoff_error').format(fc=fc))


# === FUNCIONES AUXILIARES (FRESNEL) ===
def CS_pair(x: float):
    Sx, Cx = fresnel(x)
    return (Cx, Sx)


# === CÁLCULO DE DIRECTIVIDAD — BOCINA SECTORIAL E ===
def compute_DE(lam, a, b1, rho1):
    if lam <= 0 or a <= 0 or b1 <= 0 or (rho1 <= 0):
        return np.nan
    v = b1 / np.sqrt(2.0 * lam * rho1)
    C, S = CS_pair(v)
    return 64.0 * a * rho1 / (np.pi * lam * b1) * (C ** 2 + S ** 2)


# === CÁLCULO DE DIRECTIVIDAD — BOCINA SECTORIAL H ===
def compute_DH(lam, a1, b, rho2):
    if lam <= 0 or a1 <= 0 or b <= 0 or (rho2 <= 0):
        return np.nan
    sr = np.sqrt(lam * rho2)
    u = (sr / a1 + a1 / sr) / np.sqrt(2.0)
    v = (sr / a1 - a1 / sr) / np.sqrt(2.0)
    Cu, Su = CS_pair(u)
    Cv, Sv = CS_pair(v)
    return 4.0 * np.pi * b * rho2 / (lam * a1) * ((Cu - Cv) ** 2 + (Su - Sv) ** 2)

# === SELECCIÓN Y PRESENTACIÓN DE LA DIRECTIVIDAD ===
DE = compute_DE(lam, st.session_state.a, b1_eff, st.session_state.lE)
DH = compute_DH(lam, a1_eff, st.session_state.b, st.session_state.lH)
Dp = np.nan
if st.session_state.tipo == 'Piramidal' and np.isfinite(DE) and np.isfinite(DH) and (st.session_state.a > 0) and (st.session_state.b > 0):
    Dp = np.pi * lam ** 2 / (32.0 * st.session_state.a * st.session_state.b) * DE * DH
if st.session_state.tipo == 'Sectorial (Plano E)':
    Dsel = DE
    etiqueta_lin = _T(st.session_state.lang, 'sel_DE_lin')
    etiqueta_db = _T(st.session_state.lang, 'sel_DE_db')
elif st.session_state.tipo == 'Sectorial (Plano H)':
    Dsel = DH
    etiqueta_lin = _T(st.session_state.lang, 'sel_DH_lin')
    etiqueta_db = _T(st.session_state.lang, 'sel_DH_db')
else:
    Dsel = Dp
    etiqueta_lin = _T(st.session_state.lang, 'sel_Dp_lin')
    etiqueta_db = _T(st.session_state.lang, 'sel_Dp_db')

# Unificar rótulos de directividad en todas las antenas
etiqueta_lin = _T(st.session_state.lang, 'direct_lin')
etiqueta_db = _T(st.session_state.lang, 'direct_db')
st.title(_T(st.session_state.lang, 'hero_title'))
st.subheader(_T(st.session_state.lang, 'results'))
if not np.isfinite(Dsel) or Dsel <= 0:
    st.warning(_T(st.session_state.lang, 'warn_noD'))
else:
    D_db = 10.0 * np.log10(Dsel)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**' + etiqueta_lin + '**')
        st.markdown(f"<div class='metric-big'>{Dsel:.3f}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('**' + etiqueta_db + '**')
        st.markdown(f"<div class='metric-big'>{D_db:.2f} dB</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# === CÁLCULO EXACTO DEL PATRÓN 1D (PLANOS E/H) ===
@st.cache_data(show_spinner=False)
def compute_pattern_1d(plane: str, lam: float, a1_eff: float, b1_eff: float, lE: float, lH: float, N: int):
    """
    Cortes principales EXACTOS (según referencia) (Balanis, 3rd ed.):
    - E-plane sectorial (φ = π/2): Eθ ∝ (1+cosθ) * (2/π)^2 * e^{j(kρ1 sin^2θ /2)} * F(t1',t2')  [ec. (13-12b)]
      t1' = sqrt(k/(πρ1))*(-b1/2 - ρ1 sinθ),  t2' = sqrt(k/(πρ1))*(+b1/2 - ρ1 sinθ).
    - H-plane sectorial (φ = 0):   Eφ ∝ (1+cosθ) * [ e^{jf1}F(t1',t2') + e^{jf2}F(t1'',t2'') ]  [ec. (13-32b)]
      f1 = kx'^2 ρ2/(2k), kx' = k sinθ + π/a1;   f2 = kx''^2 ρ2/(2k), kx'' = k sinθ − π/a1.
      t1',t2' per (13-26a,b) con kx' y  t1'',t2'' per (13-27a,b) con kx''.
    Devolvemos |E|^2 normalizado (las constantes globales se cancelan).
    """
    import numpy as _np
    from scipy.special import fresnel as _fresnel
    theta_deg = _np.linspace(-90.0, 90.0, 721)
    th = _np.deg2rad(theta_deg)
    k = 2.0 * _np.pi / lam if lam > 0 else 0.0
    if plane == 'E':
        if not (b1_eff > 0 and lE > 0 and (lam > 0)):
            P = _np.zeros_like(th)
        else:
            rho1 = lE
            b1 = b1_eff
            t1 = _np.sqrt(k / (_np.pi * rho1)) * (-b1 / 2.0 - rho1 * _np.sin(th))
            t2 = _np.sqrt(k / (_np.pi * rho1)) * (+b1 / 2.0 - rho1 * _np.sin(th))
            S1, C1 = _fresnel(t1)
            S2, C2 = _fresnel(t2)
            F = C2 - C1 - 1j * (S2 - S1)
            phase = _np.exp(1j * (k * rho1 * _np.sin(th) ** 2 / 2.0))
            Etheta = (2.0 / _np.pi) ** 2 * (1.0 + _np.cos(th)) * phase * F
            P = _np.abs(Etheta) ** 2
    elif not (a1_eff > 0 and lH > 0 and (lam > 0)):
        P = _np.zeros_like(th)
    else:
        rho2 = lH
        a1 = a1_eff
        kx_p = k * _np.sin(th) + _np.pi / a1
        kx_m = k * _np.sin(th) - _np.pi / a1
        f1 = kx_p ** 2 * rho2 / (2.0 * k)
        f2 = kx_m ** 2 * rho2 / (2.0 * k)
        t1p = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (-k * a1 / 2.0 - kx_p * rho2)
        t2p = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (+k * a1 / 2.0 - kx_p * rho2)
        t1m = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (-k * a1 / 2.0 - kx_m * rho2)
        t2m = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (+k * a1 / 2.0 - kx_m * rho2)
        S1p, C1p = _fresnel(t1p)
        S2p, C2p = _fresnel(t2p)
        Fp = C2p - C1p - 1j * (S2p - S1p)
        S1m, C1m = _fresnel(t1m)
        S2m, C2m = _fresnel(t2m)
        Fm = C2m - C1m - 1j * (S2m - S1m)
        Ephi = (1.0 + _np.cos(th)) * (_np.exp(1j * f1) * Fp + _np.exp(1j * f2) * Fm)
        P = _np.abs(Ephi) ** 2
    Pmax = P.max() if P.size and _np.max(P) > 0 else 1.0
    return (theta_deg, P / Pmax)


# === ANCHO DE HAZ A -3 DB (HPBW) ===
def _hpbw_from_db(x_deg, y_db):
    import numpy as _np
    if len(x_deg)==0 or len(y_db)==0: return float('nan')
    y = _np.array(y_db)
    x = _np.array(x_deg)
    i0 = int(_np.nanargmax(y))
    y_rel = y - y[i0]
    # buscar cruces -3 dB a cada lado del máximo
    def cross(idx_iter):
        prev_x = None; prev_y = None
        for k in idx_iter:
            if prev_y is not None and ((prev_y > -3 and y_rel[k] <= -3) or (prev_y < -3 and y_rel[k] >= -3)):
                # interpolación lineal
                x1, x2 = prev_x, x[k]; y1, y2 = prev_y, y_rel[k]
                if y2==y1: return x2
                return x1 + ( -3 - y1) * (x2 - x1) / (y2 - y1)
            prev_x, prev_y = x[k], y_rel[k]
        return _np.nan
    left = cross(range(i0, 0, -1))
    right = cross(range(i0, len(x)))
    if _np.isnan(left) or _np.isnan(right): return float('nan')
    return float(abs(right - left))

# === Añadido: auxiliar para reutilizar el método 1D exacto para HPBW en polares ===

# === HPBW COHERENTE CON MÉTODO 1D ===
def hpbw_from_1d_same_method(plane: str, lam: float, a1_eff: float, b1_eff: float, lE: float, lH: float, N: int = 4096) -> float:
    # Compute HPBW using the exact same method used in the 1D plots,
    # so the value shown in polar plots matches 1D exactly.
    th_deg, patt = compute_pattern_1d(plane, lam, a1_eff, b1_eff, lE, lH, max(int(N), 2048))
    return _hpbw_from_db(th_deg, patt_to_db(patt))
# === /Added ===


# === HPBW NUMÉRICO (MÉTODO ALTERNATIVO) ===
def _hpbw_numeric(plane, lam, a1_eff, b1_eff, lE, lH):
    import numpy as _np
    th_deg, P = compute_pattern_1d(plane, lam, a1_eff, b1_eff, lE, lH, 1024)
    mask = th_deg >= 0.0
    th = th_deg[mask]
    PP = P[mask]
    if not PP.size or PP.max() <= 0:
        return float('nan')
    idx = _np.where(PP < 0.5)[0]
    if idx.size == 0:
        return float('nan')
    i = int(idx[0])
    if i == 0:
        return float('nan')
    x1, x2 = (th[i - 1], th[i])
    y1, y2 = (PP[i - 1], PP[i])
    th3 = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
    return 2.0 * float(th3)


# === CURVA DE REFERENCIA PARA CORTES PRINCIPALES ===
@st.cache_data(show_spinner=False)
def compute_overlay_at_fixed_phi(plane: str, lam: float, a1_eff: float, b1_eff: float, lE: float, lH: float, a_feed: float, b_feed: float, N: int):
    """Curva de referencia para los cortes principales usando la *misma física* que 'cortes arbitrarios':
    if plane=='E' -> use fields_sectorial_E at φ=0°; if plane=='H' -> fields_sectorial_H at φ=90°.
    Returns (theta_deg, P_norm)."""
    import numpy as _np
    th_deg = _np.linspace(-90.0, 90.0, max(N, 361))
    th = _np.deg2rad(th_deg)
    if plane == 'E':
        ph = 0.0  # 0°
        Et, Ep = fields_sectorial_E(th, ph, lam, b1_eff, lE, a_feed)
    else:
        ph = _np.deg2rad(90.0)  # 90°
        Et, Ep = fields_sectorial_H(th, ph, lam, a1_eff, lH, b_feed)
    P = _np.abs(Et)**2 + _np.abs(Ep)**2
    P /= P.max() if P.size and _np.max(P) > 0 else 1.0
    return th_deg, P


# === PATRÓN POLAR (0–360°) ===
@st.cache_data(show_spinner=False)
def compute_pattern_polar(plane: str, lam: float, a1_eff: float, b1_eff: float, lE: float, lH: float, N: int):
    """
    Versión polar de los cortes principales, idéntica a compute_pattern_1d pero barriendo 0..360°.
    """
    import numpy as _np
    from scipy.special import fresnel as _fresnel
    phi = _np.linspace(0.0, 360.0, 1000)
    th = _np.deg2rad(phi)
    k = 2.0 * _np.pi / lam if lam > 0 else 0.0
    if plane == 'E':
        if not (b1_eff > 0 and lE > 0 and (lam > 0)):
            P = _np.zeros_like(th)
        else:
            rho1 = lE
            b1 = b1_eff
            t1 = _np.sqrt(k / (_np.pi * rho1)) * (-b1 / 2.0 - rho1 * _np.sin(th))
            t2 = _np.sqrt(k / (_np.pi * rho1)) * (+b1 / 2.0 - rho1 * _np.sin(th))
            S1, C1 = _fresnel(t1)
            S2, C2 = _fresnel(t2)
            F = C2 - C1 - 1j * (S2 - S1)
            phase = _np.exp(1j * (k * rho1 * _np.sin(th) ** 2 / 2.0))
            Etheta = (2.0 / _np.pi) ** 2 * (1.0 + _np.cos(th)) * phase * F
            P = _np.abs(Etheta) ** 2
    elif not (a1_eff > 0 and lH > 0 and (lam > 0)):
        P = _np.zeros_like(th)
    else:
        rho2 = lH
        a1 = a1_eff
        kx_p = k * _np.sin(th) + _np.pi / a1
        kx_m = k * _np.sin(th) - _np.pi / a1
        f1 = kx_p ** 2 * rho2 / (2.0 * k)
        f2 = kx_m ** 2 * rho2 / (2.0 * k)
        t1p = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (-k * a1 / 2.0 - kx_p * rho2)
        t2p = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (+k * a1 / 2.0 - kx_p * rho2)
        t1m = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (-k * a1 / 2.0 - kx_m * rho2)
        t2m = _np.sqrt(1.0 / (_np.pi * k * rho2)) * (+k * a1 / 2.0 - kx_m * rho2)
        S1p, C1p = _fresnel(t1p)
        S2p, C2p = _fresnel(t2p)
        Fp = C2p - C1p - 1j * (S2p - S1p)
        S1m, C1m = _fresnel(t1m)
        S2m, C2m = _fresnel(t2m)
        Fm = C2m - C1m - 1j * (S2m - S1m)
        Ephi = (1.0 + _np.cos(th)) * (_np.exp(1j * f1) * Fp + _np.exp(1j * f2) * Fm)
        P = _np.abs(Ephi) ** 2
    Pmax = P.max() if P.size and _np.max(P) > 0 else 1.0
    return (phi, P / Pmax)
import numpy as _np


# === UTILIDADES DE CONVERSIÓN Y CURVAS UNIVERSALES ===
def patt_to_db(p):
    return 10.0 * np.log10(np.clip(p, 1e-12, 1.0))

def _univ_E_db(s, x_vec):
    import numpy as _np
    from scipy.special import fresnel as _fresnel
    x_vec = _np.asarray(x_vec, dtype=float)
    s = float(s)
    rt = _np.sqrt(max(s, 1e-16))
    t1 = 2 * rt * (-1.0 - x_vec / (4.0 * max(s, 1e-16)))
    t2 = 2 * rt * (+1.0 - x_vec / (4.0 * max(s, 1e-16)))
    S1, C1 = _fresnel(t1)
    S2, C2 = _fresnel(t2)
    F = C2 - C1 - 1j * (S2 - S1)
    mag = _np.abs(F)
    mag /= mag.max() if mag.max() > 0 else 1.0
    return 20.0 * _np.log10(_np.maximum(mag, 1e-12))

def _univ_H_db(t, x_vec):
    """Curvas universales para bocina sectorial en plano H (Balanis 3e, eq. 13-33).
    Entradas: t = a1^2/(8 λ ρ2), x_vec = (a1/λ)*sinθ
    Salida: dB, normalized to 0 dB max (excludes (1+cosθ) factor).
    """
    import numpy as _np
    from scipy.special import fresnel as _fresnel
    x_vec = _np.asarray(x_vec, dtype=float)
    t = float(t)
    t_safe = max(t, 1e-16)
    rt = _np.sqrt(t_safe)
    inv_t = 1.0 / t_safe
    t1p = 2 * rt * (-1.0 - x_vec * inv_t / 4.0 - inv_t / 8.0)
    t2p = 2 * rt * (+1.0 - x_vec * inv_t / 4.0 - inv_t / 8.0)
    t1m = 2 * rt * (-1.0 - x_vec * inv_t / 4.0 + inv_t / 8.0)
    t2m = 2 * rt * (+1.0 - x_vec * inv_t / 4.0 + inv_t / 8.0)
    eps = 1e-09
    x_safe = _np.where(_np.abs(x_vec) < eps, _np.sign(x_vec) * eps + eps, x_vec)
    bracket_p = 1.0 + 0.5 / x_safe
    bracket_m = 1.0 - 0.5 / x_safe
    f1 = _np.pi / 8.0 * inv_t * x_vec ** 2 * bracket_p ** 2
    f2 = _np.pi / 8.0 * inv_t * x_vec ** 2 * bracket_m ** 2
    S1p, C1p = _fresnel(t1p)
    S2p, C2p = _fresnel(t2p)
    S1m, C1m = _fresnel(t1m)
    S2m, C2m = _fresnel(t2m)
    Fp = C2p - C1p - 1j * (S2p - S1p)
    Fm = C2m - C1m - 1j * (S2m - S1m)
    E = _np.exp(1j * f1) * Fp + _np.exp(1j * f2) * Fm
    mag = _np.abs(E)
    mag /= mag.max() if mag.max() > 0 else 1.0
    return 20.0 * _np.log10(_np.maximum(mag, 1e-12))


# === FRESNEL: FUNCIÓN F(T1,T2) ===
def fresnel_F(t1, t2):
    S1, C1 = fresnel(t1)
    S2, C2 = fresnel(t2)
    return C2 - C1 - 1j * (S2 - S1)


# === SINC SEGURA ===
def sinc_safe(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / x)


# === CAMPO LEJANO — BOCINA SECTORIAL E (Φ ARBITRARIO) ===
def fields_sectorial_E(theta, phi, lam, b1, rho1, a):
    """
    Campo lejano para bocina sectorial E con φ arbitrario.
    Se reduce al plano E principal cuando φ=90° (término sinc → 1).
    """
    th = theta
    ph = phi
    k = 2 * np.pi / lam
    sinth = np.sin(th)
    costh = np.cos(th)
    sinph = np.sin(ph)
    cosph = np.cos(ph)
    t1 = np.sqrt(k / (np.pi * rho1)) * (-b1 / 2.0 - rho1 * sinth * sinph)
    t2 = np.sqrt(k / (np.pi * rho1)) * (+b1 / 2.0 - rho1 * sinth * sinph)
    F = fresnel_F(t1, t2)
    phase = np.exp(1j * (k * rho1 * (sinth * sinph) ** 2 / 2.0))
    X = k * a / 2.0 * sinth * cosph
    S = sinc_safe(X)
    A = (2.0 / np.pi) ** 2 * (1.0 + costh) * S * phase * F
    Etheta = A * np.sin(ph)
    Ephi = A * np.cos(ph)
    return (Etheta, Ephi)


# === CAMPO LEJANO — BOCINA SECTORIAL H (Φ ARBITRARIO) ===
def fields_sectorial_H(theta, phi, lam, a1, rho2, b):
    """
    Campo lejano para bocina sectorial H con φ arbitrario.
    Se reduce al plano H principal cuando φ=0° (término sinc → 1).
    """
    th = theta
    ph = phi
    k = 2 * np.pi / lam
    sinth = np.sin(th)
    costh = np.cos(th)
    sinph = np.sin(ph)
    cosph = np.cos(ph)
    kx_p = k * sinth * cosph + np.pi / a1
    kx_m = k * sinth * cosph - np.pi / a1
    f1 = kx_p ** 2 * rho2 / (2.0 * k)
    f2 = kx_m ** 2 * rho2 / (2.0 * k)
    alpha = np.sqrt(1.0 / (np.pi * k * rho2))
    t1p = alpha * (-k * a1 / 2.0 - kx_p * rho2)
    t2p = alpha * (+k * a1 / 2.0 - kx_p * rho2)
    t1m = alpha * (-k * a1 / 2.0 - kx_m * rho2)
    t2m = alpha * (+k * a1 / 2.0 - kx_m * rho2)
    Fp = fresnel_F(t1p, t2p)
    Fm = fresnel_F(t1m, t2m)
    B = np.exp(1j * f1) * Fp + np.exp(1j * f2) * Fm
    Y = k * b / 2.0 * sinth * sinph
    S = sinc_safe(Y)
    A = (1.0 + costh) * S * B
    Etheta = A * np.sin(ph)
    Ephi = A * np.cos(ph)
    return (Etheta, Ephi)


# === PATRÓN PARA Φ ARBITRARIO ===
def pattern_arbitrary(tipo, lam, a1, b1, lE, lH, a, b, phi_deg, N=721):
    th_deg = np.linspace(-90.0, 90.0, N)
    th = np.deg2rad(th_deg)
    ph = np.deg2rad(phi_deg)
    if tipo == 'Sectorial (Plano E)':
        if lam <= 0 or b1 <= 0 or lE <= 0 or (a <= 0):
            P = np.zeros_like(th)
        else:
            Et, Ep = fields_sectorial_E(th, ph, lam, b1, lE, a)
            P = np.abs(Et) ** 2 + np.abs(Ep) ** 2
    elif tipo == 'Sectorial (Plano H)':
        if lam <= 0 or a1 <= 0 or lH <= 0 or (b <= 0):
            P = np.zeros_like(th)
        else:
            Et, Ep = fields_sectorial_H(th, ph, lam, a1, lH, b)
            P = np.abs(Et) ** 2 + np.abs(Ep) ** 2
    else:
        P = np.zeros_like(th)
    P /= P.max() if P.size and np.max(P) > 0 else 1.0
    return (th_deg, P)

# === INTERFAZ — PESTAÑAS Y VISUALIZACIÓN DE RESULTADOS ===
tabParams, tabPlots, tabUniv, tabArb = st.tabs([_T(st.session_state.lang, 'tabs_params'), _T(st.session_state.lang, 'tabs_plots'), _T(st.session_state.lang, 'tabs_univ'), _T(st.session_state.lang, 'tabs_arbitrary')])
with tabParams:
        
    s = b1_eff ** 2 / (8 * lam * st.session_state.lE) if lam > 0 and st.session_state.lE > 0 and (b1_eff > 0) else np.nan
    t = a1_eff ** 2 / (8 * lam * st.session_state.lH) if lam > 0 and st.session_state.lH > 0 and (a1_eff > 0) else np.nan

    # Solo mostrar s o t (según tipo) y la longitud de onda (alineados en la misma fila)

    colL, colR = st.columns([2, 1])

    with colL:

        if st.session_state.tipo == 'Sectorial (Plano E)':

            st.markdown(

                f"<div style='font-size:26px; font-weight:700'>s = {s:.6f}</div>"

                if np.isfinite(s) else

                "<div style='font-size:26px; font-weight:700'>s = n/a</div>",

                unsafe_allow_html=True

            )

        elif st.session_state.tipo == 'Sectorial (Plano H)':

            st.markdown(

                f"<div style='font-size:26px; font-weight:700'>t = {t:.6f}</div>"

                if np.isfinite(t) else

                "<div style='font-size:26px; font-weight:700'>t = n/a</div>",

                unsafe_allow_html=True

            )

        else:

            # Piramidal: muestra s y t

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(

                        f"<div style='font-size:26px; font-weight:700'>s = {s:.6f}</div>"

                        if np.isfinite(s) else

                        "<div style='font-size:26px; font-weight:700'>s = n/a</div>",

                        unsafe_allow_html=True

                    )

                with c2:

                    st.markdown(

                        f"<div style='font-size:26px; font-weight:700'>t = {t:.6f}</div>"

                        if np.isfinite(t) else

                        "<div style='font-size:26px; font-weight:700'>t = n/a</div>",

                        unsafe_allow_html=True

                    )

        with colR:

            st.markdown(f"<div style='font-size:26px; font-weight:700; text-align:right'>λ = {lam:.6f} m</div>", unsafe_allow_html=True)

        st.caption(_T(st.session_state.lang, 'opt_caption'))
with tabPlots:
    st.subheader(_T(st.session_state.lang, 'plots_title'))
    with st.form('plots_form'):
        prec = st.radio(_T(st.session_state.lang, 'precision'), [_T(st.session_state.lang, 'prec_normal'), _T(st.session_state.lang, 'prec_high')], horizontal=True, index=0, help=_T(st.session_state.lang, 'prec_help'))
        modo_plot = st.radio(_T(st.session_state.lang, 'mode'), [_T(st.session_state.lang, 'mode_1d'), _T(st.session_state.lang, 'mode_polar')], horizontal=True, index=0)
        st.markdown('### ' + _T(st.session_state.lang, 'scale_header'))
        escala_sel = st.selectbox(_T(st.session_state.lang, 'scale_depth'), options=[-60, -50, -40, -30], index=2)
        use_plotly = st.toggle(_T(st.session_state.lang, 'hover_enable'), value=True if PLOTLY_AVAILABLE else False)
        submitted = st.form_submit_button(_T(st.session_state.lang, 'update_btn'))
    escala_abs = abs(escala_sel)
    N = 4096 if prec == _T(st.session_state.lang, 'prec_normal') else 8192
    if submitted:
        tipo_sel = st.session_state.tipo
        escala_abs = abs(escala_sel)
        N = 4096 if prec == _T(st.session_state.lang, 'prec_normal') else 8192

        def export_assets(fig_m, xdeg, patt_norm, plane):
            png_bytes = io.BytesIO()
            fig_m.savefig(png_bytes, format='png', dpi=160, bbox_inches='tight')
            png_bytes.seek(0)
            df = pd.DataFrame({'theta_deg': xdeg, 'pattern_norm': patt_norm})
            csv_bytes = io.BytesIO()
            csv_bytes.write(df.to_csv(index=False).encode('utf-8'))
            csv_bytes.seek(0)
            return (png_bytes, csv_bytes)
        if tipo_sel == 'Piramidal':
            if modo_plot == _T(st.session_state.lang, 'mode_1d'):
                thE, pattE = compute_pattern_1d('E', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                thH, pattH = compute_pattern_1d('H', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                pattE_db = patt_to_db(pattE)
                pattH_db = patt_to_db(pattH)
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    HPBW_E_loc = _hpbw_from_db(thE.tolist(), pattE_db.tolist())
                    HPBW_H_loc = _hpbw_from_db(thH.tolist(), pattH_db.tolist())
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW E ≈ {HPBW_E_loc:.2f}° · HPBW H ≈ {HPBW_H_loc:.2f}°</div></div>", unsafe_allow_html=True)
                    fig.add_trace(go.Scatter(x=thE.tolist(), y=pattE_db.tolist(), mode='lines', line=dict(color='black', width=2), name='E', hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra>E</extra>'))
                    fig.add_trace(go.Scatter(x=thH.tolist(), y=pattH_db.tolist(), mode='lines', line=dict(color='#1f77b4', width=2), name='H', hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra>H</extra>'))
                    fig.update_yaxes(range=[-escala_abs, 0.0], title=_T(st.session_state.lang, 'ylabel_norm'), tickfont=dict(color='black'), title_font=dict(color='black'))
                    fig.update_xaxes(range=[float(thE[0]), float(thE[-1])], title=_T(st.session_state.lang, 'xlabel_deg'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, height=520, margin=dict(l=40, r=20, t=50, b=40), title=_T(st.session_state.lang, 'title_1dE').replace('— Plano E', '— Piramidal: E & H'))
                    fig.update_xaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0', tickfont=dict(color='#0f0f0f', size=12), ticks='outside', tickcolor='#444', ticklen=5, linecolor='#444', showline=True, title_font=dict(color='#0f0f0f'))
                    fig.update_yaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0', tickfont=dict(color='#0f0f0f', size=12), ticks='outside', tickcolor='#444', ticklen=5, linecolor='#444', showline=True, title_font=dict(color='#0f0f0f'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig_m, ax = plt.subplots()
                    ax.plot(thE, pattE_db, color='black', linewidth=2.0, label='Plano E')
                    ax.plot(thH, pattH_db, color='#000000', linewidth=2.0, label='Plano H')
                    ax.set_ylim(-escala_abs, 0.0)
                    ax.set_xlim(thE[0], thE[-1])
                    ax.grid(True, linestyle='--', linewidth=0.6)
                    ax.set_xlabel(_T(st.session_state.lang, 'xlabel_deg'))
                    ax.set_ylabel(_T(st.session_state.lang, 'ylabel_norm'))
                    ax.set_title(_T(st.session_state.lang, 'title_1dE').replace('— Plano E', '— Piramidal: E & H'))
                    leg = ax.legend()
                    [t.set_color('black') for t in leg.get_texts()]
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_edgecolor('#333333')
                    st.pyplot(fig_m)
                figE = plt.figure()
                axE = figE.add_subplot(111)
                axE.plot(thE, pattE_db, color='black')
                axE.set_ylim(-escala_abs, 0.0)
                figH = plt.figure()
                axH = figH.add_subplot(111)
                axH.plot(thH, pattH_db, color='#1f77b4')
                axH.set_ylim(-escala_abs, 0.0)
                pngE, csvE = export_assets(figE, thE, pattE, 'E')
                pngH, csvH = export_assets(figH, thH, pattH, 'H')
            else:
                phE, pattE = compute_pattern_polar('E', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                phH, pattH = compute_pattern_polar('H', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                rE = -patt_to_db(pattE)
                rH = -patt_to_db(pattH)
                rE = np.clip(rE, 0.0, escala_abs)
                rH = np.clip(rH, 0.0, escala_abs)
                
                # === Añadido: mostrar HPBW (same as 1D) for polar E & H ===
                try:
                    HPBW_E_loc = hpbw_from_1d_same_method('E', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    HPBW_H_loc = hpbw_from_1d_same_method('H', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    st.markdown(
                        f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>"
                        f"{_T(st.session_state.lang, 'hpbw_e') if 'hpbw_e' in TXT.get(st.session_state.lang, {}) else 'HPBW E'} ≈ {HPBW_E_loc:.2f}° · "
                        f"{_T(st.session_state.lang, 'hpbw_h') if 'hpbw_h' in TXT.get(st.session_state.lang, {}) else 'HPBW H'} ≈ {HPBW_H_loc:.2f}°"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                except Exception as _ex:
                    pass
                # === /Added ===
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(theta=phE.tolist(), r=rE.tolist(), mode='lines', line=dict(color='black', width=2), name='E', hovertemplate='θ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra>E</extra>', customdata=(-rE).tolist()))
                    fig.add_trace(go.Scatterpolar(theta=phH.tolist(), r=rH.tolist(), mode='lines', line=dict(color='#1f77b4', width=2), name='H', hovertemplate='θ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra>H</extra>', customdata=(-rH).tolist()))
                    tickvals = list(range(0, int(escala_abs) + 1, 10))
                    ticktext = [f'{-t}' for t in tickvals]
                    fig.update_polars(radialaxis=dict(range=[escala_abs, 0], tickvals=tickvals, ticktext=ticktext, gridcolor='#e0e0e0', linecolor='#909090', tickfont=dict(color='#0f0f0f', size=12)), angularaxis=dict(direction='clockwise', rotation=90, tickmode='array', tickvals=list(range(0, 360, 30)), ticktext=[f'{t}°' for t in range(0, 360, 30)], gridcolor='#e0e0e0', linecolor='#909090', tickfont=dict(color='#0f0f0f', size=12)))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, height=520, margin=dict(l=40, r=20, t=50, b=40), title=_T(st.session_state.lang, 'title_polE').replace('— Plano E', '— Piramidal: E & H'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig_m = plt.figure()
                    ax = fig_m.add_subplot(111, projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    HPBW_pE_main = _hpbw_from_db(phE, rE)
                    HPBW_pH_main = _hpbw_from_db(phH, rH)
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBWφ E ≈ {HPBW_pE_main:.2f}° · HPBWφ H ≈ {HPBW_pH_main:.2f}°</div></div>", unsafe_allow_html=True)
                    ax.plot(np.deg2rad(phE), rE, color='black', linewidth=2.0, label='Plano E')
                    ax.plot(np.deg2rad(phH), rH, color='#000000', linewidth=2.0, label='Plano H')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.set_rlim(escala_abs, 0.0)
                    leg = ax.legend()
                    [t.set_color('black') for t in leg.get_texts()]
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_edgecolor('#333333')
                    st.pyplot(fig_m)
                figE = plt.figure()
                axE = figE.add_subplot(111, projection='polar')
                axE.set_theta_zero_location('N')
                axE.set_theta_direction(-1)
                axE.plot(np.deg2rad(phE), rE, color='black')
                axE.set_rlim(escala_abs, 0.0)
                figH = plt.figure()
                axH = figH.add_subplot(111, projection='polar')
                axH.set_theta_zero_location('N')
                axH.set_theta_direction(-1)
                axH.plot(np.deg2rad(phH), rH, color='#1f77b4')
                axH.set_rlim(escala_abs, 0.0)
                pngE, csvE = export_assets(figE, phE, pattE, 'E')
                pngH, csvH = export_assets(figH, phH, pattH, 'H')
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button(_T(st.session_state.lang, 'png_E'), data=pngE, file_name=f'patron_E.png', mime='image/png', key='dl_png_E_py')
            with c2:
                st.download_button(_T(st.session_state.lang, 'csv_E'), data=csvE, file_name=f'patron_E.csv', mime='text/csv', key='dl_csv_E_py')
            with c3:
                st.download_button(_T(st.session_state.lang, 'png_H'), data=pngH, file_name=f'patron_H.png', mime='image/png', key='dl_png_H_py')
            with c4:
                st.download_button(_T(st.session_state.lang, 'csv_H'), data=csvH, file_name=f'patron_H.csv', mime='text/csv', key='dl_csv_H_py')
        else:
            plane = 'E' if tipo_sel == 'Sectorial (Plano E)' else 'H'
            if modo_plot == _T(st.session_state.lang, 'mode_1d'):
                th_deg, patt = compute_pattern_1d(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                patt_db = patt_to_db(patt)
                # Secondary principal cut (overlay): if E, also show H@φ=0°; if H, show E@φ=90°
                th2_deg, patt2 = compute_overlay_at_fixed_phi(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, st.session_state.a, st.session_state.b, N)
                patt2_db = patt_to_db(patt2)

                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    HPBW_1a = _hpbw_from_db(th_deg.tolist(), patt_db.tolist())
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW ≈ {HPBW_1a:.2f}°</div></div>", unsafe_allow_html=True)
                    fig.add_trace(go.Scatter(x=th_deg.tolist(), y=patt_db.tolist(), mode='lines', name=('E' if plane=='E' else 'H'), line=dict(color='#000000', width=2), hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra></extra>'))
                    fig.add_trace(go.Scatter(x=th2_deg.tolist(), y=patt2_db.tolist(), mode='lines',
                                         line=dict(color='gray', width=2, dash='dash'), name=('H' if plane=='E' else 'E'),
                                         hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra></extra>'))
                    fig.update_yaxes(range=[-escala_abs, 0.0], title=_T(st.session_state.lang, 'ylabel_norm'), tickfont=dict(color='black'), title_font=dict(color='black'))
                    fig.update_xaxes(range=[float(th_deg[0]), float(th_deg[-1])], title=_T(st.session_state.lang, 'xlabel_deg'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, height=520, margin=dict(l=40, r=20, t=50, b=40), title=_T(st.session_state.lang, 'title_1dE' if plane == 'E' else 'title_1dH'))
                    fig.update_xaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0')
                    fig.update_yaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig_m, ax = plt.subplots()

                    HPBW_1a = _hpbw_from_db(th_deg, patt_db)
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW ≈ {HPBW_1a:.2f}°</div></div>", unsafe_allow_html=True)
                    ax.plot(th_deg, patt_db, color='#000000', linewidth=2.0)
                    ax.plot(th2_deg, patt2_db, linestyle='--', color='gray', linewidth=1.5, alpha=0.9)
                    ax.legend(['E' if plane=='E' else 'H', 'H' if plane=='E' else 'E'])
                    ax.set_ylim(-escala_abs, 0.0)
                    ax.set_xlim(th_deg[0], th_deg[-1])
                    ax.set_xlabel(_T(st.session_state.lang, 'xlabel_deg'))
                    ax.set_ylabel(_T(st.session_state.lang, 'ylabel_norm'))
                    ax.set_title(_T(st.session_state.lang, 'title_1dE' if plane == 'E' else 'title_1dH'))
                    ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)
                fig_m2, ax2 = plt.subplots()
                ax2.plot(th_deg, patt_db, color='blue')
                ax2.set_ylim(-escala_abs, 0.0)
                png, csv = export_assets(fig_m2, th_deg, patt, plane)
            else:
                ph_deg, patt = compute_pattern_polar(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, N)
                r = np.clip(-patt_to_db(patt), 0.0, escala_abs)
                # Overlay curve emulating 'Cortes arbitrarios' at fixed phi: 0° for E, 90° for H
                theta_eval_deg = np.minimum(ph_deg, 360.0 - ph_deg)
                th2 = np.deg2rad(theta_eval_deg)
                if plane == 'E':
                    ph_const = 0.0  # 0°
                    Et2, Ep2 = fields_sectorial_E(th2, ph_const, lam, b1_eff, st.session_state.lE, st.session_state.a)
                else:
                    ph_const = np.deg2rad(90.0)  # 90°
                    Et2, Ep2 = fields_sectorial_H(th2, ph_const, lam, a1_eff, st.session_state.lH, st.session_state.b)
                P2 = np.abs(Et2)**2 + np.abs(Ep2)**2
                P2 /= P2.max() if P2.size and np.max(P2) > 0 else 1.0
                r2 = np.clip(-patt_to_db(P2), 0.0, escala_abs)

                
                # === Añadido: mostrar HPBW (same as 1D) for polar single-plane ===
                try:
                    HPBW_loc = hpbw_from_1d_same_method(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    lbl = _T(st.session_state.lang, 'hpbw_e') if plane == 'E' else _T(st.session_state.lang, 'hpbw_h')
                    if not lbl:
                        lbl = f"HPBW {plane}"
                    st.markdown(
                        f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>"
                        f"{lbl} ≈ {HPBW_loc:.2f}°"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                except Exception as _ex:
                    pass
                # === /Added ===
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(theta=ph_deg.tolist(), r=r.tolist(), mode='lines', line=dict(color='#000000', width=2), name='E' if plane == 'E' else 'H', hovertemplate='θ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra></extra>', customdata=(-r).tolist()))
                    fig.add_trace(go.Scatterpolar(theta=ph_deg.tolist(), r=r2.tolist(), mode='lines', line=dict(color='gray', width=2, dash='dash'), name=('H' if plane=='E' else 'E'), hovertemplate='φ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra></extra>', customdata=(-r2).tolist()))
                    tickvals = list(range(0, int(escala_abs) + 1, 10))
                    ticktext = [f'{-t}' for t in tickvals]
                    fig.update_polars(radialaxis=dict(range=[escala_abs, 0], tickvals=tickvals, ticktext=ticktext, gridcolor='#e0e0e0', linecolor='#909090'), angularaxis=dict(direction='clockwise', rotation=90, tickmode='array', tickvals=list(range(0, 360, 30)), ticktext=[f'{t}°' for t in range(0, 360, 30)], gridcolor='#e0e0e0', linecolor='#909090'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, height=520, margin=dict(l=40, r=20, t=50, b=40), title=_T(st.session_state.lang, 'title_polE' if plane == 'E' else 'title_polH'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig_m = plt.figure()
                    ax = fig_m.add_subplot(111, projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    ax.plot(np.deg2rad(ph_deg), r, color='#000000', linewidth=2.0)
                    ax.plot(np.deg2rad(ph_deg), r2, linestyle='--', color='gray', linewidth=1.5, alpha=0.9)
                    ax.legend(['E' if plane=='E' else 'H', 'H' if plane=='E' else 'E'])
                    ax.set_rlim(escala_abs, 0.0)
                    ax.set_title(_T(st.session_state.lang, 'title_polE' if plane == 'E' else 'title_polH'))
                    ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)
                fig_m2 = plt.figure()
                ax2 = fig_m2.add_subplot(111, projection='polar')
                ax2.set_theta_zero_location('N')
                ax2.set_theta_direction(-1)
                ax2.set_theta_zero_location('N')
                ax2.set_theta_direction(-1)
                ax2.plot(np.deg2rad(ph_deg), r, color='blue')
                ax2.set_rlim(escala_abs, 0.0)
                png, csv = export_assets(fig_m2, ph_deg, patt, plane)
            if tipo_sel != 'Piramidal':
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(_T(st.session_state.lang, 'png_E' if plane == 'E' else 'png_H'), data=png, file_name=f'patron_{plane}.png', mime='image/png')
                with c2:
                    st.download_button(_T(st.session_state.lang, 'csv_E' if plane == 'E' else 'csv_H'), data=csv, file_name=f'patron_{plane}.csv', mime='text/csv')
with tabUniv:
    st.markdown("""
    <style>
    [data-testid='stJson']{display:none !important;}
    </style>
    """, unsafe_allow_html=True)
    st.subheader(_T(st.session_state.lang, 'univ_title'))
    s_cur = b1_eff ** 2 / (8 * lam * st.session_state.lE) if lam > 0 and st.session_state.lE > 0 and (b1_eff > 0) else float('nan')
    t_cur = a1_eff ** 2 / (8 * lam * st.session_state.lH) if lam > 0 and st.session_state.lH > 0 and (a1_eff > 0) else float('nan')
    kind = st.radio(_T(st.session_state.lang, 'univ_kind'), [_T(st.session_state.lang, 'univ_E'), _T(st.session_state.lang, 'univ_H')], horizontal=True, key='univ_kind_tab')
    is_E = (kind == _T(st.session_state.lang, 'univ_E'))
    if is_E:
        st.caption(_T(st.session_state.lang, 'univ_s_current').format(s=s_cur) if np.isfinite(s_cur) else 'n/a')
        default_vals = '1, 0.5, 0.25, 0.125, 0.016'
        vals_str = st.text_input(_T(st.session_state.lang, 'univ_values_s'), value=default_vals, key='univ_vals_s_tab')
    else:
        st.caption(_T(st.session_state.lang, 'univ_t_current').format(t=t_cur) if np.isfinite(t_cur) else 'n/a')
        default_vals = '1, 0.5, 0.25, 0.125, 0.016'
        vals_str = st.text_input(_T(st.session_state.lang, 'univ_values_t'), value=default_vals, key='univ_vals_t_tab')
    y_min_univ = st.number_input('Escala (dB)', value=-40, step=1, help='Mínimo del eje Y en dB para las curvas universales (0 siempre es el máximo).', key='univ_ymin_tab')
    x_max_univ = st.slider(_T(st.session_state.lang, 'univ_xmax'), min_value=2.0, max_value=8.0, value=4.0, step=0.5, key='univ_xmax_tab')

    def _parse_floats_list_univ(s):
        out = []
        for tok in s.replace(';', ',').split(','):
            tok = tok.strip().replace(' ', '')
            if tok:
                try:
                    out.append(float(tok))
                except:
                    pass
        seen = set()
        uniq = []
        for v in out:
            if v in seen:
                continue
            seen.add(v)
            uniq.append(v)
        return uniq[:8]
    vals = _parse_floats_list_univ(vals_str)
    import numpy as _np
    x = _np.linspace(0.0, float(x_max_univ), 1201)
    if PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        fig_univ = go.Figure()
        fig_univ.update_layout(colorway=['#1f77b4', '#e74c3c', '#2ecc71', '#ff7f0e', '#8e44ad'])
        for v in vals if vals else []:
            y = _univ_E_db(v, x) if is_E else _univ_H_db(v, x)
            import numpy as np
            if (y is None) or (len(y)==0) or np.all(np.isnan(y)):
                continue
            tag = f's={v:.3f}' if is_E else f't={v:.3f}'
            fig_univ.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode='lines', name=tag))
        fig_univ.update_layout(template='plotly_white', height=500, margin=dict(l=40, r=20, t=40, b=50), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True)
        fig_univ.update_yaxes(range=[int(y_min_univ), 0], tickfont=dict(color='#0f0f0f'), title_font=dict(color='#0f0f0f'))
        fig_univ.update_xaxes(tickfont=dict(color='#0f0f0f'), title_font=dict(color='#0f0f0f'))
        fig_univ.update_xaxes(title=_T(st.session_state.lang, 'univ_xlabel_E') if is_E else _T(st.session_state.lang, 'univ_xlabel_H'))
        fig_univ.update_yaxes(title=_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
        if vals:
            st.plotly_chart(fig_univ, use_container_width=True)

        # --- Exportación (PNG + CSV) Curvas universales ---
        import io
        import pandas as pd
        import matplotlib.pyplot as _plt
        series = []
        labels = []
        for v in (vals if vals else []):
            y = _univ_E_db(v, x) if is_E else _univ_H_db(v, x)
            import numpy as np
            if (y is None) or (len(y)==0) or np.all(np.isnan(y)):
                continue
            series.append(y)
            labels.append((f's={v:.3f}' if is_E else f't={v:.3f}'))
        csv_bytes = None
        if series:
            df = pd.DataFrame({'x': x})
            for lab, y in zip(labels, series):
                df[lab] = y
            _csv_buf = io.BytesIO()
            _csv_buf.write(df.to_csv(index=False).encode('utf-8'))
            _csv_buf.seek(0)
            csv_bytes = _csv_buf
        png_bytes = None
        if series:
            fig_export, ax_export = _plt.subplots()
            for lab, y in zip(labels, series):
                ax_export.plot(x, y, linewidth=2.0, label=lab)
            ax_export.set_xlabel(_T(st.session_state.lang, 'univ_xlabel_E') if is_E else _T(st.session_state.lang, 'univ_xlabel_H'))
            ax_export.set_ylabel('Nivel (dB, normalizado)')
            ax_export.set_ylim(int(y_min_univ), 0)
            ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
            if labels:
                leg = ax_export.legend()
                [t.set_color('black') for t in leg.get_texts()]
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_edgecolor('#333333')
            _png_buf = io.BytesIO()
            fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight')
            _png_buf.seek(0)
            png_bytes = _png_buf
            _plt.close(fig_export)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                _T(st.session_state.lang, 'png_E') if is_E else _T(st.session_state.lang, 'png_H'),
                data=png_bytes,
                file_name=f'curvas_univ_{"E" if is_E else "H"}.png',
                mime='image/png',
                disabled=(png_bytes is None),
                key='dl_univ_png'
            )
        with c2:
            st.download_button(
                _T(st.session_state.lang, 'csv_E') if is_E else _T(st.session_state.lang, 'csv_H'),
                data=csv_bytes,
                file_name=f'curvas_univ_{"E" if is_E else "H"}.csv',
                mime='text/csv',
                disabled=(csv_bytes is None),
                key='dl_univ_csv'
            )
        # --- fin exportación ---

    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for v in vals if vals else []:
            y = _univ_E_db(v, x) if is_E else _univ_H_db(v, x)
            import numpy as np
            if (y is None) or (len(y)==0) or np.all(np.isnan(y)):
                continue
            ax.plot(x, y, linewidth=2.0, label=f's={v:.3f}' if is_E else f't={v:.3f}')
        ax.set_xlabel(_T(st.session_state.lang, 'univ_xlabel_E') if is_E else _T(st.session_state.lang, 'univ_xlabel_H'))
        ax.set_ylabel('Nivel (dB, normalizado)')
        ax.set_ylim(int(y_min_univ), 0)
        ax.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
        if vals:
            leg = ax.legend()
            [t.set_color('black') for t in leg.get_texts()]
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#333333')
            [t.set_color('black') for t in leg.get_texts()]
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#333')
        if vals:
            st.pyplot(fig, use_container_width=True)

        # --- Exportación (PNG + CSV) Curvas universales ---
        import io
        import pandas as pd
        import matplotlib.pyplot as _plt
        series = []
        labels = []
        for v in (vals if vals else []):
            y = _univ_E_db(v, x) if is_E else _univ_H_db(v, x)
            import numpy as np
            if (y is None) or (len(y)==0) or np.all(np.isnan(y)):
                continue
            series.append(y)
            labels.append((f's={v:.3f}' if is_E else f't={v:.3f}'))
        csv_bytes = None
        if series:
            df = pd.DataFrame({'x': x})
            for lab, y in zip(labels, series):
                df[lab] = y
            _csv_buf = io.BytesIO()
            _csv_buf.write(df.to_csv(index=False).encode('utf-8'))
            _csv_buf.seek(0)
            csv_bytes = _csv_buf
        png_bytes = None
        if series:
            fig_export, ax_export = _plt.subplots()
            for lab, y in zip(labels, series):
                ax_export.plot(x, y, linewidth=2.0, label=lab)
            ax_export.set_xlabel(_T(st.session_state.lang, 'univ_xlabel_E') if is_E else _T(st.session_state.lang, 'univ_xlabel_H'))
            ax_export.set_ylabel('Nivel (dB, normalizado)')
            ax_export.set_ylim(int(y_min_univ), 0)
            ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
            if labels:
                leg = ax_export.legend()
                [t.set_color('black') for t in leg.get_texts()]
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_edgecolor('#333333')
            _png_buf = io.BytesIO()
            fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight')
            _png_buf.seek(0)
            png_bytes = _png_buf
            _plt.close(fig_export)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                _T(st.session_state.lang, 'png_E') if is_E else _T(st.session_state.lang, 'png_H'),
                data=png_bytes,
                file_name=f'curvas_univ_{"E" if is_E else "H"}.png',
                mime='image/png',
                disabled=(png_bytes is None),
                key='dl_univ_png'
            )
        with c2:
            st.download_button(
                _T(st.session_state.lang, 'csv_E') if is_E else _T(st.session_state.lang, 'csv_H'),
                data=csv_bytes,
                file_name=f'curvas_univ_{"E" if is_E else "H"}.csv',
                mime='text/csv',
                disabled=(csv_bytes is None),
                key='dl_univ_csv'
            )
        # --- fin exportación ---

with tabArb:
    st.subheader(_T(st.session_state.lang, 'tabs_arbitrary'))
    c1, c2 = st.columns(2)
    with c1:
        phi_sel = st.slider('φ (°)', min_value=0, max_value=180, value=90, help=_T(st.session_state.lang, 'phi_help') if 'phi_help' in TXT.get(st.session_state.lang, {}) else 'φ=0° → H ; φ=90° → E')
    with c2:
        modo_plot = st.radio(_T(st.session_state.lang, 'mode'), [_T(st.session_state.lang, 'mode_1d'), _T(st.session_state.lang, 'mode_polar')], horizontal=True, index=0)
    prec = st.radio(_T(st.session_state.lang, 'precision'), [_T(st.session_state.lang, 'prec_normal'), _T(st.session_state.lang, 'prec_high')], horizontal=True, index=0, help=_T(st.session_state.lang, 'prec_help'))
    escala_sel = st.selectbox(_T(st.session_state.lang, 'scale_depth'), options=[-60, -50, -40, -30], index=2)
    use_plotly = st.toggle(_T(st.session_state.lang, 'hover_enable'), value=True if PLOTLY_AVAILABLE else False, key='arb_plotly_toggle')
    submitted = st.button(_T(st.session_state.lang, 'update_btn'), key='arb_update_btn')
    if submitted:
        N = 1441 if prec == _T(st.session_state.lang, 'prec_high') else 721
        escala_abs = abs(escala_sel)
        if modo_plot == _T(st.session_state.lang, 'mode_1d'):
            if st.session_state.tipo == 'Piramidal':
                th_deg, P_E = pattern_arbitrary('Sectorial (Plano E)', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, st.session_state.a, st.session_state.b, phi_sel, N)
                _,     P_H = pattern_arbitrary('Sectorial (Plano H)', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, st.session_state.a, st.session_state.b, phi_sel, N)
                P_E = P_E / (P_E.max() if P_E.size and P_E.max() > 0 else 1.0)
                P_H = P_H / (P_H.max() if P_H.size and P_H.max() > 0 else 1.0)
                patt_db_E = patt_to_db(P_E); patt_db_H = patt_to_db(P_H)
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    HPBW_Ea = _hpbw_from_db(th_deg.tolist(), patt_db_E.tolist())
                    HPBW_Ha = _hpbw_from_db(th_deg.tolist(), patt_db_H.tolist())
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW E ≈ {HPBW_Ea:.2f}° · HPBW H ≈ {HPBW_Ha:.2f}°</div></div>", unsafe_allow_html=True)
                    fig.add_trace(go.Scatter(x=th_deg.tolist(), y=patt_db_E.tolist(), mode='lines', name='E', line=dict(color='#000000', width=2), hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra>E</extra>'))
                    fig.add_trace(go.Scatter(x=th_deg.tolist(), y=patt_db_H.tolist(), mode='lines', name='H', line=dict(color='#1f77b4', width=2), hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra>H</extra>'))
                    fig.update_yaxes(range=[-escala_abs, 0.0], tickfont=dict(color='black'), title_font=dict(color='black'))
                    fig.update_xaxes(range=[float(th_deg[0]), float(th_deg[-1])], title=_T(st.session_state.lang, 'xlabel_deg'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#0f0f0f'), margin=dict(l=40, r=20, t=50, b=40), title=f'φ = {phi_sel}°')
                    fig.update_xaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0'); fig.update_yaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0')
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

                else:
                    fig_m, ax = plt.subplots()
                    HPBW_Ea = _hpbw_from_db(th_deg, patt_db_E)
                    HPBW_Ha = _hpbw_from_db(th_deg, patt_db_H)
                    ax.plot(th_deg, patt_db_E, color='#000000', linewidth=2.0, label='E')
                    ax.plot(th_deg, patt_db_H, color='#000000', linewidth=2.0, label='H')
                    leg = ax.legend()
                    [t.set_color('black') for t in leg.get_texts()]
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_edgecolor('#333333')
                    ax.set_xlabel(_T(st.session_state.lang, 'xlabel_deg')); ax.set_ylabel(_T(st.session_state.lang, 'ylabel_norm'))
                    ax.set_title(f'φ = {phi_sel}°'); ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

            else:
                th_deg, P = pattern_arbitrary(st.session_state.tipo, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH, st.session_state.a, st.session_state.b, phi_sel, N)
                patt_db = patt_to_db(P)
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    HPBW_1a = _hpbw_from_db(th_deg.tolist(), patt_db.tolist())
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW ≈ {HPBW_1a:.2f}°</div></div>", unsafe_allow_html=True)
                    fig.add_trace(go.Scatter(x=th_deg.tolist(), y=patt_db.tolist(), mode='lines', line=dict(color='#000000', width=2), hovertemplate='θ=%{x:.2f}°<br>dB=%{y:.2f}<extra></extra>'))
                    fig.update_yaxes(range=[-escala_abs, 0.0], tickfont=dict(color='black'), title_font=dict(color='black'))
                    fig.update_xaxes(range=[float(th_deg[0]), float(th_deg[-1])], title=_T(st.session_state.lang, 'xlabel_deg'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#0f0f0f'), margin=dict(l=40, r=20, t=50, b=40), title=f'φ = {phi_sel}°')
                    fig.update_xaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0'); fig.update_yaxes(gridcolor='#e6e6e6', zeroline=True, zerolinecolor='#b0b0b0')
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

                else:
                    fig_m, ax = plt.subplots()

                    HPBW_1a = _hpbw_from_db(th_deg, patt_db)
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBW ≈ {HPBW_1a:.2f}°</div></div>", unsafe_allow_html=True)
                    ax.plot(th_deg, patt_db, color='#000000', linewidth=2.0)
                    ax.set_ylim(-escala_abs, 0.0); ax.set_xlim(th_deg[0], th_deg[-1])
                    ax.set_xlabel(_T(st.session_state.lang, 'xlabel_deg')); ax.set_ylabel(_T(st.session_state.lang, 'ylabel_norm'))
                    ax.set_title(f'φ = {phi_sel}°'); ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

        else:
            phi_plot = np.linspace(0.0, 360.0, 720)
            theta_eval_deg = np.minimum(phi_plot, 360.0 - phi_plot)
            th = np.deg2rad(theta_eval_deg)
            ph = np.deg2rad(phi_sel)
            if st.session_state.tipo == 'Piramidal':
                EtE, EpE = fields_sectorial_E(th, ph, lam, b1_eff, st.session_state.lE, st.session_state.a)
                EtH, EpH = fields_sectorial_H(th, ph, lam, a1_eff, st.session_state.lH, st.session_state.b)
                P_E = (np.abs(EtE)**2 + np.abs(EpE)**2)
                P_H = (np.abs(EtH)**2 + np.abs(EpH)**2)
                P_E = P_E / (P_E.max() if P_E.size and P_E.max() > 0 else 1.0)
                P_H = P_H / (P_H.max() if P_H.size and P_H.max() > 0 else 1.0)
                rE = np.clip(-patt_to_db(P_E), 0.0, escala_abs)
                rH = np.clip(-patt_to_db(P_H), 0.0, escala_abs)
                # === Added (arbitrary polar): show HPBW banner using 1D method ===
                try:
                    HPBW_E_loc = hpbw_from_1d_same_method('E', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    HPBW_H_loc = hpbw_from_1d_same_method('H', lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    st.markdown(
                        f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>"
                        f"{_T(st.session_state.lang, 'hpbw_e') if 'hpbw_e' in TXT.get(st.session_state.lang, {}) else 'HPBW E'} ≈ {HPBW_E_loc:.2f}° · "
                        f"{_T(st.session_state.lang, 'hpbw_h') if 'hpbw_h' in TXT.get(st.session_state.lang, {}) else 'HPBW H'} ≈ {HPBW_H_loc:.2f}°"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                except Exception as _ex:
                    pass
                # === /Added ===
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(theta=phi_plot.tolist(), r=rE.tolist(), mode='lines', name='E', line=dict(color='#000000', width=2), hovertemplate='φ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra>E</extra>', customdata=(-rE).tolist()))
                    fig.add_trace(go.Scatterpolar(theta=phi_plot.tolist(), r=rH.tolist(), mode='lines', name='H', line=dict(color='#1f77b4', width=2), hovertemplate='φ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra>H</extra>', customdata=(-rH).tolist()))
                    tickvals = list(range(0, int(escala_abs) + 1, 10)); ticktext = [f'{-t}' for t in tickvals]
                    fig.update_polars(radialaxis=dict(range=[escala_abs, 0], tickvals=tickvals, ticktext=ticktext, angle=90, showline=True), angularaxis=dict(direction='clockwise', rotation=90, tickmode='array', tickvals=list(range(0, 360, 30)), gridcolor='#e0e0e0', linecolor='#909090'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, margin=dict(l=40, r=20, t=50, b=40), title=f'φ = {phi_sel}°')
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

                else:
                    fig_m = plt.figure()
                    ax = fig_m.add_subplot(111, projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    HPBW_pE = _hpbw_from_db(phi_plot, rE)
                    HPBW_pH = _hpbw_from_db(phi_plot, rH)
                    ax.plot(np.deg2rad(phi_plot), rE, color='#000000', linewidth=2.0, label=f'E (HPBWφ≈{HPBW_pE:.2f}°)')
                    ax.plot(np.deg2rad(phi_plot), rH, color='#000000', linewidth=2.0, label=f'H (HPBWφ≈{HPBW_pH:.2f}°)')
                    leg = ax.legend(); [t.set_color('black') for t in leg.get_texts()]; leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                    ax.set_rlim(escala_abs, 0.0)
                    ax.set_title(f'φ = {phi_sel}°'); ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

            elif st.session_state.tipo == 'Sectorial (Plano E)':
                Et, Ep = fields_sectorial_E(th, ph, lam, b1_eff, st.session_state.lE, st.session_state.a)
                Pp = np.abs(Et)**2 + np.abs(Ep)**2
                Pp /= Pp.max() if Pp.size and np.max(Pp) > 0 else 1.0
                r = np.clip(-patt_to_db(Pp), 0.0, escala_abs)
                # === Added: HPBW banner (plane based on selection) ===
                try:
                    plane = 'E' if st.session_state.tipo == 'Sectorial (Plano E)' else 'H'
                    HPBW_loc = hpbw_from_1d_same_method(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    lbl = _T(st.session_state.lang, 'hpbw_e') if plane == 'E' else _T(st.session_state.lang, 'hpbw_h')
                    if not lbl:
                        lbl = f"HPBW {plane}"
                    st.markdown(
                        f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>"
                        f"{lbl} ≈ {HPBW_loc:.2f}°"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                except Exception as _ex:
                    pass
                # === /Added ===
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(theta=phi_plot.tolist(), r=r.tolist(), mode='lines', line=dict(color='#000000', width=2), hovertemplate='φ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra></extra>', customdata=(-r).tolist()))
                    tickvals = list(range(0, int(escala_abs) + 1, 10)); ticktext = [f'{-t}' for t in tickvals]
                    fig.update_polars(radialaxis=dict(range=[escala_abs, 0], tickvals=tickvals, ticktext=ticktext, angle=90, showline=True), angularaxis=dict(direction='clockwise', rotation=90, tickmode='array', tickvals=list(range(0, 360, 30)), gridcolor='#e0e0e0', linecolor='#909090'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, margin=dict(l=40, r=20, t=50, b=40), title=f'φ = {phi_sel}°')
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

                else:
                    fig_m = plt.figure()
                    ax = fig_m.add_subplot(111, projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    HPBW_p1 = _hpbw_from_db(phi_plot, r)
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBWφ ≈ {HPBW_p1:.2f}°</div></div>", unsafe_allow_html=True)
                    ax.plot(np.deg2rad(phi_plot), r, color='#000000', linewidth=2.0)
                    ax.set_rlim(escala_abs, 0.0)
                    ax.set_title(f'φ = {phi_sel}°'); ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

            else:  # Sectorial (Plano H)
                Et, Ep = fields_sectorial_H(th, ph, lam, a1_eff, st.session_state.lH, st.session_state.b)
                Pp = np.abs(Et)**2 + np.abs(Ep)**2
                Pp /= Pp.max() if Pp.size and np.max(Pp) > 0 else 1.0
                r = np.clip(-patt_to_db(Pp), 0.0, escala_abs)
                # === Added: HPBW banner (plane based on selection) ===
                try:
                    plane = 'E' if st.session_state.tipo == 'Sectorial (Plano E)' else 'H'
                    HPBW_loc = hpbw_from_1d_same_method(plane, lam, a1_eff, b1_eff, st.session_state.lE, st.session_state.lH)
                    lbl = _T(st.session_state.lang, 'hpbw_e') if plane == 'E' else _T(st.session_state.lang, 'hpbw_h')
                    if not lbl:
                        lbl = f"HPBW {plane}"
                    st.markdown(
                        f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>"
                        f"{lbl} ≈ {HPBW_loc:.2f}°"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
                except Exception as _ex:
                    pass
                # === /Added ===
                if use_plotly and PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(theta=phi_plot.tolist(), r=r.tolist(), mode='lines', line=dict(color='#000000', width=2), hovertemplate='φ=%{theta:.1f}°<br>dB=%{customdata:.2f}<extra></extra>', customdata=(-r).tolist()))
                    tickvals = list(range(0, int(escala_abs) + 1, 10)); ticktext = [f'{-t}' for t in tickvals]
                    fig.update_polars(radialaxis=dict(range=[escala_abs, 0], tickvals=tickvals, ticktext=ticktext, angle=90, showline=True), angularaxis=dict(direction='clockwise', rotation=90, tickmode='array', tickvals=list(range(0, 360, 30)), gridcolor='#e0e0e0', linecolor='#909090'))
                    fig.update_layout(template='plotly_white', paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'), legend=dict(font=dict(color='black', size=13), bgcolor='white', bordercolor='#333', borderwidth=1), showlegend=True, margin=dict(l=40, r=20, t=50, b=40), title=f'φ = {phi_sel}°')
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---

                else:
                    fig_m = plt.figure()
                    ax = fig_m.add_subplot(111, projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    HPBW_p1 = _hpbw_from_db(phi_plot, r)
                    st.markdown(f"<div class='hero' style='margin-bottom:8px'><div style='font-weight:700'>HPBWφ ≈ {HPBW_p1:.2f}°</div></div>", unsafe_allow_html=True)
                    ax.plot(np.deg2rad(phi_plot), r, color='#000000', linewidth=2.0)
                    ax.set_rlim(escala_abs, 0.0)
                    ax.set_title(f'φ = {phi_sel}°'); ax.grid(True, linestyle='--', linewidth=0.6)
                    st.pyplot(fig_m)

                    # --- Exportación (PNG + CSV) Cortes arbitrarios ---
                    import io
                    import pandas as pd
                    import matplotlib.pyplot as _plt
                    csv_bytes = None; png_bytes = None
                    try:
                        if 'patt_db_E' in locals() and 'patt_db_H' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['E_dB'] = patt_db_E; df['H_dB'] = patt_db_H
                            series_to_plot = ('theta_deg', [('E', patt_db_E), ('H', patt_db_H)])
                        elif 'patt_db' in locals() and 'th_deg' in locals():
                            df = pd.DataFrame({'theta_deg': th_deg}); df['pattern_dB'] = patt_db
                            series_to_plot = ('theta_deg', [('pattern', patt_db)])
                        elif 'rE' in locals() and 'rH' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['E_dB'] = rE; df['H_dB'] = rH
                            series_to_plot = ('phi_deg', [('E', rE), ('H', rH)])
                        elif 'r' in locals() and 'phi_plot' in locals():
                            df = pd.DataFrame({'phi_deg': phi_plot}); df['pattern_dB'] = r
                            series_to_plot = ('phi_deg', [('pattern', r)])
                        else:
                            raise RuntimeError('No exportable data')
                        _csv_buf = io.BytesIO(); _csv_buf.write(df.to_csv(index=False).encode('utf-8')); _csv_buf.seek(0); csv_bytes = _csv_buf
                        fig_export, ax_export = _plt.subplots(subplot_kw={'projection': 'polar'} if series_to_plot[0]=='phi_deg' else {})
                        x_axis = phi_plot if series_to_plot[0]=='phi_deg' else th_deg
                        for _lab, _yy in series_to_plot[1]:
                            if series_to_plot[0]=='phi_deg':
                                ax_export.plot(np.deg2rad(x_axis), _yy, linewidth=2.0, label=_lab); ax_export.set_rlim(abs(escala_sel), 0.0)
                            else:
                                ax_export.plot(x_axis, _yy, linewidth=2.0, label=_lab); ax_export.set_ylim(-abs(escala_sel), 0.0)
                        if series_to_plot[0]=='phi_deg':
                            ax_export.set_title(f'φ = {phi_sel}°')
                        else:
                            ax_export.set_xlabel(_T(st.session_state.lang, 'xlabel_deg') if 'xlabel_deg' in TXT.get(st.session_state.lang, {}) else 'θ (°)')
                            ax_export.set_ylabel(_T(st.session_state.lang, 'ylabel_norm') if 'ylabel_norm' in TXT.get(st.session_state.lang, {}) else 'Nivel (dB, normalizado)')
                        ax_export.grid(True, linestyle='--', linewidth=0.6, color='#dddddd')
                        if len(series_to_plot[1]) > 1:
                            leg = ax_export.legend(); [t.set_color('black') for t in leg.get_texts()]
                            leg.get_frame().set_facecolor('white'); leg.get_frame().set_edgecolor('#333333')
                        _png_buf = io.BytesIO(); fig_export.savefig(_png_buf, format='png', dpi=160, bbox_inches='tight'); _png_buf.seek(0); png_bytes = _png_buf
                        _plt.close(fig_export)
                    
                    except Exception as _ex:
                        csv_bytes = None; png_bytes = None
                    c1, c2 = st.columns(2)
                    with c1:
                        if png_bytes is not None:
                            st.download_button('Descargar PNG', data=png_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.png', mime='image/png', key=f'arb_png_{phi_sel}')
                    with c2:
                        if csv_bytes is not None:
                            st.download_button('Descargar CSV', data=csv_bytes, file_name=f'corte_arbitrario_phi_{phi_sel}.csv', mime='text/csv', key=f'arb_csv_{phi_sel}')
                    # --- fin exportación ---
