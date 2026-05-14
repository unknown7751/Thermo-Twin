"""
Phase 6 -- Thermo-Twin Operator Dashboard
Streamlit dashboard for live HVAC fault monitoring and prescriptive diagnostics.

Run:
    streamlit run dashboard/app.py
    (backend must be running: python backend/app.py)
"""

import sys
import time
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.threshold import SEVERITY_PROFILES, SeverityClassifier

BACKEND = "http://localhost:5000"

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="Thermo-Twin | Carrier HVAC Diagnostics",
    page_icon="🏭",
    layout="wide",
)

# -- CSS -----------------------------------------------------------------------

st.markdown("""
<style>
  .block-container { padding-top: 1rem; padding-bottom: 0; }
  .header-title {
    font-size: 2rem; font-weight: 800; color: #E6EDF3; margin-bottom: 0;
  }
  .header-sub { font-size: 0.88rem; color: #8B949E; margin-top: 0.15rem; }
  .carrier-bar {
    height: 4px;
    background: linear-gradient(90deg, #1A2E4A 50%, #C95F0A 50%);
    border-radius: 2px; margin: 0.5rem 0 1rem 0;
  }
  /* Severity card */
  .sev-card {
    padding: 1.2rem; border-radius: 12px; text-align: center;
  }
  .sev-normal   { background:#0d2b0d; border: 2px solid #4CAF50; }
  .sev-warning  { background:#2b1a00; border: 2px solid #FF9800; }
  .sev-critical { background:#2b0505; border: 2px solid #F44336; }
  /* Prescription card */
  .presc-card {
    padding: 1.2rem 1.6rem; border-radius: 10px; margin-top: 0.5rem;
  }
  .presc-warning  { background:#241500; border-left: 6px solid #FF9800; }
  .presc-critical { background:#200606; border-left: 6px solid #F44336; }
  .presc-normal   { background:#161B22; border: 1px solid #30363D; border-radius: 10px; padding: 1rem; }
  .plabel {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
    color: #8B949E; text-transform: uppercase;
  }
  .pvalue { font-size: 1rem; color: #E6EDF3; margin-bottom: 0.5rem; }
  /* Machine info */
  .mcard {
    background:#161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 1rem;
  }
  .mlabel { color:#8B949E; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px; }
  .mvalue { color:#E6EDF3; font-size:0.96rem; font-weight:600; margin-bottom:0.5rem; }
  .fault-big { font-size:1.4rem; font-weight:800; color:#F44336; }
</style>
""", unsafe_allow_html=True)

# -- Signal simulation ---------------------------------------------------------

# Per-stream ranges for 0-1 normalisation (for chart clarity)
_RANGES = {
    "compressor_power_kw":    (2.0,  6.5),
    "discharge_pressure_psi": (130., 460.),
    "fan_rpm":                (600., 2200.),
    "supply_air_temp_c":      (4.0,  18.0),
}


def _norm(arr, col):
    lo, hi = _RANGES[col]
    return np.clip((np.asarray(arr) - lo) / (hi - lo), -0.2, 1.4)


def _normal_signal(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    demand = 0.4 * np.sin(2 * np.pi * 0.02 * t) + 0.15 * np.sin(2 * np.pi * 0.007 * t)
    comp = np.clip(3.5 + demand + rng.normal(0, 0.08, n), 2.0, 6.0)
    return pd.DataFrame({
        "compressor_power_kw":    comp,
        "discharge_pressure_psi": 70.0 * comp + rng.normal(0, 4, n),
        "fan_rpm":                340.0 * comp + rng.normal(0, 30, n),
        "supply_air_temp_c":      18.0 - 2.0 * comp + rng.normal(0, 0.3, n),
    })


def _fault_signal(scenario: str, n: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    comp = np.clip(3.5 + rng.normal(0, 0.08, n), 2.0, 6.0)
    disc = 70.0 * comp + rng.normal(0, 4, n)
    fan  = 340.0 * comp + rng.normal(0, 30, n)
    temp = 18.0 - 2.0 * comp + rng.normal(0, 0.3, n)

    if "refrigerant" in scenario:
        disc -= disc.mean() * 0.40          # pressure drops 40 %
        temp += 7.0                          # temp spikes

    elif "fan" in scenario:
        fan  -= fan.mean() * 0.80           # fan RPM collapses
        ramp  = np.linspace(0, 1, n)
        comp += ramp * 1.5                   # compressor overworks
        disc += ramp * 60.0
        temp += ramp * 4.0

    elif "compressor" in scenario:
        ramp  = np.linspace(0, 1, n)
        comp += ramp * 1.8                   # power creep
        disc -= ramp * 50.0                  # pressure slowly drops
        temp += ramp * 3.5

    return pd.DataFrame({
        "compressor_power_kw":    comp,
        "discharge_pressure_psi": disc,
        "fan_rpm":                fan,
        "supply_air_temp_c":      temp,
    })


# -- Session state -------------------------------------------------------------

if "signal" not in st.session_state:
    st.session_state.signal = _normal_signal(200)
if "fault_at" not in st.session_state:
    st.session_state.fault_at = None
if "last_count" not in st.session_state:
    st.session_state.last_count = 0


# -- Backend helpers -----------------------------------------------------------

def _get_alerts() -> list:
    try:
        r = requests.get(f"{BACKEND}/alerts", timeout=1.5)
        return r.json().get("alerts", [])
    except Exception:
        return []


def _trigger(scenario: str, tag: str):
    try:
        requests.post(f"{BACKEND}/demo/{scenario}", timeout=2)
    except Exception:
        st.error("Backend unreachable -- start it with: python backend/app.py")
        return
    fault_df = _fault_signal(tag)
    st.session_state.fault_at = len(st.session_state.signal)
    st.session_state.signal = pd.concat(
        [st.session_state.signal, fault_df], ignore_index=True
    ).iloc[-350:]
    time.sleep(0.4)
    st.rerun()



# -- Sidebar: severity profile ------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Severity Profile")
    profile_labels = {
        "hospital":         "🏥 Hospital / Critical Care",
        "cold_chain":       "❄️ Cold Chain / Food Storage",
        "commercial_office":"🏢 Commercial Office (default)",
        "warehouse":        "🏭 Warehouse / Industrial",
    }
    selected_key = st.radio(
        "Operational context",
        options=list(profile_labels.keys()),
        format_func=lambda k: profile_labels[k],
        index=2,
    )
    _clf = SeverityClassifier(profile=selected_key)
    _warn_t, _crit_t = _clf.thresholds()
    st.markdown(f"""
    <div style="background:#161B22;border:1px solid #30363D;border-radius:8px;padding:.8rem;margin-top:.5rem">
      <div style="color:#8B949E;font-size:.7rem;text-transform:uppercase;letter-spacing:1px">Thresholds</div>
      <div style="color:#FF9800;font-size:.9rem;margin-top:.3rem">⚠️ Warning  ≥ {_warn_t}</div>
      <div style="color:#F44336;font-size:.9rem">🚨 Critical ≥ {_crit_t}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Unit Baselines**")
    import json
    from pathlib import Path as _P
    _bdir = _P("model/checkpoints/unit_baselines")
    if _bdir.exists():
        for _bf in sorted(_bdir.glob("*.json")):
            with open(_bf) as _f:
                _r = json.load(_f)
            st.markdown(
                f"<div style='background:#161B22;border:1px solid #30363D;"
                f"border-radius:6px;padding:.5rem .7rem;margin-bottom:.4rem'>"
                f"<div style='color:#4C9BE8;font-size:.72rem;font-weight:700;"
                f"text-transform:uppercase'>{_r['machine_id']}</div>"
                f"<div style='color:#8B949E;font-size:.75rem;margin-top:.2rem'>"
                f"Disc/Comp: <span style='color:#E6EDF3'>{_r['k_disc']:.1f}</span> &nbsp;"
                f"Fan/Comp: <span style='color:#E6EDF3'>{_r['k_fan']:.1f}</span><br>"
                f"Temp slope: <span style='color:#E6EDF3'>{_r['k_temp_b']:.3f}</span> &nbsp;"
                f"Temp base: <span style='color:#E6EDF3'>{_r['k_temp_a']:.1f}°C</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='color:#8B949E;font-size:.8rem'>Run preprocess.py first</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "<div style='color:#8B949E;font-size:.8rem'>"
        "Thermo-Twin v2 · Autoencoder + SHAP + MC-Dropout + Physics-Informed Loss"
        "</div>",
        unsafe_allow_html=True,
    )
# -- Header --------------------------------------------------------------------

st.markdown('<p class="header-title">🏭 Thermo-Twin | Carrier HVAC Diagnostics</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="header-sub">Unsupervised Thermodynamic Fault Isolation -- '
    'Powered by Autoencoder + SHAP + MC-Dropout</p>', unsafe_allow_html=True)
st.markdown('<div class="carrier-bar"></div>', unsafe_allow_html=True)

# -- Demo trigger buttons ------------------------------------------------------

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🔴  Trigger: Refrigerant Leak", width="stretch"):
        _trigger("scenario_1_refrigerant_leak", "refrigerant")
with c2:
    if st.button("🟡  Trigger: Fan Failure", width="stretch"):
        _trigger("scenario_2_fan_failure", "fan")
with c3:
    if st.button("🔵  Trigger: Compressor Wear", width="stretch"):
        _trigger("scenario_3_compressor_wear", "compressor")

# -- Fetch latest state --------------------------------------------------------

alerts = _get_alerts()
latest = alerts[0] if alerts else None
st.session_state.last_count = len(alerts)

# -- Live signal plot ----------------------------------------------------------

sig   = st.session_state.signal
n_pts = len(sig)
xs    = list(range(n_pts))

expl = (latest or {}).get("explanation", {})
pcts = {
    "compressor_power_kw":    expl.get("compressor_power_pct",   0),
    "discharge_pressure_psi": expl.get("discharge_pressure_pct", 0),
    "fan_rpm":                expl.get("fan_rpm_pct",            0),
    "supply_air_temp_c":      expl.get("supply_air_temp_pct",    0),
}
max_pct = max(pcts.values()) if pcts else 0
sev     = (latest or {}).get("severity_score", 0)
anomalous = {k for k, v in pcts.items() if v >= max_pct * 0.6} if sev > 40 else set()

STREAM_META = [
    ("compressor_power_kw",    "Compressor Power (kW)",       "#4C9BE8"),
    ("discharge_pressure_psi", "Discharge Pressure (PSI)",    "#4CAF50"),
    ("fan_rpm",                "Fan RPM",                     "#FF9800"),
    ("supply_air_temp_c",      "Supply Air Temp (°C)",        "#e05bfd"),
]

fig = go.Figure()
for col, label, base_color in STREAM_META:
    is_anom = col in anomalous
    color   = "#F44336" if is_anom else base_color
    width   = 2.8 if is_anom else 1.5
    opacity = 1.0 if (is_anom or not anomalous) else 0.35
    fig.add_trace(go.Scatter(
        x=xs, y=_norm(sig[col], col).tolist(),
        mode="lines", name=label,
        line=dict(color=color, width=width),
        opacity=opacity,
    ))

fault_at = st.session_state.fault_at
if fault_at is not None and fault_at < n_pts:
    fault_label = (latest or {}).get("fault_type", "Fault Detected")
    fig.add_vline(
        x=fault_at, line_dash="dash", line_color="#FF5722", line_width=2,
        annotation_text=f"⚡ {fault_label}",
        annotation_font=dict(color="#FF5722", size=12),
        annotation_position="top left",
        annotation_bgcolor="rgba(22,27,34,0.75)",
        annotation_bordercolor="#FF5722",
        annotation_borderwidth=1,
        annotation_borderpad=4,
    )

fig.update_layout(
    paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
    font_color="#E6EDF3", height=320,
    margin=dict(l=45, r=20, t=40, b=30),
    title=dict(
        text="Live Sensor Streams -- Normalised (0 = min, 1 = max per stream)",
        font=dict(color="#8B949E", size=11), x=0,
    ),
    legend=dict(
        bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363D", borderwidth=1,
        font=dict(color="#E6EDF3", size=11),
        orientation="v",
        x=1.0, xanchor="right",
        y=1.0, yanchor="top",
    ),
    xaxis=dict(showgrid=True, gridcolor="#21262D",
               tickfont=dict(color="#8B949E"), title="Sample", title_font=dict(color="#8B949E", size=10)),
    yaxis=dict(showgrid=True, gridcolor="#21262D",
               tickfont=dict(color="#8B949E"),
               title="Normalised Value", title_font=dict(color="#8B949E", size=11),
               range=[-0.1, 1.25]),
)
st.plotly_chart(fig, width="stretch")

# -- Severity | SHAP | Machine info --------------------------------------------

col_sev, col_shap, col_info = st.columns([1, 2, 1])

# - Severity gauge -
with col_sev:
    st.markdown("**Severity Score**")
    uncertainty   = (latest or {}).get("uncertainty", None)
    confidence    = (latest or {}).get("confidence_pct", None)
    action_ovride = (latest or {}).get("action_override", None)

    _level, _act_label = _clf.classify(sev)
    if _level == "NORMAL":
        cls, col_hex = "sev-normal",   "#4CAF50"
    elif _level == "WARNING":
        cls, col_hex = "sev-warning",  "#FF9800"
    else:
        cls, col_hex = "sev-critical", "#F44336"
    act = action_ovride if action_ovride else _act_label

    unc_line  = (f'<div style="font-size:1.1rem;color:#8B949E;margin:-.1rem 0 .1rem">'
                 f'&plusmn;&thinsp;{uncertainty}</div>') if uncertainty is not None else ""
    conf_line = (f'<div style="font-size:0.7rem;color:#8B949E;margin:.0rem 0 .3rem">'
                 f'{confidence}% confidence</div>') if confidence is not None else ""

    st.markdown(f"""
    <div class="sev-card {cls}">
      <div style="font-size:3.8rem;font-weight:900;color:{col_hex};line-height:1.0">{sev}</div>
      {unc_line}
      {conf_line}
      <div style="font-size:0.72rem;color:#8B949E;margin:.2rem 0 .5rem">/ 100</div>
      <div style="font-size:0.82rem;font-weight:700;color:{col_hex}">{act}</div>
    </div>
    """, unsafe_allow_html=True)

# - SHAP bar chart -
with col_shap:
    st.markdown("**Fault Attribution -- Which Sensor Drove This?**")
    shap_vals = {
        "Compressor Power":   expl.get("compressor_power_pct",   25.0 if not latest else 0),
        "Discharge Pressure": expl.get("discharge_pressure_pct", 25.0 if not latest else 0),
        "Fan RPM":            expl.get("fan_rpm_pct",            25.0 if not latest else 0),
        "Supply Air Temp":    expl.get("supply_air_temp_pct",    25.0 if not latest else 0),
    }
    sorted_sv  = sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)
    bar_labels = [k for k, _ in sorted_sv]
    bar_values = [v for _, v in sorted_sv]
    bar_colors = ["#F44336" if i == 0 else "#4C9BE8" for i in range(len(bar_values))]

    fig_shap = go.Figure(go.Bar(
        x=bar_values, y=bar_labels, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in bar_values],
        textposition="outside",
        textfont=dict(color="#E6EDF3", size=12),
    ))
    fig_shap.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font_color="#E6EDF3", height=210,
        margin=dict(l=0, r=55, t=10, b=0),
        xaxis=dict(
            range=[0, 115], showgrid=True, gridcolor="#21262D",
            tickfont=dict(color="#8B949E"),
            title="Attribution (%)", title_font=dict(color="#8B949E", size=11),
        ),
        yaxis=dict(tickfont=dict(color="#E6EDF3", size=12)),
    )
    st.plotly_chart(fig_shap, width="stretch")

# - Machine info -
with col_info:
    st.markdown("**Machine Info**")
    if latest:
        mid   = latest.get("machine_id", "—")
        ts    = latest.get("timestamp",  "—")
        ft    = latest.get("fault_type", "—")
        act   = latest.get("action",     "—")
        a_col = "#F44336" if act in ("STOP UNIT", "INVESTIGATE") else ("#FF9800" if act == "WARNING" else "#4CAF50")
        st.markdown(f"""
        <div class="mcard">
          <div class="mlabel">Machine</div>
          <div class="mvalue">{mid}</div>
          <div class="mlabel">Last Alert</div>
          <div class="mvalue" style="font-size:.82rem">{ts}</div>
          <div class="fault-big">{ft}</div>
          <div style="color:{a_col};font-weight:700;font-size:.95rem;margin-top:.3rem">{act}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mcard">
          <div style="color:#8B949E;font-style:italic;font-size:.9rem">
            No alerts yet.<br>Trigger a scenario above.
          </div>
        </div>
        """, unsafe_allow_html=True)

# -- Prescription card ---------------------------------------------------------

st.markdown("---")
if latest and sev > 40:
    presc = latest.get("prescription", {})
    p_cls = "presc-critical" if sev > 70 else "presc-warning"
    icon  = "🚨" if sev > 70 else "⚠️"
    st.markdown(f"""
    <div class="presc-card {p_cls}">
      <div style="font-size:1.05rem;font-weight:800;color:#E6EDF3;margin-bottom:.9rem">
        {icon}&nbsp; Prescriptive Maintenance Alert
      </div>
      <div class="plabel">Fault</div>
      <div class="pvalue">{presc.get("fault", "—")}</div>
      <div class="plabel">Impact</div>
      <div class="pvalue">{presc.get("impact", "—")}</div>
      <div class="plabel">Prescription</div>
      <div class="pvalue" style="color:#FF9800;font-weight:700">{presc.get("action", "—")}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="presc-normal">
      <span style="color:#8B949E;font-size:.9rem">
        ✅ No active fault -- system operating normally.
        Trigger a demo scenario to see prescriptive output.
      </span>
    </div>
    """, unsafe_allow_html=True)

# -- Energy cost card ----------------------------------------------------------

if latest and sev > 40:
    ec = latest.get("energy_cost", {})
    if ec:
        kwh      = ec.get("energy_waste_kwh_per_hr", 0)
        inr_day  = ec.get("cost_per_day_inr", 0)
        usd_day  = ec.get("cost_per_day_usd", 0)
        inr_mon  = ec.get("cost_per_month_inr", 0)
        part     = ec.get("part_cost_inr", 0)
        payback  = ec.get("payback_days", 0)
        eff_loss = ec.get("efficiency_loss_pct", 0)
        st.markdown(f"""
        <div style="background:#0D1E10;border:1px solid #2E7D32;border-left:6px solid #4CAF50;
                    border-radius:10px;padding:1.2rem 1.6rem;margin-top:.5rem">
          <div style="font-size:1.05rem;font-weight:800;color:#E6EDF3;margin-bottom:.9rem">
            💰&nbsp; Energy Cost Impact
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem">
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Efficiency Loss</div>
              <div style="font-size:1.1rem;font-weight:700;color:#F44336">{eff_loss}%</div>
            </div>
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Wasted Energy</div>
              <div style="font-size:1.1rem;font-weight:700;color:#FF9800">{kwh} kWh/hr</div>
            </div>
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Payback Period</div>
              <div style="font-size:1.1rem;font-weight:700;color:#4CAF50">{payback} days</div>
            </div>
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Cost Today</div>
              <div style="font-size:1.1rem;font-weight:700;color:#E6EDF3">&#8377;{inr_day:,.0f} / ${usd_day}</div>
            </div>
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Cost This Month</div>
              <div style="font-size:1.1rem;font-weight:700;color:#E6EDF3">&#8377;{inr_mon:,.0f}</div>
            </div>
            <div>
              <div style="font-size:.7rem;font-weight:700;letter-spacing:1.5px;color:#8B949E;text-transform:uppercase">Fix Cost (Part)</div>
              <div style="font-size:1.1rem;font-weight:700;color:#E6EDF3">~&#8377;{part:,}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# -- Alert log table -----------------------------------------------------------

st.markdown("---")
st.markdown("**Alert Log -- Last 10 Events**")

if alerts:
    _SEN_MAP = {
        "compressor_power_pct":   "Compressor Power",
        "discharge_pressure_pct": "Discharge Pressure",
        "fan_rpm_pct":            "Fan RPM",
        "supply_air_temp_pct":    "Supply Air Temp",
    }

    def _dominant(e: dict) -> str:
        if not e:
            return "—"
        best = max(_SEN_MAP, key=lambda k: e.get(k, 0))
        return f"{_SEN_MAP[best]} ({e.get(best, 0):.1f}%)"

    rows = []
    for a in alerts[:10]:
        e   = a.get("explanation", {})
        unc = a.get("uncertainty", None)
        sev_display = str(a.get("severity_score", 0))
        if unc is not None:
            sev_display = f"{a.get('severity_score', 0)} ±{unc}"
        rows.append({
            "Time":            a.get("timestamp", "—")[-8:],
            "Machine":         a.get("machine_id", "—"),
            "Fault Type":      a.get("fault_type", "—"),
            "Severity":        a.get("severity_score", 0),
            "Confidence":      f"{a.get('confidence_pct', '—')}%" if a.get("confidence_pct") else "—",
            "Action":          a.get("action", "—"),
            "Dominant Sensor": _dominant(e),
        })

    df_log = pd.DataFrame(rows)

    def _colour_sev(val):
        try:
            s = int(val)
        except Exception:
            return ""
        if s <= 40:  return "background-color:#0d2b0d;color:#4CAF50"
        if s <= 70:  return "background-color:#2b1a00;color:#FF9800"
        return "background-color:#2b0505;color:#F44336"

    styled = df_log.style.map(_colour_sev, subset=["Severity"])
    st.dataframe(styled, width="stretch", hide_index=True)
else:
    st.markdown(
        "<div style='color:#8B949E;font-style:italic;padding:.5rem'>No alerts yet.</div>",
        unsafe_allow_html=True,
    )

# -- Auto-poll every 2 seconds -------------------------------------------------

time.sleep(2)
st.rerun()