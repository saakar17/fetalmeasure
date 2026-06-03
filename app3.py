"""
UltraMeasure — Fetal Growth Assessment
Streamlit wizard application — enhanced clinical UI.

Run locally:
    streamlit run app.py

File structure expected:
    ultrameasure/
        app.py
        pipeline.py
        growth_standards.py
        models/
            best_model1.pth
            best_seg_model.pth
"""

import os
import io
import time
import zipfile
import tempfile
import datetime
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pipeline         import InferencePipeline, estimate_ac
from growth_standards import interpret_ac, get_chart_data, lmp_to_ga_weeks

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title           = "UltraMeasure — Fetal Growth Assessment",
    page_icon            = "🔬",
    layout               = "wide",
    initial_sidebar_state= "collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 16px; }

  :root {
    --navy       : #050d1f;
    --navy-2     : #0a1628;
    --navy-3     : #0f2040;
    --navy-4     : #162a52;
    --electric   : #2d7cf6;
    --electric-2 : #5b9cf8;
    --electric-3 : #a8ccfc;
    --green      : #00c48c;
    --amber      : #f59e0b;
    --red        : #f43f5e;
    --border     : rgba(45,124,246,0.18);
    --border-soft: rgba(255,255,255,0.07);
    --text       : #f0f4ff;
    --text-soft  : #8fa4c8;
    --text-muted : #4a5d80;
  }

  .stApp, .main,
  div[data-testid="stAppViewContainer"],
  div[data-testid="stVerticalBlock"] {
    background-color: var(--navy) !important;
  }

  div[data-testid="stForm"],
  div[data-testid="column"],
  div[data-testid="stVerticalBlock"] > div {
    background: transparent !important;
  }

  p, span, label, div, li, td, th,
  .stMarkdown, .stText { color: var(--text) !important; }

  input[type="text"], input[type="number"], textarea,
  div[data-baseweb="input"] input {
    background: var(--navy-3) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
  }

  label[data-testid="stWidgetLabel"] p,
  .stTextInput label, .stNumberInput label,
  .stDateInput label, .stSelectbox label {
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important; color: var(--text-soft) !important;
    text-transform: uppercase !important;
  }

  .stButton > button {
    background: var(--navy-3) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 14px 28px !important; transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    background: var(--navy-4) !important;
    border-color: var(--electric) !important; transform: translateY(-1px) !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--electric) !important;
    border-color: var(--electric) !important; color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #1a6ae8 !important;
    box-shadow: 0 4px 24px rgba(45,124,246,0.4) !important;
  }

  div[data-testid="stFileUploader"] {
    background: var(--navy-3) !important;
    border: 2px dashed var(--border) !important; border-radius: 16px !important;
    padding: 32px !important;
  }

  div[data-testid="stProgressBar"] > div {
    background: var(--navy-3) !important; border-radius: 8px !important;
  }
  div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--electric), var(--electric-2)) !important;
    border-radius: 8px !important;
  }

  div[data-testid="stAlert"] {
    border-radius: 12px !important; border: 1px solid var(--border) !important;
    background: var(--navy-3) !important;
  }

  div[data-baseweb="select"] > div {
    background: var(--navy-3) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
  }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--navy); }
  ::-webkit-scrollbar-thumb { background: var(--navy-4); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--electric); }

  /* ── Header ── */
  .um-header {
    background: linear-gradient(135deg, var(--navy-2) 0%, var(--navy-4) 100%);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 32px 40px; margin-bottom: 36px;
    position: relative; overflow: hidden;
  }
  .um-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(45,124,246,0.15) 0%, transparent 70%);
    pointer-events: none;
  }
  .um-header h1 {
    font-size: 34px !important; font-weight: 800 !important;
    margin: 0 0 6px !important; color: white !important; letter-spacing: -0.5px;
  }
  .um-header p {
    font-size: 15px !important; margin: 0 !important;
    color: var(--electric-3) !important;
  }
  .um-header .tag {
    display: inline-block;
    background: rgba(45,124,246,0.2); border: 1px solid rgba(45,124,246,0.4);
    border-radius: 20px; padding: 3px 12px; font-size: 11px;
    font-weight: 600; color: var(--electric-2);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px;
  }

  /* ── Step indicator ── */
  .step-bar {
    display: flex; align-items: center; margin-bottom: 40px;
    background: var(--navy-2); border: 1px solid var(--border-soft);
    border-radius: 16px; padding: 16px 24px; gap: 0;
  }
  .step-item { display: flex; align-items: center; gap: 10px; flex: 1; }
  .step-circle {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; flex-shrink: 0;
  }
  .step-circle.done  { background: var(--green); color: #001a0f; font-size: 16px; }
  .step-circle.active {
    background: var(--electric); color: white;
    box-shadow: 0 0 0 4px rgba(45,124,246,0.25), 0 0 20px rgba(45,124,246,0.3);
  }
  .step-circle.todo  {
    background: var(--navy-4); color: var(--text-muted);
    border: 1.5px solid var(--border-soft);
  }
  .step-label { font-size: 13px; font-weight: 600; color: var(--text-muted); }
  .step-label.active { color: var(--electric-2); }
  .step-label.done   { color: var(--green); }
  .step-connector { flex: 1; height: 2px; background: var(--border-soft); margin: 0 6px; }
  .step-connector.done { background: var(--green); }

  /* ── Cards ── */
  .um-card {
    background: var(--navy-2); border: 1px solid var(--border-soft);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
  }
  .um-card-title {
    font-size: 18px; font-weight: 700; color: var(--text) !important;
    margin-bottom: 16px; display: flex; align-items: center;
    gap: 10px; letter-spacing: -0.2px;
  }

  /* ── Badges ── */
  .badge {
    display: inline-block; padding: 6px 18px; border-radius: 24px;
    font-size: 14px; font-weight: 700; letter-spacing: 0.3px;
  }
  .badge-green { background: rgba(0,196,140,0.15); color: #00c48c; border: 1px solid rgba(0,196,140,0.3); }
  .badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
  .badge-red   { background: rgba(244,63,94,0.15);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }
  .badge-gray  { background: var(--navy-3); color: var(--text-soft); border: 1px solid var(--border); }

  /* ── Metric block ── */
  .metric-block {
    background: var(--navy-3); border: 1px solid var(--border);
    border-left: 4px solid var(--electric); border-radius: 0 12px 12px 0;
    padding: 18px 24px; margin-bottom: 14px;
  }
  .metric-block .label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--electric-2); margin-bottom: 4px;
  }
  .metric-block .value {
    font-size: 36px; font-weight: 800; color: white; letter-spacing: -1px; line-height: 1.1;
  }
  .metric-block .unit { font-size: 16px; color: var(--text-soft); margin-left: 6px; }

  /* ── Inference stats bar ── */
  .infer-bar {
    background: var(--navy-3); border: 1px solid var(--border);
    border-left: 4px solid var(--green); border-radius: 0 12px 12px 0;
    padding: 12px 20px; margin-bottom: 14px;
    display: flex; gap: 32px; flex-wrap: wrap; align-items: center;
  }
  .infer-bar .stat { text-align: center; }
  .infer-bar .stat .v {
    font-size: 22px; font-weight: 800; color: var(--green); line-height: 1.1;
  }
  .infer-bar .stat .l {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--text-muted); margin-top: 2px;
  }

  /* ── Patient bar ── */
  .patient-bar {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 28px; margin-bottom: 24px;
    display: flex; gap: 40px; flex-wrap: wrap;
  }
  .patient-bar .field-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 4px;
  }
  .patient-bar .field-value { font-size: 16px; font-weight: 600; color: var(--text); }

  /* ── Percentile gauge ── */
  .pct-gauge-wrap {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 22px; margin: 16px 0;
  }
  .pct-gauge-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--electric-2); margin-bottom: 10px;
  }
  .pct-track {
    position: relative; height: 18px; background: var(--navy-4);
    border-radius: 9px; overflow: visible; margin: 8px 0;
  }
  .pct-fill {
    height: 100%; border-radius: 9px;
    transition: width 0.6s ease;
  }
  .pct-marker {
    position: absolute; top: -4px; width: 26px; height: 26px;
    background: white; border-radius: 50%; transform: translateX(-50%);
    border: 3px solid var(--electric); box-shadow: 0 0 10px rgba(45,124,246,0.5);
  }
  .pct-labels {
    display: flex; justify-content: space-between;
    font-size: 10px; color: var(--text-muted); margin-top: 6px;
  }
  .pct-value-label {
    font-size: 13px; font-weight: 700; text-align: center; margin-top: 6px;
  }

  /* ── Seg image card ── */
  .seg-img-card {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px; margin-bottom: 8px;
  }
  .seg-img-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--electric-2); margin-bottom: 8px;
  }
  .seg-img-desc {
    font-size: 11px; color: var(--text-muted); margin-top: 6px; line-height: 1.5;
  }

  /* ── Equation box ── */
  .eq-box {
    background: var(--navy-4); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    color: var(--electric-3); line-height: 1.8;
  }
  .eq-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 8px;
  }

  /* ── Clinical expand ── */
  .clinical-expand {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 22px; margin-top: 14px;
    font-size: 14px; color: var(--text-soft); line-height: 1.8;
  }
  .clinical-expand strong { color: var(--text); }

  /* ── Interp box ── */
  .interp-box {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 22px; margin-top: 14px;
    font-size: 15px; color: var(--text-soft); line-height: 1.8;
  }

  /* ── Ref values ── */
  .ref-values {
    background: var(--navy-3); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 20px; margin-top: 14px;
    font-size: 13px; color: var(--text-soft); line-height: 1.7;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px; padding: 16px 20px; font-size: 13px;
    color: #fbbf24; margin-top: 20px; line-height: 1.6;
  }

  div[data-testid="stForm"] { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_pipeline():
    base     = os.path.dirname(__file__)
    cls_path = os.path.join(base, "models", "best_model1.pth")
    seg_path = os.path.join(base, "models", "best_seg_model.pth")
    if not os.path.exists(cls_path):
        st.error(f"Classifier weights not found at: {cls_path}")
        st.stop()
    if not os.path.exists(seg_path):
        st.error(f"Segmentation weights not found at: {seg_path}")
        st.stop()
    return InferencePipeline(cls_path, seg_path)


# ── Session state ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'step': 1, 'patient': {}, 'images': [],
        'image_names': [], 'results': None,
        'interpretation': None, 'ga_weeks': None,
        'timing': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Step indicator ─────────────────────────────────────────────────────────
def render_steps(current: int):
    steps = ["Patient Info", "Upload Scans", "Classification", "Segmentation", "Results"]
    html  = '<div class="step-bar">'
    for i, label in enumerate(steps, 1):
        if i < current:
            cc, lc, icon = "done",   "done",   "✓"
        elif i == current:
            cc, lc, icon = "active", "active", str(i)
        else:
            cc, lc, icon = "todo",   "",       str(i)
        html += (f'<div class="step-item">'
                 f'<div class="step-circle {cc}">{icon}</div>'
                 f'<span class="step-label {lc}">{label}</span></div>')
        if i < len(steps):
            conn = "done" if i < current else ""
            html += f'<div class="step-connector {conn}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Image loader ───────────────────────────────────────────────────────────
def load_images_from_upload(uploaded_files) -> tuple:
    images = []; names = []
    for uf in uploaded_files:
        name = uf.name.lower()
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(uf.read())) as z:
                entries = sorted([
                    n for n in z.namelist()
                    if n.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and not n.startswith('__MACOSX')
                ])
                for entry in entries:
                    try:
                        img = Image.open(io.BytesIO(z.read(entry))).convert("L")
                        images.append(img); names.append(os.path.basename(entry))
                    except Exception: pass

        elif name.endswith(('.mha', '.mhd')):
            try:
                import SimpleITK as sitk
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(name)[1], delete=False
                ) as tmp:
                    tmp.write(uf.read()); tmp_path = tmp.name
                volume = sitk.GetArrayFromImage(sitk.ReadImage(tmp_path))
                os.unlink(tmp_path)
                for i, slc in enumerate(volume):
                    slc_norm = ((slc - slc.min()) /
                                (slc.max() - slc.min() + 1e-8) * 255).astype(np.uint8)
                    images.append(Image.fromarray(slc_norm).convert("L"))
                    names.append(f"slice_{i+1:04d}.png")
            except ImportError:
                st.warning("SimpleITK required for MHA/MHD. Install: pip install SimpleITK")

        elif name.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(io.BytesIO(uf.read())).convert("L")
                images.append(img); names.append(uf.name)
            except Exception: pass

    return images, names


# ── Percentile gauge ───────────────────────────────────────────────────────
def render_percentile_gauge(ac_mm, p5, p50, p95, status):
    """Render a visual horizontal gauge showing where AC sits between 5th-95th pct."""
    colour_map = {
        'normal': ('#00c48c', 'rgba(0,196,140,0.25)'),
        'small' : ('#f43f5e', 'rgba(244,63,94,0.25)'),
        'large' : ('#f59e0b', 'rgba(245,158,11,0.25)'),
        'unknown': ('#8fa4c8', 'rgba(143,164,200,0.25)'),
    }
    col, bg = colour_map.get(status, colour_map['unknown'])

    # Clamp to [p5-20, p95+20] range for display
    lo = p5 - 20; hi = p95 + 20
    pct = max(0.0, min(1.0, (ac_mm - lo) / (hi - lo))) * 100
    norm_5  = max(0.0, min(1.0, (p5  - lo) / (hi - lo))) * 100
    norm_50 = max(0.0, min(1.0, (p50 - lo) / (hi - lo))) * 100
    norm_95 = max(0.0, min(1.0, (p95 - lo) / (hi - lo))) * 100

    deviation  = ac_mm - p50
    dev_label  = f"+{deviation:.1f} mm above median" if deviation >= 0 else f"{deviation:.1f} mm below median"

    st.markdown(f"""
    <div class="pct-gauge-wrap">
      <div class="pct-gauge-label">Percentile Position — Hadlock (1984) Reference</div>
      <div class="pct-track">
        <div class="pct-fill" style="width:{pct:.1f}%;background:{bg};border-radius:9px;height:100%"></div>
        <div class="pct-marker" style="left:{pct:.1f}%;border-color:{col}"></div>
      </div>
      <div class="pct-labels">
        <span>← SGA</span>
        <span>5th pct: {p5} mm</span>
        <span>50th pct: {p50} mm</span>
        <span>95th pct: {p95} mm</span>
        <span>LGA →</span>
      </div>
      <div class="pct-value-label" style="color:{col}">
        {ac_mm:.1f} mm &nbsp;·&nbsp; {dev_label}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Clinical expand content ────────────────────────────────────────────────
def render_clinical_expand(status):
    content = {
        'small': {
            'title': '🔴 What SGA means clinically',
            'text' : (
                "<strong>Fetal Growth Restriction (FGR)</strong> occurs when the placenta fails to "
                "deliver adequate nutrition and oxygen to the developing fetus. The fetal body responds "
                "through a mechanism called <strong>brain-sparing</strong> — blood flow is preferentially "
                "redirected to the brain and heart at the expense of abdominal organs. The fetal liver, "
                "which is the single largest contributor to abdominal circumference, depletes its glycogen "
                "reserves and shrinks in volume. This is why <strong>AC is the first biometric to fall in FGR</strong>, "
                "often before head circumference, biparietal diameter, or femur length change. "
                "<br><br>"
                "<strong>Recommended clinical actions:</strong> Doppler velocimetry of the umbilical artery "
                "to assess placental resistance, serial biometry every 2 weeks to monitor growth velocity, "
                "consideration of antenatal corticosteroids if early delivery is anticipated, and tertiary "
                "referral if umbilical artery end-diastolic flow is absent or reversed."
            )
        },
        'large': {
            'title': '🟡 What LGA means clinically',
            'text' : (
                "<strong>Macrosomia</strong> describes a fetus that is growing larger than expected for its "
                "gestational age. The most common cause is <strong>gestational diabetes mellitus (GDM)</strong> — "
                "maternal hyperglycaemia drives excess glucose across the placenta, stimulating fetal "
                "insulin secretion and promoting adipose deposition, particularly in the fetal abdomen and "
                "shoulders. AC tends to increase disproportionately relative to head circumference in "
                "macrosomic fetuses, making it the most sensitive biometric indicator for this condition. "
                "<br><br>"
                "<strong>Recommended clinical actions:</strong> Oral glucose tolerance test (OGTT) if not "
                "already performed, dietary modification and blood glucose monitoring, serial growth scans "
                "every 3–4 weeks, assessment of estimated fetal weight, and delivery planning to manage "
                "risks of shoulder dystocia and emergency caesarean section."
            )
        },
        'normal': {
            'title': '🟢 What AGA means',
            'text' : (
                "Appropriate for Gestational Age means the measured AC falls within the <strong>normal "
                "growth range</strong> — between the 5th and 95th percentile for this gestational age "
                "according to the Hadlock (1984) reference standard. "
                "Fetal abdominal growth is proceeding as expected. "
                "<br><br>"
                "This finding does not exclude all fetal pathology — some conditions affecting other "
                "biometric parameters may coexist — but it provides reassurance that fetal nutritional "
                "status and abdominal growth are within population norms. "
                "Routine antenatal follow-up is recommended."
            )
        },
    }
    if status not in content:
        return
    c = content[status]
    with st.expander(c['title'], expanded=False):
        st.markdown(
            f'<div class="clinical-expand">{c["text"]}</div>',
            unsafe_allow_html=True
        )


# ── Inference stats bar ────────────────────────────────────────────────────
def render_infer_stats(n_slices, cls_total_ms, seg_ms=None):
    per_slice_ms = cls_total_ms / max(n_slices, 1)
    stats = [
        ("Classifier", "ResNet-34  ·  21.3M params  ·  ~80 MB"),
        (f"{per_slice_ms:.1f} ms", "per slice (CPU)"),
        (f"{cls_total_ms:.0f} ms", f"total for {n_slices} slices"),
    ]
    if seg_ms is not None:
        stats.append((f"{seg_ms:.0f} ms", "segmentation (U-Net ~90 MB)"))

    html = '<div class="infer-bar">'
    for v, l in stats:
        html += f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── PDF report generator ───────────────────────────────────────────────────
def generate_pdf_report(patient, ga, results, interp, timing):
    """
    Generate a clinical PDF report using ReportLab.
    Returns bytes of the PDF.
    """
    from reportlab.lib.pagesizes   import A4
    from reportlab.lib.units       import mm, cm
    from reportlab.lib             import colors
    from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus        import (SimpleDocTemplate, Paragraph, Spacer,
                                           Table, TableStyle, HRFlowable,
                                           Image as RLImage, KeepTogether)
    from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus        import PageBreak

    # ── Colour palette ──
    NAVY      = colors.HexColor("#0A1628")
    BLUE      = colors.HexColor("#1E5FA8")
    ELECTRIC  = colors.HexColor("#2D7CF6")
    TEAL      = colors.HexColor("#0D9488")
    GREEN     = colors.HexColor("#00C48C")
    AMBER     = colors.HexColor("#D97706")
    RED       = colors.HexColor("#DC2626")
    LIGHT_BG  = colors.HexColor("#F0F4FF")
    MID_GRAY  = colors.HexColor("#64748B")
    DARK_TEXT = colors.HexColor("#0F172A")
    WHITE     = colors.white

    status_colour = {'normal': GREEN, 'small': RED, 'large': AMBER, 'unknown': MID_GRAY}
    status_label  = {
        'normal' : 'Appropriate for Gestational Age (AGA)',
        'small'  : 'Small for Gestational Age (SGA)',
        'large'  : 'Large for Gestational Age (LGA)',
        'unknown': 'Gestational Age Not Provided',
    }

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )

    W = A4[0] - 40*mm   # usable width

    styles   = getSampleStyleSheet()
    body     = ParagraphStyle('body',    fontSize=10, leading=15,
                               textColor=DARK_TEXT, alignment=TA_JUSTIFY)
    bold     = ParagraphStyle('bold',    parent=body, fontName='Helvetica-Bold')
    caption  = ParagraphStyle('caption', fontSize=8.5, leading=12,
                               textColor=MID_GRAY, alignment=TA_CENTER)
    heading2 = ParagraphStyle('h2', fontSize=12, fontName='Helvetica-Bold',
                               textColor=BLUE, spaceBefore=12, spaceAfter=4)
    small    = ParagraphStyle('small', fontSize=8.5, leading=12,
                               textColor=MID_GRAY)

    story = []

    # ── HEADER ─────────────────────────────────────────────────────────────
    # Navy header band
    hdr_data = [[
        Paragraph('<font color="white" size="18"><b>🔬 UltraMeasure</b></font>', styles['Normal']),
        Paragraph('<font color="#A8CCFC" size="8">AUTOMATED FETAL GROWTH ASSESSMENT<br/>Powered by Deep Learning · Hadlock (1984) Standards</font>',
                  ParagraphStyle('hdr_sub', alignment=TA_RIGHT)),
    ]]
    hdr_tbl = Table(hdr_data, colWidths=[W*0.6, W*0.4])
    hdr_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',(0,0), (-1,-1), 12),
        ('RIGHTPADDING',(0,0),(-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0),(-1,-1),14),
        ('ROUNDEDCORNERS', (0,0),(-1,-1), 6),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 8))

    # Blue accent line
    story.append(HRFlowable(width=W, thickness=2, color=ELECTRIC))
    story.append(Spacer(1, 10))

    # ── PATIENT DETAILS ────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", heading2))
    pat_data = [
        ["Patient Name",   patient['name'],       "Patient ID",    patient['id']],
        ["Scan Date",      patient['scan_date'],   "LMP",           patient['lmp']],
        ["Gestational Age",f"{ga:.1f} weeks" if ga else "Not provided",
         "Report Generated", datetime.date.today().strftime("%d %B %Y")],
    ]
    pat_tbl = Table(pat_data, colWidths=[W*0.2, W*0.3, W*0.2, W*0.3])
    pat_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), LIGHT_BG),
        ('BACKGROUND',    (0,0), (0,-1), colors.HexColor("#E2E8F0")),
        ('BACKGROUND',    (2,0), (2,-1), colors.HexColor("#E2E8F0")),
        ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9.5),
        ('TEXTCOLOR',     (0,0), (-1,-1), DARK_TEXT),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [LIGHT_BG, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    story.append(pat_tbl)
    story.append(Spacer(1, 14))

    # ── MEASUREMENT RESULTS ────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Measurement Results", heading2))

    ac_mm    = results.get('ac_mm')
    ellipse  = results.get('ellipse')

    # Large AC metric
    ac_str = f"{ac_mm:.1f} mm" if ac_mm else "Not available"
    ac_data = [
        [Paragraph(f'<font size="22" color="#1E5FA8"><b>{ac_str}</b></font>', styles['Normal']),
         Paragraph(f'<font size="9" color="#64748B">Abdominal Circumference<br/>'
                   f'<b>Best slice:</b> {results.get("best_name","—")}<br/>'
                   f'<b>Classifier confidence:</b> {results.get("best_prob",0):.1%}<br/>'
                   f'<b>Model:</b> ResNet-34 · 21.3M params</font>', styles['Normal'])],
    ]
    ac_tbl = Table(ac_data, colWidths=[W*0.45, W*0.55])
    ac_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), LIGHT_BG),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 14),
        ('BOX',           (0,0), (-1,-1), 1, ELECTRIC),
        ('ROUNDEDCORNERS',(0,0), (-1,-1), 4),
    ]))
    story.append(ac_tbl)
    story.append(Spacer(1, 10))

    # Ellipse parameters
    if ellipse is not None and ac_mm is not None:
        PIXEL_SPACING_MM = 0.930
        a_mm = (ellipse[1][0] / 2.0) * PIXEL_SPACING_MM
        b_mm = (ellipse[1][1] / 2.0) * PIXEL_SPACING_MM
        h    = ((a_mm - b_mm)**2) / ((a_mm + b_mm)**2)

        story.append(Paragraph("Geometric Measurement — Ellipse Fitting", heading2))
        eq_data = [
            ["Semi-axis a",         f"{a_mm:.2f} mm",
             "Semi-axis b",         f"{b_mm:.2f} mm"],
            ["h = (a−b)² / (a+b)²", f"{h:.6f}",
             "Scanner pixel spacing","0.28 mm/px (original)"],
            ["Effective spacing",    "0.930 mm/px (224×224 space)",
             "Formula", "Ramanujan's approximation"],
        ]
        eq_tbl = Table(eq_data, colWidths=[W*0.25, W*0.25, W*0.25, W*0.25])
        eq_tbl.setStyle(TableStyle([
            ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',      (2,0), (2,-1), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('TEXTCOLOR',     (0,0), (-1,-1), DARK_TEXT),
            ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ]))
        story.append(eq_tbl)

        story.append(Spacer(1, 6))
        formula_para = Paragraph(
            '<font name="Courier" size="9" color="#1E5FA8">'
            'AC = π × (a + b) × [1 + 3h / (10 + √(4 − 3h))]<br/>'
            f'AC = π × ({a_mm:.2f} + {b_mm:.2f}) × [1 + 3×{h:.4f} / (10 + √(4 − 3×{h:.4f}))]<br/>'
            f'AC = <b>{ac_mm:.2f} mm</b>'
            '</font>',
            ParagraphStyle('formula', fontSize=9, leading=14, leftIndent=10,
                           backColor=colors.HexColor("#EFF6FF"),
                           borderColor=ELECTRIC, borderWidth=1,
                           borderPadding=8)
        )
        story.append(formula_para)
        story.append(Spacer(1, 10))

    # Inference timing
    if timing:
        story.append(Paragraph("Pipeline Performance", heading2))
        timing_data = [["Metric", "Value", "Context"]]
        if 'cls_total_ms' in timing:
            n  = timing.get('n_slices', 1)
            ms = timing['cls_total_ms']
            timing_data += [
                ["Classification — total",  f"{ms:.0f} ms",    f"{n} slices @ {ms/n:.1f} ms/slice"],
                ["Classifier model",        "ResNet-34",       "21.3M parameters · ~80 MB"],
            ]
        if 'seg_ms' in timing:
            timing_data += [
                ["Segmentation",            f"{timing['seg_ms']:.0f} ms", "U-Net · ~90 MB · 1 slice"],
                ["Total pipeline",          f"{timing.get('total_ms', timing['cls_total_ms'] + timing['seg_ms']):.0f} ms",
                 "End-to-end on CPU"],
            ]
        t_tbl = Table(timing_data, colWidths=[W*0.38, W*0.22, W*0.4])
        t_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), BLUE),
            ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor("#F8FAFC"), LIGHT_BG]),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ]))
        story.append(t_tbl)
        story.append(Spacer(1, 14))

    # ── ULTRASOUND IMAGES ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(HRFlowable(width=W, thickness=2, color=TEAL))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Ultrasound Image Analysis", heading2))

    img_captions = [
        ("best_img",    "Original best slice\n(highest classifier confidence)"),
        ("overlay",     "Predicted segmentation mask\n(U-Net pixel-level output)"),
        ("prob_map",    "Probability heatmap\n(spatial confidence map)"),
        ("ellipse_img", "Fitted ellipse\n(geometric AC measurement boundary)"),
    ]

    # Render 2×2 image grid
    row_images = []
    row_caps   = []
    for key, cap in img_captions:
        val = results.get(key)
        if val is None:
            continue
        try:
            if key == 'prob_map':
                # Convert numpy heatmap to PIL
                fig_tmp, ax_tmp = plt.subplots(figsize=(3, 3))
                ax_tmp.imshow(val, cmap='hot', vmin=0, vmax=1)
                ax_tmp.axis('off')
                fig_tmp.tight_layout(pad=0)
                buf_tmp = io.BytesIO()
                fig_tmp.savefig(buf_tmp, format='PNG', dpi=120, bbox_inches='tight',
                                pad_inches=0)
                plt.close(fig_tmp)
                buf_tmp.seek(0)
                pil_img = Image.open(buf_tmp).convert("RGB")
            elif isinstance(val, np.ndarray):
                pil_img = Image.fromarray((val * 255).astype(np.uint8)).convert("RGB")
            else:
                pil_img = val.convert("RGB")

            # Resize to thumbnail
            pil_img.thumbnail((300, 300))
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format='PNG')
            img_buf.seek(0)
            rl_img = RLImage(img_buf, width=W*0.45, height=W*0.45)
            row_images.append(rl_img)
            row_caps.append(Paragraph(cap, caption))
        except Exception:
            row_images.append(Spacer(1, 1))
            row_caps.append(Paragraph(cap, caption))

    # Arrange as 2×2 table
    if len(row_images) >= 2:
        img_tbl_data = []
        cap_tbl_data = []
        for i in range(0, len(row_images), 2):
            r1 = row_images[i]
            r2 = row_images[i+1] if i+1 < len(row_images) else Spacer(1,1)
            c1 = row_caps[i]
            c2 = row_caps[i+1] if i+1 < len(row_caps) else Spacer(1,1)
            img_tbl_data.append([r1, r2])
            cap_tbl_data.append([c1, c2])

        for img_row, cap_row in zip(img_tbl_data, cap_tbl_data):
            combined = [[img_row[0], img_row[1]], [cap_row[0], cap_row[1]]]
            tbl = Table(combined, colWidths=[W*0.5, W*0.5])
            tbl.setStyle(TableStyle([
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING',    (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LINEBELOW',     (0,0), (-1,0), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))

    # ── CLINICAL INTERPRETATION ────────────────────────────────────────────
    story.append(PageBreak())
    story.append(HRFlowable(width=W, thickness=2, color=ELECTRIC))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Clinical Interpretation", heading2))

    if interp and interp['ref']['in_table']:
        ref    = interp['ref']
        sc     = status_colour.get(interp['status'], MID_GRAY)
        slabel = status_label.get(interp['status'], interp['label'])

        # Status badge row
        badge_data = [[
            Paragraph(f'<font size="14" color="white"><b>{slabel}</b></font>',
                      ParagraphStyle('badge', alignment=TA_CENTER)),
        ]]
        badge_tbl = Table(badge_data, colWidths=[W])
        badge_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), sc),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('ROUNDEDCORNERS',(0,0), (-1,-1), 4),
        ]))
        story.append(badge_tbl)
        story.append(Spacer(1, 10))

        # Reference table
        ref_data = [
            ["Gestational Age",  f"{ga:.1f} weeks",
             "Measured AC",      f"{ac_mm:.1f} mm" if ac_mm else "—"],
            ["5th Percentile",   f"{ref['p5']} mm",
             "50th Percentile",  f"{ref['p50']} mm"],
            ["95th Percentile",  f"{ref['p95']} mm",
             "Deviation from median",
             f"{(ac_mm - ref['p50']):.1f} mm" if ac_mm else "—"],
        ]
        ref_tbl = Table(ref_data, colWidths=[W*0.25, W*0.25, W*0.25, W*0.25])
        ref_tbl.setStyle(TableStyle([
            ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',      (2,0), (2,-1), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 9.5),
            ('TEXTCOLOR',     (0,0), (-1,-1), DARK_TEXT),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS',(0,0), (-1,-1), [LIGHT_BG, colors.HexColor("#F8FAFC")]),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ]))
        story.append(ref_tbl)
        story.append(Spacer(1, 10))

        # Interpretation narrative
        story.append(Paragraph("Clinical Narrative", heading2))
        story.append(Paragraph(interp['message'], body))
        story.append(Spacer(1, 10))

        # Biological significance
        story.append(Paragraph("Biological Significance", heading2))
        bio_texts = {
            'small': (
                "Abdominal circumference is the most sensitive biometric indicator of fetal growth "
                "restriction because the fetal liver — the primary anatomical contributor to AC — is "
                "the first organ to be compromised when placental perfusion is inadequate. The "
                "brain-sparing reflex redistributes cardiac output toward the fetal cerebral circulation "
                "at the expense of hepatic and splanchnic vessels, causing progressive depletion of "
                "hepatic glycogen stores and a measurable reduction in liver volume. This manifests as "
                "a falling AC centile before other biometric parameters are affected. Clinical correlation "
                "with umbilical artery Doppler and middle cerebral artery pulsatility index is recommended."
            ),
            'large': (
                "Large for Gestational Age AC is most commonly associated with gestational diabetes "
                "mellitus, in which maternal hyperglycaemia drives excess transplacental glucose delivery, "
                "stimulating fetal pancreatic insulin secretion and promoting adipose deposition — "
                "particularly in the fetal abdomen and shoulders. The disproportionate abdominal "
                "enlargement relative to head circumference (abdominal/head ratio) is a distinguishing "
                "feature of diabetic macrosomia. Risks include shoulder dystocia, neonatal hypoglycaemia, "
                "and increased caesarean section rate. Oral glucose tolerance test is indicated if not "
                "previously performed."
            ),
            'normal': (
                "The measured abdominal circumference is consistent with normal fetal growth for this "
                "gestational age. The AC reflects adequate hepatic glycogen stores and appropriate "
                "subcutaneous fat deposition, consistent with normal placental nutrient transfer. "
                "Routine antenatal surveillance is recommended. Serial growth assessment at standard "
                "intervals is advised to monitor growth trajectory."
            ),
        }
        bio_text = bio_texts.get(interp['status'], "")
        if bio_text:
            story.append(Paragraph(bio_text, body))
        story.append(Spacer(1, 10))

        # Reference citation
        story.append(Paragraph(
            "<i>Reference: Hadlock FP et al. (1984). Sonographic estimation of fetal age and weight. "
            "Radiology, 150(3), 535–540.</i>", small
        ))

    elif ac_mm:
        story.append(Paragraph(
            f"AC measured: {ac_mm:.1f} mm. Gestational age not provided — "
            "clinical interpretation against Hadlock reference not available.", body
        ))

    story.append(Spacer(1, 14))

    # ── DISCLAIMER ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 8))
    disc_data = [[
        Paragraph(
            '<font size="9" color="#92400E"><b>⚠ Clinical Disclaimer:</b> This report is generated '
            'by an automated deep learning system (UltraMeasure) for research and educational purposes '
            'only. It must not be used as a sole basis for clinical decisions. All measurements and '
            'interpretations must be verified by a qualified obstetrician or medical sonographer. '
            'The system has not been approved for clinical diagnostic use.</font>',
            ParagraphStyle('disc', fontSize=9, leading=13)
        )
    ]]
    disc_tbl = Table(disc_data, colWidths=[W])
    disc_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor("#FEF3C7")),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor("#D97706")),
    ]))
    story.append(disc_tbl)

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="um-header">
  <div class="tag">Fetal Growth Assessment · AI-Powered</div>
  <h1>🔬 UltraMeasure</h1>
  <p>Automated abdominal circumference measurement from ultrasound · Hadlock (1984) standards</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — PATIENT INFORMATION
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    render_steps(1)
    st.markdown('<div class="um-card"><div class="um-card-title">👤 Patient Information</div></div>',
                unsafe_allow_html=True)

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_id   = st.text_input("Patient ID *",   placeholder="e.g. PT-2024-001")
            patient_name = st.text_input("Patient Name *", placeholder="Full name")
        with col2:
            lmp_date  = st.date_input(
                "Last Menstrual Period (LMP)",
                value=None,
                min_value=datetime.date.today() - datetime.timedelta(days=300),
                max_value=datetime.date.today(),
                help="Used to calculate gestational age automatically",
            )
            ga_manual = st.number_input(
                "Gestational Age (weeks) — if LMP unknown",
                min_value=0.0, max_value=45.0, value=0.0, step=0.5,
                help="Leave at 0 if LMP is provided above",
            )

        scan_date = st.date_input("Scan Date", value=datetime.date.today(),
                                  max_value=datetime.date.today())
        submitted = st.form_submit_button("Continue to Upload →",
                                          use_container_width=True)
        if submitted:
            if not patient_id.strip() or not patient_name.strip():
                st.error("Patient ID and Patient Name are required.")
            else:
                ga = (lmp_to_ga_weeks(lmp_date, scan_date) if lmp_date is not None
                      else float(ga_manual) if ga_manual > 0 else None)
                st.session_state.patient = {
                    'id': patient_id.strip(), 'name': patient_name.strip(),
                    'lmp': str(lmp_date) if lmp_date else "Not provided",
                    'ga_weeks': ga, 'scan_date': str(scan_date),
                }
                st.session_state.ga_weeks = ga
                st.session_state.step     = 2
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — UPLOAD SCANS
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    render_steps(2)
    p = st.session_state.patient
    st.markdown(f"""
    <div class="um-card">
      <div class="um-card-title">📁 Upload Ultrasound Images</div>
      <p style="font-size:13px;color:#8fa4c8;margin:0">
        Patient: <strong style="color:#f0f4ff">{p['name']}</strong> &nbsp;·&nbsp;
        ID: <strong style="color:#f0f4ff">{p['id']}</strong> &nbsp;·&nbsp;
        GA: <strong style="color:#f0f4ff">{f"{p['ga_weeks']:.1f} weeks" if p['ga_weeks'] else "Not provided"}</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("**Supported formats:** PNG / JPG (individual slices) · "
            "ZIP (folder of images) · MHA / MHD (3-D volumetric acquisition — "
            "automatically decomposed into axial slices via SimpleITK)", icon="ℹ️")

    uploaded = st.file_uploader(
        "Choose files",
        type=['png', 'jpg', 'jpeg', 'zip', 'mha', 'mhd'],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Reading images…"):
            images, names = load_images_from_upload(uploaded)

        if not images:
            st.error("No valid images could be read from the uploaded files.")
        else:
            st.success(f"✓ {len(images)} slice{'s' if len(images)>1 else ''} loaded.")
            st.markdown('<div class="um-card-title" style="margin-top:16px">🖼 Uploaded Slices Preview</div>',
                        unsafe_allow_html=True)

            cols_per_row = 6
            for row_start in range(0, min(len(images), 30), cols_per_row):
                row_items = list(zip(images[row_start:row_start+cols_per_row],
                                     names[row_start:row_start+cols_per_row]))
                cols = st.columns(len(row_items))
                for col, (img, nm) in zip(cols, row_items):
                    col.image(img, caption=nm, use_container_width=True)

            if len(images) > 30:
                st.caption(f"Showing first 30 of {len(images)} slices.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back", use_container_width=True):
                    st.session_state.step = 1; st.rerun()
            with col2:
                if st.button("Run Classification →", use_container_width=True, type="primary"):
                    st.session_state.images      = images
                    st.session_state.image_names = names
                    st.session_state.step        = 3
                    st.rerun()
    else:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    render_steps(3)
    images = st.session_state.images
    names  = st.session_state.image_names

    st.markdown("""
    <div class="um-card">
      <div class="um-card-title">🧠 Classification — Slice Selection</div>
      <p style="font-size:13px;color:#8fa4c8;margin:0">
        ResNet-34 (21.3M params · pretrained ImageNet) scores every slice for
        fetal abdomen presence. The single highest-confidence slice is selected
        as the candidate for segmentation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline    = load_pipeline()
    progress_bar = st.progress(0, text="Classifying slices…")
    classified   = []

    import torch
    from pipeline import preprocess

    t_cls_start = time.time()
    with st.spinner(""):
        for i, img in enumerate(images):
            t    = preprocess(img).unsqueeze(0).to(pipeline.device)
            with torch.no_grad():
                prob = torch.sigmoid(pipeline.classifier(t)).item()
            classified.append((img, prob, names[i] if i < len(names) else f"slice_{i+1}"))
            progress_bar.progress((i+1)/len(images),
                                  text=f"Classifying slice {i+1} / {len(images)}…")
    cls_total_ms = (time.time() - t_cls_start) * 1000
    progress_bar.empty()

    classified_sorted          = sorted(classified, key=lambda x: x[1], reverse=True)
    best_img, best_prob, best_name = classified_sorted[0]
    all_probs                  = [x[1] for x in classified]

    st.success(f"✓ Classification complete — Best slice: **{best_name}** "
               f"(confidence: **{best_prob:.1%}**)")

    # ── Inference stats bar ────────────────────────────────────────────────
    render_infer_stats(len(images), cls_total_ms)

    # ── Confidence distribution chart ─────────────────────────────────────
    st.markdown('<div class="um-card-title" style="margin-top:8px">📊 Confidence Distribution Across All Slices</div>',
                unsafe_allow_html=True)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=all_probs, nbinsx=20,
        marker_color='#2d7cf6', opacity=0.85,
        name='All slices',
    ))
    fig_dist.add_vline(x=0.5, line_dash="dash", line_color="#f43f5e",
                       annotation_text="Threshold 0.5",
                       annotation_font_color="#f43f5e")
    fig_dist.add_vline(x=best_prob, line_dash="dot", line_color="#00c48c",
                       annotation_text=f"Best slice {best_prob:.1%}",
                       annotation_font_color="#00c48c")
    fig_dist.update_layout(
        xaxis_title="Classifier confidence score",
        yaxis_title="Number of slices",
        plot_bgcolor='#0a1628', paper_bgcolor='#0a1628',
        font_color='#8fa4c8',
        xaxis=dict(gridcolor='rgba(45,124,246,0.08)', color='#8fa4c8'),
        yaxis=dict(gridcolor='rgba(45,124,246,0.08)', color='#8fa4c8'),
        margin=dict(l=40, r=20, t=20, b=40), height=240,
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Slice grid ─────────────────────────────────────────────────────────
    st.markdown('<div class="um-card-title">🖼 All Slices — Scored</div>',
                unsafe_allow_html=True)
    cols_per_row = 5
    for row_start in range(0, min(len(classified), 25), cols_per_row):
        row_items = classified[row_start:row_start+cols_per_row]
        cols      = st.columns(len(row_items))
        for col, (img, prob, nm) in zip(cols, row_items):
            is_best = (img is best_img)
            col.image(img, use_container_width=True)
            colour = "#00c48c" if prob >= 0.5 else "#4a5d80"
            col.markdown(
                f"<p style='font-size:11px;text-align:center;color:{colour};margin:0'>"
                f"{'⭐ ' if is_best else ''}{prob:.1%}</p>",
                unsafe_allow_html=True
            )
    if len(classified) > 25:
        st.caption(f"Showing first 25 of {len(classified)} slices.")

    # ── Best slice ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="um-card-title">⭐ Selected Best Slice</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(best_img, caption=f"{best_name} · {best_prob:.1%}",
                 use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="metric-block">
          <div class="label">Classification Confidence</div>
          <div class="value">{best_prob:.1%}</div>
        </div>
        <div class="metric-block">
          <div class="label">Slice Selected</div>
          <div class="value" style="font-size:16px">{best_name}</div>
        </div>
        <div class="metric-block">
          <div class="label">Total Slices Scanned</div>
          <div class="value">{len(images)}</div>
        </div>
        """, unsafe_allow_html=True)
        if best_prob < 0.5:
            st.warning("⚠️ No slice exceeded the 0.5 threshold. "
                       "The best available slice is shown but results may be unreliable.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2; st.rerun()
    with col2:
        if st.button("Run Segmentation →", use_container_width=True, type="primary"):
            st.session_state.results = {
                'best_img': best_img, 'best_prob': best_prob,
                'best_name': best_name, 'classified': classified_sorted,
            }
            st.session_state.timing['cls_total_ms'] = cls_total_ms
            st.session_state.timing['n_slices']     = len(images)
            st.session_state.step = 4
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    render_steps(4)
    best_img  = st.session_state.results['best_img']
    best_name = st.session_state.results['best_name']
    best_prob = st.session_state.results['best_prob']

    st.markdown("""
    <div class="um-card">
      <div class="um-card-title">🔬 Segmentation — U-Net Pixel-Level Analysis</div>
      <p style="font-size:13px;color:#8fa4c8;margin:0">
        U-Net with pretrained ResNet-34 encoder (~90 MB · Dice 0.9409 on test set).
        Ellipse fitted via OpenCV fitEllipse. AC computed via Ramanujan's approximation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = load_pipeline()

    t_seg_start = time.time()
    with st.spinner("Running segmentation…"):
        prob_map, binary_mask = pipeline.segment(best_img)
        overlay               = pipeline.make_overlay(best_img, binary_mask)
    seg_ms = (time.time() - t_seg_start) * 1000

    ac_mm, ellipse, _ = estimate_ac(binary_mask)
    ellipse_img = (pipeline.draw_ellipse(best_img, ellipse)
                   if ellipse is not None else overlay)

    if ellipse is None:
        st.warning("⚠️ Ellipse fitting failed — mask contour may be too small.")

    st.session_state.results.update({
        'prob_map': prob_map, 'binary_mask': binary_mask,
        'overlay': overlay, 'ellipse_img': ellipse_img,
        'ellipse': ellipse, 'ac_mm': ac_mm,
    })
    st.session_state.timing['seg_ms']   = seg_ms
    st.session_state.timing['total_ms'] = (
        st.session_state.timing.get('cls_total_ms', 0) + seg_ms
    )

    # ── Inference stats ────────────────────────────────────────────────────
    render_infer_stats(
        st.session_state.timing.get('n_slices', len(st.session_state.images)),
        st.session_state.timing.get('cls_total_ms', 0),
        seg_ms=seg_ms
    )

    # ── Four image outputs with clinical labels ────────────────────────────
    st.markdown('<div class="um-card-title" style="margin-top:8px">🔍 Segmentation Output</div>',
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    col1.image(best_img, use_container_width=True)
    col1.markdown("""
    <div class="seg-img-title">Original slice</div>
    <div class="seg-img-desc">Best slice selected by ResNet-34 classifier.
    Highest confidence score from the uploaded stack.</div>
    """, unsafe_allow_html=True)

    col2.image(overlay, use_container_width=True)
    col2.markdown("""
    <div class="seg-img-title">Predicted mask overlay</div>
    <div class="seg-img-desc">U-Net pixel-level foreground prediction.
    Teal region = predicted fetal abdominal boundary. Dice 0.9409 on test set.</div>
    """, unsafe_allow_html=True)

    fig_h, ax_h = plt.subplots(figsize=(3, 3))
    ax_h.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax_h.axis('off')
    fig_h.tight_layout(pad=0)
    col3.pyplot(fig_h, use_container_width=True)
    plt.close(fig_h)
    col3.markdown("""
    <div class="seg-img-title">Probability heatmap</div>
    <div class="seg-img-desc">Raw sigmoid output per pixel before thresholding.
    Warmer = higher foreground confidence. Shows spatial uncertainty at boundaries.</div>
    """, unsafe_allow_html=True)

    col4.image(ellipse_img, use_container_width=True)
    col4.markdown("""
    <div class="seg-img-title">Fitted ellipse</div>
    <div class="seg-img-desc">Least-squares ellipse fitted to the largest contour.
    Green boundary = geometric measurement used for AC calculation.</div>
    """, unsafe_allow_html=True)

    # ── Ellipse parameters + equation ─────────────────────────────────────
    if ellipse is not None and ac_mm is not None:
        PIXEL_SPACING_MM = 0.930
        a_mm = (ellipse[1][0] / 2.0) * PIXEL_SPACING_MM
        b_mm = (ellipse[1][1] / 2.0) * PIXEL_SPACING_MM
        h    = ((a_mm - b_mm)**2) / ((a_mm + b_mm)**2)

        st.markdown("---")
        st.markdown('<div class="um-card-title">📐 Geometric Measurement Chain</div>',
                    unsafe_allow_html=True)

        gc1, gc2, gc3, gc4 = st.columns(4)
        for col, label, val in [
            (gc1, "Semi-axis a", f"{a_mm:.2f} mm"),
            (gc2, "Semi-axis b", f"{b_mm:.2f} mm"),
            (gc3, "h parameter", f"{h:.6f}"),
            (gc4, "Pixel spacing", "0.930 mm/px"),
        ]:
            col.markdown(f"""
            <div class="metric-block">
              <div class="label">{label}</div>
              <div class="value" style="font-size:22px">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="eq-box">
          <div class="eq-label">Ramanujan's Approximation — Ellipse Perimeter</div>
          AC = π × (a + b) × [1 + 3h / (10 + √(4 − 3h))]<br>
          AC = π × ({a_mm:.2f} + {b_mm:.2f}) × [1 + 3 × {h:.4f} / (10 + √(4 − 3 × {h:.4f}))]<br>
          <strong>AC = {ac_mm:.2f} mm</strong>
          &nbsp;&nbsp;·&nbsp;&nbsp;
          Pixel spacing correction: 0.28 mm/px ÷ (224/744) = 0.930 mm/px (effective in 224×224 space)
        </div>
        """, unsafe_allow_html=True)

    # ── AC result ──────────────────────────────────────────────────────────
    st.markdown("---")
    if ac_mm is not None:
        st.markdown(f"""
        <div class="metric-block">
          <div class="label">Estimated Abdominal Circumference</div>
          <div class="value">{ac_mm:.1f}<span class="unit">mm</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("AC estimation failed — no valid contour detected.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 3; st.rerun()
    with col2:
        if st.button("View Clinical Results →", use_container_width=True, type="primary"):
            ga    = st.session_state.ga_weeks
            interp = interpret_ac(ac_mm, ga) if ac_mm and ga else None
            st.session_state.interpretation = interp
            st.session_state.step = 5
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — CLINICAL RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    render_steps(5)

    p      = st.session_state.patient
    res    = st.session_state.results
    interp = st.session_state.interpretation
    timing = st.session_state.timing
    ac_mm  = res.get('ac_mm')
    ga     = st.session_state.ga_weeks

    # ── Patient bar ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="patient-bar">
      <div><div class="field-label">Patient</div>
           <div class="field-value">{p['name']}</div></div>
      <div><div class="field-label">ID</div>
           <div class="field-value">{p['id']}</div></div>
      <div><div class="field-label">Scan Date</div>
           <div class="field-value">{p['scan_date']}</div></div>
      <div><div class="field-label">LMP</div>
           <div class="field-value">{p['lmp']}</div></div>
      <div><div class="field-label">Gestational Age</div>
           <div class="field-value">{f"{ga:.1f} weeks" if ga else "Not provided"}</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # AC metric
        if ac_mm:
            st.markdown(f"""
            <div class="metric-block">
              <div class="label">Abdominal Circumference</div>
              <div class="value">{ac_mm:.1f}<span class="unit">mm</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical badge
        if interp:
            badge_map = {
                'normal': 'badge-green', 'small': 'badge-red',
                'large': 'badge-amber',  'unknown': 'badge-gray',
            }
            badge_cls = badge_map.get(interp['status'], 'badge-gray')
            st.markdown(
                f'<span class="badge {badge_cls}">{interp["label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="interp-box">{interp["message"]}</div>',
                        unsafe_allow_html=True)

            ref = interp['ref']
            if ref['in_table']:
                # Percentile gauge
                render_percentile_gauge(ac_mm, ref['p5'], ref['p50'],
                                        ref['p95'], interp['status'])

                # Reference values
                st.markdown(f"""
                <div class="ref-values">
                  <strong style="color:#8fa4c8">Hadlock (1984) at {ga:.1f}w</strong><br>
                  5th percentile: <strong style="color:#f0f4ff">{ref['p5']} mm</strong>
                  &nbsp;·&nbsp;
                  50th percentile: <strong style="color:#f0f4ff">{ref['p50']} mm</strong>
                  &nbsp;·&nbsp;
                  95th percentile: <strong style="color:#f0f4ff">{ref['p95']} mm</strong>
                  &nbsp;·&nbsp;
                  Deviation: <strong style="color:#f0f4ff">{ac_mm - ref['p50']:+.1f} mm</strong>
                </div>
                """, unsafe_allow_html=True)

            # Clinical significance expandable
            render_clinical_expand(interp['status'])

        elif ac_mm and not ga:
            st.info("Gestational age not provided — enter GA to see clinical interpretation.",
                    icon="ℹ️")

        # Segmentation images
        st.markdown('<div class="um-card-title" style="margin-top:20px">🖼 Segmentation Summary</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.image(res['overlay'],     caption="Mask overlay", use_container_width=True)
        c2.image(res['ellipse_img'], caption="Fitted ellipse", use_container_width=True)

    with col_right:
        # Growth chart
        if ga and interp and interp['ref']['in_table']:
            chart_data = get_chart_data()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p95'],
                name='95th pct', line=dict(color='#2d7cf6', dash='dash', width=1),
            ))
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p50'],
                name='50th pct (median)', line=dict(color='#5b9cf8', width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p5'],
                name='5th pct', line=dict(color='#2d7cf6', dash='dash', width=1),
                fill='tonexty', fillcolor='rgba(45,124,246,0.08)',
            ))
            colour_map = {
                'normal': '#00c48c', 'small': '#f43f5e',
                'large': '#f59e0b',  'unknown': '#6b7280',
            }
            fig.add_trace(go.Scatter(
                x=[ga], y=[ac_mm], mode='markers',
                marker=dict(color=colour_map.get(interp['status'], '#2d7cf6'),
                            size=16, symbol='star',
                            line=dict(color='white', width=2)),
                name=f'Patient ({ac_mm:.1f} mm)',
            ))
            fig.update_layout(
                title=dict(text='Fetal AC Growth Chart — Hadlock (1984)',
                           font=dict(size=14, color='#8fa4c8')),
                xaxis=dict(title='Gestational Age (weeks)', range=[14, 42], dtick=2,
                           gridcolor='rgba(45,124,246,0.08)', color='#8fa4c8',
                           title_font=dict(color='#8fa4c8')),
                yaxis=dict(title='AC (mm)', gridcolor='rgba(45,124,246,0.08)',
                           color='#8fa4c8', title_font=dict(color='#8fa4c8')),
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font=dict(size=11, color='#8fa4c8'),
                            bgcolor='rgba(0,0,0,0)'),
                plot_bgcolor='#0a1628', paper_bgcolor='#0a1628',
                margin=dict(l=48, r=20, t=60, b=48), height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Pipeline timing
        if timing:
            st.markdown('<div class="um-card-title">⚡ Pipeline Performance</div>',
                        unsafe_allow_html=True)
            render_infer_stats(
                timing.get('n_slices', 1),
                timing.get('cls_total_ms', 0),
                seg_ms=timing.get('seg_ms'),
            )

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for research and educational
      purposes only. Results are generated by an automated deep learning system and must not be
      used as a sole basis for clinical decisions. All measurements should be verified by a
      qualified obstetrician or medical sonographer.
    </div>
    """, unsafe_allow_html=True)

    # ── PDF Report ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="um-card-title">📄 Download Clinical Report</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:13px;color:#8fa4c8;margin-bottom:16px">
      The PDF report includes patient details, ultrasound images, segmentation outputs,
      the full geometric measurement chain with equations, clinical interpretation,
      Hadlock reference values, biological significance, and pipeline performance metrics.
    </p>
    """, unsafe_allow_html=True)

    with st.spinner("Generating PDF report…"):
        try:
            pdf_bytes = generate_pdf_report(p, ga, res, interp, timing)
            st.download_button(
                label               = "⬇ Download Clinical Report (PDF)",
                data                = pdf_bytes,
                file_name           = f"UltraMeasure_{p['id']}_{p['scan_date']}.pdf",
                mime                = "application/pdf",
                use_container_width = True,
                type                = "primary",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}. Ensure ReportLab is installed: pip install reportlab")

    # ── Navigation ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Segmentation", use_container_width=True):
            st.session_state.step = 4; st.rerun()
    with col2:
        if st.button("🔄 Start New Assessment", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()