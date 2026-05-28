"""
UltraMeasure — Fetal Growth Assessment
Streamlit wizard application.

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
import zipfile
import tempfile
import datetime
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

from pipeline        import InferencePipeline
from growth_standards import interpret_ac, get_chart_data, lmp_to_ga_weeks

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "UltraMeasure — Fetal Growth Assesst, designed by 8th Sem, NIET",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Typography & palette ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
  }

  /* FULL PAGE BACKGROUND */
  .stApp {
    background-color: #eaf4ff !important;
  }
  /* FORCE GLOBAL TEXT COLOR */
  body, p, span, div, label, h1, h2, h3, h4, h5, h6,
  .stMarkdown, .stText, .stApp {
    color: #000000 !important;
  }

  /* Brand colours */
  :root {
    --blue-dark : #1a3a5c;
    --blue-mid  : #1e5fa8;
    --blue-light: #e8f1fb;
    --teal      : #0d9488;
    --green     : #059669;
    --amber     : #d97706;
    --red       : #dc2626;
    --gray-soft : #f1f5f9;
    --gray-border:#e2e8f0;
    --text-main : #0f172a;
    --text-soft : #475569;
  }

  h1 {
    font-size: 34px !important;
    font-weight: 700 !important;
  }

  h2 {
    font-size: 26px !important;
    font-weight: 600 !important;
  }

  h3 {
    font-size: 22px !important;
  }

  /* ── Header bar ── */
  .um-header {
    background: white;
    color: #1a3a5c;
    padding: 10px 0px;
    border-radius: 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .um-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0;
    color: #000000;
  }

  .um-header p {
    font-size: 13px;
    margin: 2px 0 0;
    color: #000000;
    opacity: 0.8;
  }

  /* ── Step indicator ── */
  .step-bar {
    display: flex;
    align-items: center;
    margin-bottom: 32px;
    gap: 0;
  }
  .step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
  }
  .step-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    flex-shrink: 0;
  }
  .step-circle.done   { background: #059669; color: white; }
  .step-circle.active { background: #1e5fa8; color: white;
                        box-shadow: 0 0 0 3px #bfdbfe; }
  .step-circle.todo   { background: #e2e8f0; color: #94a3b8; }
  .step-label { font-size: 12px; font-weight: 600; color: #475569; }
  .step-label.active  { color: #1e5fa8; }
  .step-connector {
    flex: 1; height: 2px;
    background: #e2e8f0; margin: 0 4px;
  }
  .step-connector.done { background: #059669; }

  /* ── Cards (SOFT SAAS STYLE) ── */
  .um-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
  }
  .um-card-title {
    font-size: 15px; font-weight: 700;
    color: var(--blue-dark);
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px;
  }

  /* ── Result badge ── */
  .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px; font-weight: 600;
  }
  .badge-green  { background: #d1fae5; color: #065f46; }
  .badge-amber  { background: #fef3c7; color: #92400e; }
  .badge-red    { background: #fee2e2; color: #991b1b; }
  .badge-gray   { background: #f1f5f9; color: #475569; }

  /* ── Metric block ── */
  .metric-block {
    background: #f8fafc;
    border-left: 4px solid #4169E1;
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin-bottom: 12px;
  }
  .metric-block .label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--blue-mid); margin-bottom: 2px;
  }
  .metric-block .value {
    font-size: 28px; font-weight: 700; color: var(--blue-dark);
  }
  .metric-block .unit {
    font-size: 14px; color: var(--text-soft); margin-left: 4px;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px; color: #78350f;
    margin-top: 16px;
  }

  /* ── Stacked image grid ── */
  .slice-grid img { border-radius: 6px; }

  /* ── Button overrides ── */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    background-color: #ef4444 !important;
    color: black !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(239, 68, 68, 0.20) !important;
  }

  .stButton > button:hover {
    background-color: #dc2626 !important;
    color: black !important;
    transform: translateY(-1px);
  }

  /* FORCE ALL STREAMLIT BUTTON VARIANTS RED */
  button[kind="primary"],
  button[kind="secondary"],
  button[data-testid="baseButton-primary"],
  button[data-testid="baseButton-secondary"] {
      background-color: #ef4444 !important;
      color: black !important;
      border: none !important;
  }

  button[kind="primary"]:hover,
  button[kind="secondary"]:hover,
  button[data-testid="baseButton-primary"]:hover,
  button[data-testid="baseButton-secondary"]:hover {
      background-color: #dc2626 !important;
      color: black !important;
  }

  /* INPUT FIELDS WHITE */
  input, textarea, [data-baseweb="input"] input, [data-baseweb="select"] {
      background-color: white !important;
      color: black !important;
  }

  div[data-testid="stTextInput"] input,
  div[data-testid="stNumberInput"] input,
  div[data-testid="stDateInput"] input,
  div[data-testid="stSelectbox"] div,
  textarea {
      background-color: white !important;
      color: black !important;
      border: 1px solid #e2e8f0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_pipeline():
    base = os.path.dirname(__file__)
    cls_path = os.path.join(base, "models", "best_model1.pth")
    seg_path = os.path.join(base, "models", "best_seg_model.pth")

    if not os.path.exists(cls_path):
        st.error(f"Classifier weights not found at: {cls_path}")
        st.stop()
    if not os.path.exists(seg_path):
        st.error(f"Segmentation weights not found at: {seg_path}")
        st.stop()

    return InferencePipeline(cls_path, seg_path)


# ── Session state initialisation ──────────────────────────────────────────

def init_state():
    defaults = {
        'step'         : 1,
        'patient'      : {},
        'images'       : [],
        'image_names'  : [],
        'results'      : None,
        'interpretation': None,
        'ga_weeks'     : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helper: step progress bar ─────────────────────────────────────────────

def render_steps(current: int):
    steps  = ["Patient Info", "Upload Scans",
               "Classification", "Segmentation", "Results"]
    html   = '<div class="step-bar">'
    for i, label in enumerate(steps, 1):
        if i < current:
            circle_cls = "done";   label_cls = ""
            icon = "✓"
        elif i == current:
            circle_cls = "active"; label_cls = "active"
            icon = str(i)
        else:
            circle_cls = "todo";   label_cls = ""
            icon = str(i)

        html += f'''
          <div class="step-item">
            <div class="step-circle {circle_cls}">{icon}</div>
            <span class="step-label {label_cls}">{label}</span>
          </div>
        '''
        if i < len(steps):
            conn_cls = "done" if i < current else ""
            html += f'<div class="step-connector {conn_cls}"></div>'

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Helper: image loader from various formats ─────────────────────────────

def load_images_from_upload(uploaded_files) -> tuple:
    """
    Accept uploaded files (PNG/JPG, ZIP, or MHA) and return
    (list of PIL images, list of names).
    """
    images = []; names = []

    for uf in uploaded_files:
        name = uf.name.lower()

        # ── ZIP ──────────────────────────────────────────────────────
        if name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(uf.read())) as z:
                img_entries = sorted([
                    n for n in z.namelist()
                    if n.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and not n.startswith('__MACOSX')
                ])
                for entry in img_entries:
                    try:
                        img = Image.open(io.BytesIO(z.read(entry))).convert("L")
                        images.append(img)
                        names.append(os.path.basename(entry))
                    except Exception:
                        pass

        # ── MHA / MHD (3-D stack) ────────────────────────────────────
        elif name.endswith(('.mha', '.mhd')):
            try:
                import SimpleITK as sitk
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(name)[1], delete=False
                ) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                volume = sitk.GetArrayFromImage(sitk.ReadImage(tmp_path))
                os.unlink(tmp_path)

                # volume shape: (slices, H, W)
                for i, slc in enumerate(volume):
                    slc_norm = ((slc - slc.min()) /
                                (slc.max() - slc.min() + 1e-8) * 255
                               ).astype(np.uint8)
                    images.append(Image.fromarray(slc_norm).convert("L"))
                    names.append(f"slice_{i+1:04d}.png")

            except ImportError:
                st.warning(
                    "SimpleITK is required for MHA/MHD files. "
                    "Install with: `pip install SimpleITK`"
                )

        # ── Single image ─────────────────────────────────────────────
        elif name.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(io.BytesIO(uf.read())).convert("L")
                images.append(img)
                names.append(uf.name)
            except Exception:
                pass

    return images, names


# ── Header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="um-header">
  <div>
    <h1>🔬 UltraMeasure</h1>
    <p>Automated Fetal Abdominal Circumference Assessment · Hadlock (1984) Standards</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — PATIENT INFORMATION
# ══════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    render_steps(1)

    st.markdown("""
    <div class="um-card">
      <div class="um-card-title">👤 Patient Information</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            patient_id  = st.text_input("Patient ID *",
                                        placeholder="e.g. PT-2024-001")
            patient_name = st.text_input("Patient Name *",
                                         placeholder="Full name")

        with col2:
            lmp_date = st.date_input(
                "Last Menstrual Period (LMP)",
                value      = None,
                min_value  = datetime.date.today() - datetime.timedelta(days=300),
                max_value  = datetime.date.today(),
                help       = "Used to calculate gestational age automatically",
            )
            ga_manual = st.number_input(
                "Gestational Age (weeks) — if LMP unknown",
                min_value = 0.0, max_value = 45.0,
                value     = 0.0, step      = 0.5,
                help      = "Leave at 0 if LMP is provided above",
            )

        scan_date = st.date_input(
            "Scan Date",
            value     = datetime.date.today(),
            max_value = datetime.date.today(),
        )

        submitted = st.form_submit_button(
            "Continue to Upload →",
            use_container_width = True,
        )

        if submitted:
            if not patient_id.strip() or not patient_name.strip():
                st.error("Patient ID and Patient Name are required.")
            else:
                # Determine gestational age
                if lmp_date is not None:
                    ga = lmp_to_ga_weeks(lmp_date, scan_date)
                elif ga_manual > 0:
                    ga = float(ga_manual)
                else:
                    ga = None

                st.session_state.patient = {
                    'id'        : patient_id.strip(),
                    'name'      : patient_name.strip(),
                    'lmp'       : str(lmp_date) if lmp_date else "Not provided",
                    'ga_weeks'  : ga,
                    'scan_date' : str(scan_date),
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
      <p style="font-size:13px;color:#475569;margin:0">
        Patient: <strong>{p['name']}</strong> &nbsp;·&nbsp;
        ID: <strong>{p['id']}</strong> &nbsp;·&nbsp;
        GA: <strong>{f"{p['ga_weeks']:.1f} weeks" if p['ga_weeks'] else "Not provided"}</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**Supported formats:** PNG / JPG (individual slices), "
        "ZIP (folder of images), MHA / MHD (3-D volume). "
        "You can upload multiple files at once.",
        icon="ℹ️"
    )

    uploaded = st.file_uploader(
        "Choose files",
        type            = ['png', 'jpg', 'jpeg', 'zip', 'mha', 'mhd'],
        accept_multiple_files = True,
        label_visibility = "collapsed",
    )

    if uploaded:
        with st.spinner("Reading images…"):
            images, names = load_images_from_upload(uploaded)

        if not images:
            st.error("No valid images could be read from the uploaded files.")
        else:
            st.success(f"✓ {len(images)} slice{'s' if len(images)>1 else ''} loaded.")

            # Show thumbnail grid
            st.markdown(
                "<div class='um-card-title' style='margin-top:16px'>"
                "🖼 Uploaded Slices Preview</div>",
                unsafe_allow_html=True
            )
            cols_per_row = 6
            rows = [images[i:i+cols_per_row]
                    for i in range(0, min(len(images), 30), cols_per_row)]

            for row_imgs in rows:
                cols = st.columns(len(row_imgs))
                for col, img, nm in zip(cols, row_imgs,
                                        names[:cols_per_row]):
                    col.image(img, caption=nm, use_container_width=True)

            if len(images) > 30:
                st.caption(f"Showing first 30 of {len(images)} slices.")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
            with col2:
                if st.button("Run Classification →",
                             use_container_width=True,
                             type="primary"):
                    st.session_state.images      = images
                    st.session_state.image_names = names
                    st.session_state.step        = 3
                    st.rerun()
    else:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    render_steps(3)

    images = st.session_state.images
    names  = st.session_state.image_names

    st.markdown("""
    <div class="um-card">
      <div class="um-card-title">🧠 Step 1 — Classification</div>
      <p style="font-size:13px;color:#475569;margin:0">
        The ResNet-34 classifier scores every slice for abdomen presence.
        The slice with the highest confidence is selected for segmentation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = load_pipeline()

    progress_bar = st.progress(0, text="Classifying slices…")
    status_text  = st.empty()

    # Run classification with live progress
    classified = []
    with st.spinner(""):
        import torch
        from pipeline import preprocess
        for i, img in enumerate(images):
            t    = preprocess(img).unsqueeze(0).to(pipeline.device)
            with torch.no_grad():
                prob = torch.sigmoid(pipeline.classifier(t)).item()
            classified.append((img, prob, names[i] if i < len(names) else f"slice_{i+1}"))
            progress_bar.progress(
                (i + 1) / len(images),
                text=f"Classifying slice {i+1} / {len(images)}…"
            )

    progress_bar.empty()
    classified_sorted = sorted(classified, key=lambda x: x[1], reverse=True)
    best_img, best_prob, best_name = classified_sorted[0]

    st.success(
        f"✓ Classification complete. "
        f"Best slice: **{best_name}** "
        f"(confidence: **{best_prob:.1%}**)"
    )

    # Show all slices with scores
    st.markdown(
        "<div class='um-card-title' style='margin-top:16px'>"
        "📊 Classification Scores — All Slices</div>",
        unsafe_allow_html=True
    )

    cols_per_row = 5
    for row_start in range(0, min(len(classified), 25), cols_per_row):
        row_items = classified[row_start:row_start + cols_per_row]
        cols      = st.columns(len(row_items))
        for col, (img, prob, nm) in zip(cols, row_items):
            is_best = (img is best_img)
            border  = "3px solid #1e5fa8" if is_best else "1px solid #e2e8f0"
            col.image(img, use_container_width=True)
            colour = "#059669" if prob >= 0.5 else "#94a3b8"
            col.markdown(
                f"<p style='font-size:11px;text-align:center;"
                f"color:{colour};margin:0'>"
                f"{'⭐ ' if is_best else ''}{prob:.1%}</p>",
                unsafe_allow_html=True
            )

    if len(classified) > 25:
        st.caption(f"Showing first 25 of {len(classified)} slices.")

    # Highlighted best slice
    st.markdown("---")
    st.markdown(
        "<div class='um-card-title'>⭐ Selected Best Slice</div>",
        unsafe_allow_html=True
    )
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
            st.warning(
                "⚠️ No slice with sufficient confidence was found. "
                "The best available slice is shown but results may be unreliable."
            )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Run Segmentation →",
                     use_container_width=True, type="primary"):
            st.session_state.results = {
                'best_img'  : best_img,
                'best_prob' : best_prob,
                'best_name' : best_name,
                'classified': classified_sorted,
            }
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
      <div class="um-card-title">✂️ Step 2 — Segmentation</div>
      <p style="font-size:13px;color:#475569;margin:0">
        The U-Net model segments the fetal abdominal boundary at pixel level.
        An ellipse is fitted to the predicted contour and Ramanujan's
        approximation computes the abdominal circumference.
      </p>
    </div>
    """, unsafe_allow_html=True)

    pipeline = load_pipeline()

    with st.spinner("Running segmentation…"):
        prob_map, binary_mask = pipeline.segment(best_img)
        overlay               = pipeline.make_overlay(best_img, binary_mask)

    from pipeline import estimate_ac
    ac_mm, ellipse, _ = estimate_ac(binary_mask)

    if ellipse is not None:
        ellipse_img = pipeline.draw_ellipse(best_img, ellipse)
    else:
        ellipse_img = overlay
        st.warning("⚠️ Ellipse fitting failed — mask contour may be too small.")

    # Store for results step
    st.session_state.results.update({
        'prob_map'   : prob_map,
        'binary_mask': binary_mask,
        'overlay'    : overlay,
        'ellipse_img': ellipse_img,
        'ellipse'    : ellipse,
        'ac_mm'      : ac_mm,
    })

    # Display segmentation outputs
    st.markdown(
        "<div class='um-card-title' style='margin-top:8px'>"
        "🔍 Segmentation Output</div>",
        unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.image(best_img,    caption="Original slice",       use_container_width=True)
    col2.image(overlay,     caption="Predicted mask overlay", use_container_width=True)

    # Probability heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    col3.pyplot(fig, use_container_width=True)
    col3.caption("Probability heatmap")
    plt.close(fig)

    col4.image(ellipse_img, caption="Fitted ellipse",       use_container_width=True)

    # AC result
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

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("View Clinical Results →",
                     use_container_width=True, type="primary"):
            # Compute clinical interpretation
            ga      = st.session_state.ga_weeks
            interp  = interpret_ac(ac_mm, ga) if ac_mm and ga else None
            st.session_state.interpretation = interp
            st.session_state.step = 5
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — RESULTS + REPORT
# ══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 5:
    render_steps(5)

    p        = st.session_state.patient
    res      = st.session_state.results
    interp   = st.session_state.interpretation
    ac_mm    = res.get('ac_mm')
    ga       = st.session_state.ga_weeks

    st.markdown("""
    <div class="um-card">
      <div class="um-card-title">📋 Clinical Results</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Patient summary bar ────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:10px;padding:16px;margin-bottom:20px;
                display:flex;gap:32px;flex-wrap:wrap">
      <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:1px">Patient</span><br>
           <strong>{p['name']}</strong></div>
      <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:1px">ID</span><br>
           <strong>{p['id']}</strong></div>
      <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:1px">Scan Date</span><br>
           <strong>{p['scan_date']}</strong></div>
      <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:1px">LMP</span><br>
           <strong>{p['lmp']}</strong></div>
      <div><span style="font-size:11px;color:#64748b;text-transform:uppercase;
           letter-spacing:1px">Gestational Age</span><br>
           <strong>{f"{ga:.1f} weeks" if ga else "Not provided"}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # ── AC measurement + interpretation ───────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # AC metric
        if ac_mm:
            st.markdown(f"""
            <div class="metric-block">
              <div class="label">Abdominal Circumference</div>
              <div class="value">{ac_mm:.1f}
                <span class="unit">mm</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical badge + message
        if interp:
            badge_map = {
                'normal': 'badge-green',
                'small' : 'badge-red',
                'large' : 'badge-amber',
                'unknown': 'badge-gray',
            }
            badge_cls = badge_map.get(interp['status'], 'badge-gray')
            st.markdown(
                f'<span class="badge {badge_cls}">{interp["label"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:8px;
                        padding:14px;margin-top:12px;font-size:13px;
                        color:#334155;line-height:1.7">
              {interp['message']}
            </div>
            """, unsafe_allow_html=True)

            # Reference values
            ref = interp['ref']
            if ref['in_table']:
                st.markdown(f"""
                <div style="margin-top:12px;font-size:12px;color:#475569">
                  <strong>Hadlock (1984) Reference at {ga:.1f}w:</strong><br>
                  5th percentile: {ref['p5']} mm &nbsp;·&nbsp;
                  50th percentile: {ref['p50']} mm &nbsp;·&nbsp;
                  95th percentile: {ref['p95']} mm
                </div>
                """, unsafe_allow_html=True)
        elif ac_mm and not ga:
            st.info(
                "Gestational age was not provided. "
                "Enter GA to see clinical interpretation.",
                icon="ℹ️"
            )

        # Segmentation images
        st.markdown(
            "<div class='um-card-title' style='margin-top:20px'>"
            "🖼 Segmentation Images</div>",
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        c1.image(res['overlay'],     caption="Mask overlay", use_container_width=True)
        c2.image(res['ellipse_img'], caption="Ellipse fit",  use_container_width=True)

    with col_right:
        # Growth chart
        if ga and interp and interp['ref']['in_table']:
            chart_data = get_chart_data()
            fig = go.Figure()

            # Percentile bands
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p95'],
                name='95th', line=dict(color='#93c5fd', dash='dash', width=1),
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p50'],
                name='50th (median)', line=dict(color='#1e5fa8', width=2),
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=chart_data['weeks'], y=chart_data['p5'],
                name='5th', line=dict(color='#93c5fd', dash='dash', width=1),
                fill='tonexty', fillcolor='rgba(147,197,253,0.15)',
                showlegend=True,
            ))

            # Patient measurement point
            colour_map = {
                'normal': '#059669', 'small': '#dc2626',
                'large': '#d97706',  'unknown': '#6b7280'
            }
            point_colour = colour_map.get(interp['status'], '#1e5fa8')
            fig.add_trace(go.Scatter(
                x=[ga], y=[ac_mm],
                mode='markers',
                marker=dict(color=point_colour, size=14, symbol='star',
                            line=dict(color='white', width=2)),
                name=f'Patient ({ac_mm:.1f} mm)',
                showlegend=True,
            ))

            fig.update_layout(
                title=dict(
                    text='Fetal AC Growth Chart — Hadlock (1984)',
                    font=dict(size=13, color='#1a3a5c')
                ),
                xaxis=dict(
                    title='Gestational Age (weeks)',
                    range=[14, 42], dtick=2,
                    gridcolor='#f1f5f9',
                ),
                yaxis=dict(
                    title='AC (mm)',
                    gridcolor='#f1f5f9',
                ),
                legend=dict(
                    orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1,
                    font=dict(size=11),
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=20, t=60, b=40),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Disclaimer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for
      research and educational purposes only. Results are generated by an
      automated deep learning system and must not be used as a sole basis
      for clinical decisions. All measurements should be verified by a
      qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)

    # ── Download report ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='um-card-title'>📄 Download Report</div>",
        unsafe_allow_html=True
    )

    def generate_report_text() -> str:
        lines = [
            "=" * 60,
            "       ULTRAMEASURE — FETAL GROWTH ASSESSMENT REPORT",
            "=" * 60,
            f"Patient Name    : {p['name']}",
            f"Patient ID      : {p['id']}",
            f"Scan Date       : {p['scan_date']}",
            f"LMP             : {p['lmp']}",
            f"Gestational Age : {f'{ga:.1f} weeks' if ga else 'Not provided'}",
            "",
            "-" * 60,
            "MEASUREMENT RESULTS",
            "-" * 60,
            f"Abdominal Circumference : {f'{ac_mm:.1f} mm' if ac_mm else 'Not available'}",
            f"Best Slice              : {res['best_name']}",
            f"Classification Conf.    : {res['best_prob']:.1%}",
            "",
        ]
        if interp and interp['ref']['in_table']:
            ref = interp['ref']
            lines += [
                "-" * 60,
                "CLINICAL INTERPRETATION (Hadlock 1984)",
                "-" * 60,
                f"Status          : {interp['label']}",
                f"5th Percentile  : {ref['p5']} mm",
                f"50th Percentile : {ref['p50']} mm",
                f"95th Percentile : {ref['p95']} mm",
                "",
                interp['message'],
                "",
            ]
        lines += [
            "-" * 60,
            "DISCLAIMER",
            "-" * 60,
            "This report is generated by an automated deep learning system",
            "for research and educational purposes only. It must not be",
            "used as a sole basis for clinical decisions. All measurements",
            "should be verified by a qualified healthcare professional.",
            "=" * 60,
        ]
        return "\n".join(lines)

    report_text = generate_report_text()
    st.download_button(
        label             = "⬇ Download Report (.txt)",
        data              = report_text,
        file_name         = f"UltraMeasure_{p['id']}_{p['scan_date']}.txt",
        mime              = "text/plain",
        use_container_width = True,
    )

    # ── Navigation ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Segmentation", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 Start New Assessment",
                     use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
