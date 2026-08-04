
# UltraMeasure — Fetal Growth Assessment
Automated fetal abdominal circumference measurement from ultrasound images.
<img width="1383" height="741" alt="Screenshot 2026-08-04 at 20 33 10" src="https://github.com/user-attachments/assets/680a1a81-b1d5-4e3c-b328-c13c9ff7c081" />
---

## Folder structure

```
ultrameasure/
    app.py                  ← Streamlit application (main entry point)
    pipeline.py             ← ML inference pipeline
    growth_standards.py     ← Hadlock growth chart + clinical interpretation
    requirements.txt        ← Python dependencies
    models/
        best_model1.pth     ← Stage 1 ResNet-34 classifier weights
        best_seg_model.pth  ← Stage 2 U-Net segmentation weights
```

---

## Setup — local

### Step 1 — Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add model weights
Copy your trained model files into the `models/` folder:
```
ultrameasure/models/best_model1.pth
ultrameasure/models/best_seg_model.pth
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Setup — Streamlit Cloud deployment

1. Push the entire `ultrameasure/` folder to a GitHub repository.
2. Add your model `.pth` files to `models/` and push them
   (files under 100 MB each are accepted by GitHub directly;
   larger files require Git LFS).
3. Go to https://share.streamlit.io → New app → select your repo.
4. Set the main file path to `app.py`.
5. Deploy.

> **Note on model file size:** If your `.pth` files exceed GitHub's
> 100 MB limit, use Git LFS or host them on Google Drive and load
> via a URL in `pipeline.py` using `torch.hub.download_url_to_file`.

---

## Running on Google Colab (ngrok tunnel)

```python
# In a Colab cell:
!pip install streamlit pyngrok -q
!ngrok authtoken YOUR_NGROK_TOKEN

from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen(['streamlit', 'run', 'app.py',
                         '--server.port', '8501'])
time.sleep(3)
tunnel = ngrok.connect(8501)
print("App URL:", tunnel.public_url)
```

---

## App workflow

```
Step 1 — Patient Info     Enter patient ID, name, LMP / gestational age
Step 2 — Upload Scans     Upload ZIP, PNG/JPG images, or MHA/MHD volume
Step 3 — Classification   All slices classified; best slice selected
Step 4 — Segmentation     U-Net segments abdomen; ellipse fitted; AC computed
Step 5 — Results          AC value, Hadlock growth chart, clinical interpretation,
                          downloadable report
```

---

## Supported input formats

| Format | Description |
|--------|-------------|
| `.png` / `.jpg` | Individual ultrasound slices |
| `.zip` | ZIP archive containing PNG/JPG images |
| `.mha` / `.mhd` | 3-D ultrasound volume (requires SimpleITK) |

---

## Clinical note

This tool is for **research and educational purposes only**.
Results must be verified by a qualified healthcare professional.
Growth interpretation is based on Hadlock (1984) reference standards.
