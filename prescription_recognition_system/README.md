# 💊 MediScan AI — AI-Assisted Prescription Recognition and Medicine Retrieval System

A premium, healthcare-themed Streamlit application that combines a classical **Digital Image
Processing (DIP)** pipeline with **Groq's Llama-4 Vision AI** to automatically extract medicines,
dosage strengths, and usage instructions from photographed prescriptions — and exports the
result as a polished PDF report.

Built as a Final Year Project (FYP) deliverable / professional portfolio showcase.

---

## ✨ Features

- **Premium healthcare UI** — gradient hero header, card-based layout, animated states, KPI tiles,
  color-coded confidence badges, sidebar navigation (Dashboard / Analytics / Settings / About).
- **Image upload module** — JPG/PNG/JPEG upload with live preview and metadata (dimensions, size,
  color mode, megapixels).
- **5-stage DIP pipeline** (OpenCV): Grayscale → Median Filter (noise reduction) → CLAHE contrast
  enhancement → Unsharp-mask sharpening → Adaptive Gaussian thresholding. Every stage is rendered
  as its own captioned card, with per-stage timing.
- **Groq Vision AI integration** — sends the chosen processed image to Groq's hosted Llama-4 model
  and parses a strict JSON response (medicine name, strength, frequency, per-item confidence,
  overall confidence, prescription quality rating). Runs on Groq's LPU inference, so responses
  come back very fast.
- **Medicine dashboard** — searchable, filterable results table with a confidence progress bar.
- **Analytics page** — per-medicine confidence bar chart, overall-confidence gauge, processing-time
  breakdown chart, and local (non-AI) sharpness/brightness/contrast heuristics, all via Plotly.
- **One-click PDF report** — branded report with both images, the medicines table, KPIs, and notes
  (ReportLab).
- **Robust error handling** — invalid images, missing/invalid API keys, quota limits, and JSON
  parse failures are all caught and shown as friendly, styled messages.

---

## 🗂️ Project Structure

```
prescription_recognition_system/
├── app.py                      # Main Streamlit app (UI + page routing)
├── modules/
│   ├── __init__.py
│   ├── image_processing.py     # DIP pipeline (OpenCV/NumPy/Pillow)
│   ├── groq_integration.py     # Groq Vision (Llama-4) API wrapper
│   ├── pdf_generator.py        # ReportLab PDF report builder
│   └── ui_components.py        # CSS theming + reusable UI cards/badges
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── .env.example                # Template for your Groq API key
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

**1. Clone / copy the project and create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your Groq API key**

Copy `.env.example` to `.env` and paste in your key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_actual_key_here
```

*(Alternatively, you can paste the key directly into the app's **Settings** page at runtime —
no `.env` file required for a quick demo.)*

**4. Run the app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🔑 Getting a Groq API Key

1. Go to **https://console.groq.com/keys**
2. Sign in / create a free account
3. Click **Create API Key** (free tier available, no credit card required)
4. Copy the key into `.env` or the in-app Settings page

---

## 🧠 Sample Groq Integration Code

This is a minimal standalone example of what `modules/groq_integration.py` does internally,
using the official **`groq`** SDK (OpenAI-compatible chat completions endpoint):

```python
import base64
from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

with open("prescription.jpg", "rb") as f:
    image_bytes = f.read()
b64_image = base64.b64encode(image_bytes).decode("utf-8")

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract medicine names, strengths, and frequencies "
                                          "from this prescription and return them as JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ],
        }
    ],
    temperature=0.2,
    response_format={"type": "json_object"},   # forces clean JSON output
)

print(response.choices[0].message.content)
```

> The full app uses a more detailed structured prompt (see `SYSTEM_PROMPT` in
> `modules/groq_integration.py`) that also requests per-item confidence scores and an
> overall prescription-quality rating.

---

## 🛠️ Technology Stack

| Layer              | Technology                              |
|---------------------|------------------------------------------|
| UI / Dashboard      | Streamlit + streamlit-option-menu        |
| Image Processing    | OpenCV, NumPy, Pillow                    |
| AI / Vision Model   | Groq — Llama-4 Scout/Maverick (`groq` SDK) |
| Data & Charts       | Pandas, Plotly                           |
| PDF Reports         | ReportLab                                |
| Config              | python-dotenv                            |

---

## ⚠️ Disclaimer

This application is built for **academic and portfolio demonstration purposes**. AI-extracted
medicine information must always be verified by a licensed pharmacist or physician before any
clinical or dispensing decision is made. It is not a certified medical device.

---

## 📌 Notes for Demo / Viva

- If `streamlit-option-menu` isn't installed, the sidebar automatically falls back to a plain
  `st.radio` menu — the app never crashes due to a missing optional UI dependency.
- All DIP functions are unit-testable independently of Streamlit (see `modules/image_processing.py`)
  — useful if your evaluators ask you to demonstrate individual processing stages in isolation.
- The Groq analyzer is fully decoupled from the UI (`modules/groq_integration.py`) so it can be
  reused in a CLI script, a FastAPI backend, or a batch-processing job without modification.
- Groq's free tier has request-per-minute and token-per-minute limits; if you hit a 429 error
  during a demo, wait a few seconds and retry, or switch to a paid tier for higher limits.
