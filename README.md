# SCENE — Intel Image Classifier Web App

A Flask-based web application that classifies natural scene images
using two trained CNN models: **PyTorch** (`.pth`) and **TensorFlow** (`.keras`).

---

## Project Structure

```
flask_app/
├── app.py                   ← Flask backend + inference logic
├── pytorch_model.py         ← (copy from training project)
├── requirements.txt
├── Dockerfile               ← For Fly.io deployment
├── fly.toml                 ← Fly.io config
├── pythonanywhere_wsgi.py   ← PythonAnywhere config
├── models/
│   ├── your_firstname_model.pth    ← trained PyTorch model
│   └── your_firstname_model.keras  ← trained TensorFlow model
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Local Setup & Run

```bash
# 1. Copy your trained models
mkdir models
cp path/to/your_firstname_model.pth   models/
cp path/to/your_firstname_model.keras models/

# 2. Copy the CNN architecture module from the training project
cp path/to/pytorch_model.py .

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the development server
python app.py
# → Open http://localhost:5000
```

### GPU Support (local)
The app automatically uses the best available device:
- **CUDA** (NVIDIA GPU) — detected via `torch.cuda.is_available()`
- **MPS** (Apple Silicon) — detected via `torch.backends.mps.is_available()`
- **CPU** — fallback

No code changes needed — the device is selected at model-load time.

---

## Web Interface Features

| Feature | Detail |
|---------|--------|
| Model selector | Radio buttons for PyTorch vs TensorFlow |
| Image upload | Drag-and-drop zone + file browser |
| Preview | Shows thumbnail + filename after upload |
| Classification | Animated confidence bar + probability chart |
| Responsive | Works on mobile and desktop |

---

## Deployment

### Option A — PythonAnywhere (free tier)

1. Create a free account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload the project to `~/scene_classifier/` via the Files tab
3. Upload your trained models to `~/scene_classifier/models/`
4. Open a Bash console:
   ```bash
   mkvirtualenv --python=python3.11 scene_venv
   pip install -r ~/scene_classifier/requirements.txt
   ```
5. In the **Web** tab:
   - Add new web app → Manual configuration → Python 3.11
   - Source code: `/home/<user>/scene_classifier`
   - Virtualenv: `/home/<user>/.virtualenvs/scene_venv`
   - Edit the WSGI file (replace with contents of `pythonanywhere_wsgi.py`,
     updating `<YOUR_USERNAME>`)
6. Click **Reload**.

> ⚠️ PythonAnywhere does **not** provide GPU access. Both models will run on CPU.

---

### Option B — Fly.io (supports GPU)

```bash
# 1. Install Fly CLI
brew install flyctl          # macOS
# or: curl -L https://fly.io/install.sh | sh

# 2. Authenticate
fly auth login

# 3. Initialise (first time only)
fly launch --name scene-intel-classifier

# 4. Deploy
fly deploy

# 5. Open in browser
fly open
```

#### Enabling GPU on Fly.io (optional)
Edit `fly.toml`:
```toml
[[vm]]
  memory   = "16gb"
  cpus     = 8
  gpu_kind = "a10"     # NVIDIA A10 GPU
```
Then redeploy: `fly deploy`

---

## API Reference

### `POST /predict`

**Form data:**
| Field | Type | Description |
|-------|------|-------------|
| `image` | file | JPG, PNG, WEBP, or BMP (max 16 MB) |
| `model` | string | `"pytorch"` or `"tensorflow"` |

**Response (200 OK):**
```json
{
  "predicted_class": "forest",
  "emoji": "🌲",
  "confidence": 97.43,
  "probabilities": {
    "buildings": 0.12,
    "forest": 97.43,
    "glacier": 0.08,
    "mountain": 1.93,
    "sea": 0.22,
    "street": 0.22
  },
  "model_used": "pytorch"
}
```

**Response (error):**
```json
{ "error": "No image file provided." }
```

### `GET /health`
Returns server status and list of currently-loaded models.

---

## Classes

| Label | Emoji | Description |
|-------|-------|-------------|
| buildings | 🏙️ | Architecture, city skylines |
| forest | 🌲 | Dense tree canopy |
| glacier | 🧊 | Ice fields, snow |
| mountain | ⛰️ | Peaks, ridgelines |
| sea | 🌊 | Ocean, coastal views |
| street | 🛣️ | Roads, urban scenes |
# flask-app
