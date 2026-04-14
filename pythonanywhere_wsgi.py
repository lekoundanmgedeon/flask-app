"""
pythonanywhere_wsgi.py
──────────────────────
WSGI configuration for PythonAnywhere deployment.

SETUP STEPS ON PYTHONANYWHERE
──────────────────────────────
1. Upload your project to ~/scene_classifier/
2. In the Web tab → Add new web app → Manual configuration → Python 3.11
3. Set "Source code" to:       /home/<username>/scene_classifier
4. Set "Working directory" to: /home/<username>/scene_classifier
5. Set "WSGI configuration file" to point to this file (or paste it in)
6. In a Bash console, create a virtualenv and install deps:
       mkvirtualenv --python=python3.11 scene_venv
       pip install -r requirements.txt
7. In the Web tab → Virtualenv: /home/<username>/.virtualenvs/scene_venv
8. Place your trained models in:
       ~/scene_classifier/models/your_firstname_model.pth
       ~/scene_classifier/models/your_firstname_model.keras
9. Hit the green "Reload" button.

IMPORTANT — GPU NOTE
─────────────────────
PythonAnywhere free/paid tiers do not provide GPU access. The app will
run on CPU automatically (torch / TF both handle this gracefully).
For GPU inference, use Fly.io with a GPU machine (see fly.toml).
"""

import sys
import os

# Path to your project
project_home = '/home/<YOUR_USERNAME>/scene_classifier'  # ← change this

if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)
os.environ['FLASK_ENV'] = 'production'

from app import app as application   # noqa: E402  (Gunicorn/uWSGI entry point)
