# run_app.py — Streamlit App Launcher
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = str(ROOT / "app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"]
    sys.exit(stcli.main())
