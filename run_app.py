# run_app.py — Streamlit App Launcher
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Check multiple candidates for venv site-packages
candidate_site_packages = [
    ROOT / "venv" / "Lib" / "site-packages",
    Path(r"D:\plant-veg-health\venv\Lib\site-packages"),
    Path(r"C:\Users\kurtz\Downloads\MLMProjectFeed\venv\Lib\site-packages"),
]

for p in candidate_site_packages:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if Path(r"D:\plant-veg-health\src").exists() and str(Path(r"D:\plant-veg-health\src")) not in sys.path:
    sys.path.insert(0, str(Path(r"D:\plant-veg-health\src")))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = str(ROOT / "app.py")
    if not (ROOT / "app.py").exists() and Path(r"D:\plant-veg-health\app.py").exists():
        app_path = str(Path(r"D:\plant-veg-health\app.py"))
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"]
    sys.exit(stcli.main())

