"""Laboratorio 8: check non distruttivi su GPU, vLLM e API HTTP locale."""
import os
import platform
import shutil
import subprocess
import sys
import requests

URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1").rstrip("/")
print("Architettura:", platform.machine())
print("Python:", sys.executable, sys.version.split()[0])
print("Eseguibile vllm:", shutil.which("vllm") or "assente in questo ambiente client")

if shutil.which("nvidia-smi"):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10
        )
        print("GPU:\n" + result.stdout.strip())
    except (subprocess.SubprocessError, OSError) as exc:
        print("GPU info non disponibile:", type(exc).__name__)
else:
    print("nvidia-smi non trovato")

try:
    resp = requests.get(URL + "/models", timeout=8)
    print("Endpoint /models HTTP:", resp.status_code)
    if resp.ok:
        print("Modelli:", [m.get("id") for m in resp.json().get("data", [])])
    else:
        print("Diagnostica HTTP:", resp.text[:200])
except requests.RequestException as exc:
    print("Server irraggiungibile:", type(exc).__name__, str(exc)[:200])

print("Per ulteriori controlli nell'ambiente vLLM: vllm --version; vllm serve --help")
