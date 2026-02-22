import json
import secrets
import time
import urllib.request

import modal

app = modal.App.lookup("nne-jupyter", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        "jupyter~=1.1.0",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes>=0.46.1",
        "datasets",
        "pandas",
        "nltk",
        "rouge-score",
        "scikit-learn",
        "matplotlib",
    )
)

token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})

JUPYTER_PORT = 8888

print("Creating sandbox")

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "jupyter",
        "notebook",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.allow_remote_access=1",
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=3* 60 * 60,
        image=image,
        app=app,
        gpu="A100",
    )

print(f"Sandbox ID: {sandbox.object_id}")

tunnel = sandbox.tunnels()[JUPYTER_PORT]
url = f"{tunnel.url}/?token={token}"
print(f"Jupyter notebook is running at: {url}")


def is_jupyter_up():
    try:
        response = urllib.request.urlopen(f"{tunnel.url}/api/status?token={token}")
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
    except Exception:
        return False
    return False


timeout_sec = 60
start_time = time.time()
while time.time() - start_time < timeout_sec:
    if is_jupyter_up():
        print("Jupyter is up and running!")
        break
    time.sleep(1)
else:
    print("Timed out waiting for Jupyter to start.")
