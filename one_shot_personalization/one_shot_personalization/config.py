from pathlib import Path
import subprocess, sys, importlib.util
import torch

# ---------------------------------------------------------------------------
# Paths / IDs
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "SG161222/RealVisXL_V5.0"          # photorealistic SDXL base
INSTANTID_REPO = "InstantX/InstantID"             # hosts ControlNet + IP‑Adapter
PIPELINE_FILE = "pipeline_stable_diffusion_xl_instantid.py"  # community pipeline
_FACE_ADAPTER_FILE = "ip-adapter.bin"

# ---------------------------------------------------------------------------
# Runtime / compute
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# HiresFix defaults recommended for RealVisXL V5.0
# ---------------------------------------------------------------------------
HIRES_STEPS = 3
HIRES_DENOISE = 0.5
CFG_MIN, CFG_MAX = 1.0, 2.0

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
HF_CACHE_DIR = Path.home() / ".cache/huggingface/hub"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_pipeline_file() -> Path:
    """Ensure the InstantID XL community pipeline file exists locally."""
    path = Path(__file__).resolve().parent.parent / PIPELINE_FILE
    if path.exists():
        return path
    # fetch from GitHub (same path as model card)
    url = (
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/" + PIPELINE_FILE
    )
    print(f"▸ Downloading community pipeline to {path} …")
    import urllib.request, ssl

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: S501
    with urllib.request.urlopen(url) as resp:
        path.write_bytes(resp.read())
    return path


def face_adapter_path() -> Path:
    """Return local path to IP‑Adapter weights, downloading once if missing."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=INSTANTID_REPO,
            filename=_FACE_ADAPTER_FILE,
            cache_dir=HF_CACHE_DIR,
        )
    )