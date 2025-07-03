from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List

from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# InsightFace model selection
#   * antelopev2 : light but **recognition‑only** (no detection)
#   * buffalo_l  : full (detection + recognition) – default fallback
# ---------------------------------------------------------------------------
_MODEL_CANDIDATES: List[str] = ["antelopev2", "buffalo_l", "buffalo_m"]
_APP: FaceAnalysis | None = None


def _init_app() -> FaceAnalysis:
    """Try candidates until one provides a detection model."""
    for name in _MODEL_CANDIDATES:
        try:
            app = FaceAnalysis(
                name=name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            if "detection" in app.models:
                print(f"✓ InsightFace model set '{name}' loaded (with detection).")
                return app
            print(f"✗ Model set '{name}' lacks detection – skipping.")
        except Exception as exc:  # pragma: no cover
            print(f"✗ Failed to init '{name}': {exc}")
    raise RuntimeError("No InsightFace model with detection was found."
                       " Please ensure at least 'buffalo_l' is downloadable.")


def _get_app() -> FaceAnalysis:
    global _APP
    if _APP is None:
        _APP = _init_app()
    return _APP


def analyse_face(img_pil: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embedding, keypoints) or raise ValueError if no face."""
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    faces = _get_app().get(bgr)
    if not faces:
        raise ValueError("No face detected")
    # choose largest face
    face = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
    return face["embedding"], face["kps"]


def overlay_kps(img_pil: Image.Image, kps: np.ndarray, color=(0, 255, 0)) -> Image.Image:
    """Draw facial key‑points (for ControlNet conditioning)."""
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for x, y in kps:
        cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))