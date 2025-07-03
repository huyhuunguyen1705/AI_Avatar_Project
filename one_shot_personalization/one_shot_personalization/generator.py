from __future__ import annotations

from typing import Tuple
from PIL import Image

from . import config, pipelines, face_utils


def _build_scale(ip_pipe, adapter_scale: float):
    """Return appropriate controlnet_conditioning_scale based on ControlNet(s)."""
    cn = ip_pipe.controlnet
    if isinstance(cn, list):
        # diffusers expects a list of floats equal to number of controlnets
        return [float(adapter_scale)] * len(cn)
    return float(adapter_scale)


def generate(
    face_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    adapter_scale: float,
    upscale_factor: float,
    cfg_scale: float,
) -> Tuple[Image.Image, Image.Image]:
    if face_img is None:
        raise ValueError("Please provide a face image")

    adapter_scale = float(adapter_scale)
    upscale_factor = float(upscale_factor)
    cfg_scale = float(cfg_scale)

    # --------------------------------------------------
    # 1) Face embedding + key‑points
    # --------------------------------------------------
    emb, kps = face_utils.analyse_face(face_img)
    kps_img = face_utils.overlay_kps(face_img, kps)

    # --------------------------------------------------
    # 2) InstantID (identity‑preserving base generation)
    # --------------------------------------------------
    ip_pipe = pipelines.instantid_pipe()
    ip_pipe.set_ip_adapter_scale(adapter_scale)

    base = ip_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=emb,
        image=kps_img,
        controlnet_conditioning_scale=_build_scale(ip_pipe, adapter_scale),
    ).images[0]

    # --------------------------------------------------
    # 3) HiresFix (upscale + 3‑step denoise)
    # --------------------------------------------------
    w, h = base.size
    upscaled = base.resize((int(w * upscale_factor), int(h * upscale_factor)), Image.LANCZOS)

    final = pipelines.hires_pipe()(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=upscaled,
        num_inference_steps=config.HIRES_STEPS,
        strength=config.HIRES_DENOISE,
        guidance_scale=cfg_scale,
    ).images[0]

    return base, final