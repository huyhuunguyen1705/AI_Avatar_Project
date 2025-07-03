from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

from . import config

# ---------------------------------------------------------------------------
# Dynamically load the community pipeline file if necessary
# ---------------------------------------------------------------------------
PIPE_PATH: Path = config.ensure_pipeline_file()

spec = importlib.util.spec_from_file_location("pipeline_stable_diffusion_xl_instantid", PIPE_PATH)
if spec is None or spec.loader is None:  # pragma: no cover
    raise ImportError("Cannot load InstantID community pipeline file")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # type: ignore[arg-type]
spec.loader.exec_module(module)  # type: ignore[union-attr]

# Now we can import from it
StableDiffusionXLInstantIDPipeline = module.StableDiffusionXLInstantIDPipeline  # type: ignore[attr-defined]

# ------------------------------
# ControlNet (IdentityNet)
# ------------------------------
_controlnet: ControlNetModel | None = None


def _get_controlnet() -> ControlNetModel:
    global _controlnet
    if _controlnet is None:
        _controlnet = ControlNetModel.from_pretrained(
            config.INSTANTID_REPO,
            subfolder="ControlNetModel",
            torch_dtype=config.DTYPE,
        )
    return _controlnet

# ------------------------------
# InstantID pipeline
# ------------------------------
_pipe_id: StableDiffusionXLInstantIDPipeline | None = None


def instantid_pipe() -> StableDiffusionXLInstantIDPipeline:
    global _pipe_id
    if _pipe_id is None:
        _pipe_id = StableDiffusionXLInstantIDPipeline.from_pretrained(
            config.BASE_MODEL_ID,
            controlnet=_get_controlnet(),
            torch_dtype=config.DTYPE,
            variant="fp16" if config.DTYPE == torch.float16 else None,
        )
        _pipe_id.to(config.DEVICE)
        _pipe_id.load_ip_adapter_instantid(config.face_adapter_path())
        _pipe_id.enable_model_cpu_offload()
    return _pipe_id

# ------------------------------
# Img2Img pipeline for HiresFix
# ------------------------------
_pipe_hires: StableDiffusionXLImg2ImgPipeline | None = None


def hires_pipe() -> StableDiffusionXLImg2ImgPipeline:
    global _pipe_hires
    if _pipe_hires is None:
        _pipe_hires = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config.BASE_MODEL_ID,
            torch_dtype=config.DTYPE,
            variant="fp16" if config.DTYPE == torch.float16 else None,
        )
        _pipe_hires.scheduler = DPMSolverMultistepScheduler.from_config(_pipe_hires.scheduler.config)
        _pipe_hires.to(config.DEVICE)
        _pipe_hires.enable_model_cpu_offload()
    return _pipe_hires