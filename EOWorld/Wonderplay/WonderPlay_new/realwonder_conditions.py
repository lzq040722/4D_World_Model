"""RealWonder condition generation from native WonderPlay simulation outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_REALWONDER_ROOT = Path(
    os.environ.get("REALWONDER_ROOT", "/root/autodl-tmp/RealWonder")
)
TARGET_HEIGHT = 512
TARGET_WIDTH = 512
LATENT_DOWNSAMPLE = 8
NOISE_CHANNELS = 32


def _get_config_value(config, section, key, default):
    if config is None:
        return default
    values = config.get(section, {})
    if values is None:
        return default
    return values.get(key, default)


def _load_pixel_flows(flow_dir: Path, flow_format: str = "auto"):
    flow_paths = sorted(flow_dir.glob("flow_*.npy"))
    if len(flow_paths) < 2:
        raise FileNotFoundError(f"At least two flow files are required in {flow_dir}")

    raw_flows = [np.load(path).astype(np.float32) for path in flow_paths]
    first_shape = raw_flows[0].shape
    if first_shape[0] != 2:
        raise ValueError(f"Expected flow shape [2,H,W], got {first_shape}")
    if any(flow.shape != first_shape for flow in raw_flows):
        raise ValueError("All flow arrays must have the same shape")

    detected_format = flow_format
    if flow_format == "auto":
        sample_min = min(float(flow.min()) for flow in raw_flows[:5])
        sample_max = max(float(flow.max()) for flow in raw_flows[:5])
        detected_format = (
            "normalized"
            if sample_min >= -0.05 and sample_max <= 1.05
            else "pixels"
        )

    if detected_format == "normalized":
        height, width = first_shape[1:]
        for flow in raw_flows:
            flow[0] = (flow[0] * 2.0 - 1.0) * width
            flow[1] = (flow[1] * 2.0 - 1.0) * height

    return raw_flows[1:], detected_format


def _import_realwonder_noise_warper(realwonder_root: Path):
    if not realwonder_root.joinpath("infer_sim.py").exists():
        raise FileNotFoundError(f"RealWonder was not found at {realwonder_root}")

    root = str(realwonder_root)
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        from simulation.image23D.noise_warp.make_warped_noise import NoiseWarper
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
    return NoiseWarper


def generate_realwonder_noise(
    traj_dir: Path,
    config=None,
    realwonder_root: Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Generate ``traj_00/noises.npy`` from ``traj_00/flows_actual``."""
    traj_dir = Path(traj_dir)
    noise_path = traj_dir / "noises.npy"
    if noise_path.exists() and not overwrite:
        return noise_path

    realwonder_root = Path(
        realwonder_root
        or _get_config_value(config, "realwonder", "root", DEFAULT_REALWONDER_ROOT)
    )
    flow_format = _get_config_value(config, "realwonder", "flow_format", "auto")
    output_height = int(
        _get_config_value(config, "realwonder", "output_height", TARGET_HEIGHT)
    )
    output_width = int(
        _get_config_value(config, "realwonder", "output_width", TARGET_WIDTH)
    )

    flows, detected_format = _load_pixel_flows(
        traj_dir / "flows_actual", flow_format=flow_format
    )
    print(
        f"[realwonder] Loaded {len(flows) + 1} flow files from "
        f"{traj_dir / 'flows_actual'}; using {len(flows)} transitions "
        f"as {detected_format} flow",
        flush=True,
    )

    NoiseWarper = _import_realwonder_noise_warper(realwonder_root.resolve())
    NoiseWarper().process(
        flows,
        str(traj_dir),
        input_flow=True,
        output_height=output_height,
        output_width=output_width,
        debug=False,
    )

    noise = np.load(noise_path, mmap_mode="r")
    expected_spatial_channels = (
        output_height // LATENT_DOWNSAMPLE,
        output_width // LATENT_DOWNSAMPLE,
        NOISE_CHANNELS,
    )
    if tuple(noise.shape[1:]) != expected_spatial_channels:
        raise RuntimeError(
            f"Generated noise has shape {noise.shape}; expected "
            f"[T,{expected_spatial_channels[0]},{expected_spatial_channels[1]},"
            f"{expected_spatial_channels[2]}]"
        )

    print(f"[realwonder] Saved structured noise to {noise_path}", flush=True)
    return noise_path


def prepare_realwonder_conditions(
    view_dir: Path,
    config=None,
    traj_id: int = 0,
    realwonder_root: Path | None = None,
    overwrite: bool = True,
) -> dict:
    """Make a WonderPlay view directory directly usable by RealWonder."""
    view_dir = Path(view_dir)
    traj_dir = view_dir / f"traj_{traj_id:02d}"
    first_frame_path = view_dir / "gt.png"
    frames_dir = traj_dir / "frames"
    prompt_path = view_dir / "text_prompt.txt"

    if not first_frame_path.exists():
        raise FileNotFoundError(f"Missing RealWonder first frame: {first_frame_path}")
    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.png")):
        raise FileNotFoundError(f"Missing RealWonder RGB frames in {frames_dir}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing RealWonder prompt: {prompt_path}")

    noise_path = generate_realwonder_noise(
        traj_dir,
        config=config,
        realwonder_root=realwonder_root,
        overwrite=overwrite,
    )
    return {
        "view_dir": view_dir,
        "traj_dir": traj_dir,
        "first_frame_path": first_frame_path,
        "frames_dir": frames_dir,
        "prompt_path": prompt_path,
        "noise_path": noise_path,
    }
