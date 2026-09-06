"""Update a simulated dynamic 3D Gaussian scene from generated video frames.

This module intentionally implements only the update described in WonderPlay:
generated video frames supervise differentiable renders with a photometric L1
loss, optimizing foreground motion/appearance and background color.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from gaussian_renderer.living_world_render import render_interaction_mlp


def _cfg(config, key, default):
    values = (config or {}).get("video_scene_refinement", {}) or {}
    return values.get(key, default)


def _load_rgb(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def refine_dynamic_scene_from_video(
    *,
    target_frames_dir,
    simulation_states,
    motion_model,
    gaussians,
    viewpoint_camera,
    opt,
    background,
    config,
    output_dir,
    fps=8,
    status_callback=None,
):
    """Minimize ``|V - render(S)|_1`` over the dynamic 3D scene sequence."""
    target_paths = sorted(Path(target_frames_dir).glob("frame_*.png"))
    if not target_paths:
        raise FileNotFoundError(f"No generated target frames found in {target_frames_dir}")
    if not simulation_states:
        raise ValueError("simulation_states is empty")
    if len(target_paths) != len(simulation_states):
        raise ValueError(
            "Video supervision requires one generated frame per simulated 3D state, "
            f"but received {len(target_paths)} video frames and "
            f"{len(simulation_states)} simulation states."
        )

    output_dir = Path(output_dir)
    frames_dir = output_dir / "optimized_3d_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    device = gaussians.get_xyz_all.device
    dtype = gaussians.get_xyz_all.dtype

    object_xyz_static, _ = gaussians._tmp_get_xyz_all_separate()
    object_count = int(object_xyz_static.shape[0])
    total_count = int(gaussians.get_xyz_all.shape[0])
    if object_count == 0 or object_count >= total_count:
        raise ValueError(
            f"Expected object and background Gaussians, got object={object_count}, "
            f"total={total_count}"
        )

    base_features = (
        gaussians.get_features_all.detach()
        .transpose(1, 2)
        .reshape(total_count, 3)
        .clone()
    )
    object_base_features = base_features[:object_count]
    background_base_features = base_features[object_count:]

    iterations = int(_cfg(config, "iterations_per_frame", 30))
    if iterations < 1:
        raise ValueError("video_scene_refinement.iterations_per_frame must be at least 1")
    position_lr = float(_cfg(config, "position_lr", 1e-3))
    object_feature_lr = float(_cfg(config, "object_feature_lr", 1e-2))
    background_feature_lr = float(_cfg(config, "background_feature_lr", 1e-2))
    log_interval = max(1, int(_cfg(config, "log_interval", 10)))
    scale_factor = float(
        (config or {}).get("interaction", {}).get(
            "environment_scale_factor",
            (config or {}).get("environment_motion", {}).get("scale_factor", 1.0),
        )
    )

    optimized_records = []
    playback_states = []
    rendered_frames = []

    if status_callback is not None:
        status_callback(f"3D scene refinement: 0/{len(target_paths)} frames")

    # The video generator runs under inference mode. This stage explicitly
    # enables gradients because the Gaussian renderer must backpropagate.
    with torch.enable_grad():
        for frame_index, target_path in enumerate(target_paths):
            state = simulation_states[frame_index]
            if "obj_0000" not in state:
                raise KeyError("Video scene refinement currently expects state['obj_0000']")
            simulated_xyz = state["obj_0000"]["xyz"].detach().to(
                device=device, dtype=dtype
            )
            if simulated_xyz.shape != (object_count, 3):
                raise ValueError(
                    f"Object state shape {tuple(simulated_xyz.shape)} does not match "
                    f"Gaussian object shape {(object_count, 3)}"
                )

            target = _load_rgb(target_path, width, height).to(
                device=device, dtype=dtype
            )

            # Paper variables: foreground trajectory, foreground appearance,
            # and per-Gaussian background color for this frame.
            object_xyz = nn.Parameter(simulated_xyz.clone())
            object_features = nn.Parameter(object_base_features.clone())
            background_features = nn.Parameter(background_base_features.clone())
            optimizer = torch.optim.Adam(
                [
                    {"params": [object_xyz], "lr": position_lr},
                    {"params": [object_features], "lr": object_feature_lr},
                    {"params": [background_features], "lr": background_feature_lr},
                ],
                eps=1e-8,
            )

            for iteration in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                feature_logits = torch.cat(
                    [object_features, background_features], dim=0
                )
                override_color = gaussians.color_activation(feature_logits)
                render_pkg = render_interaction_mlp(
                    viewpoint_camera=viewpoint_camera,
                    pc=gaussians,
                    motion_model=motion_model,
                    obj_xyz_t=object_xyz,
                    t=frame_index,
                    opt=opt,
                    bg_color=background,
                    override_color=override_color,
                    render_visible=False,
                    scale_factor=scale_factor,
                )
                rendered = render_pkg["render"].clamp(0.0, 1.0)
                loss = (rendered - target).abs().mean()
                loss.backward()
                optimizer.step()

                if iteration % log_interval == 0 or iteration == iterations - 1:
                    print(
                        f"[video-scene-refinement] frame "
                        f"{frame_index + 1}/{len(target_paths)} iter "
                        f"{iteration + 1}/{iterations} L1={loss.item():.6f}",
                        flush=True,
                    )

            optimized_object_xyz = object_xyz.detach()
            optimized_object_features = object_features.detach()
            optimized_background_features = background_features.detach()

            with torch.no_grad():
                final_colors = gaussians.color_activation(
                    torch.cat(
                        [optimized_object_features, optimized_background_features],
                        dim=0,
                    )
                )
                final_pkg = render_interaction_mlp(
                    viewpoint_camera=viewpoint_camera,
                    pc=gaussians,
                    motion_model=motion_model,
                    obj_xyz_t=optimized_object_xyz,
                    t=frame_index,
                    opt=opt,
                    bg_color=background,
                    override_color=final_colors,
                    render_visible=False,
                    scale_factor=scale_factor,
                )
                final_image = final_pkg["render"].detach().cpu().clamp(0, 1)
                final_array = (
                    final_image.permute(1, 2, 0).numpy() * 255.0
                ).round().astype(np.uint8)
                Image.fromarray(final_array).save(
                    frames_dir / f"frame_{frame_index:08d}.png"
                )
                rendered_frames.append(final_array)

            optimized_records.append(
                {
                    "frame_index": frame_index,
                    "object_xyz": optimized_object_xyz.cpu(),
                    "object_features": optimized_object_features.cpu(),
                    "background_features": optimized_background_features.cpu(),
                }
            )
            playback_states.append(
                {
                    "obj_0000": {"xyz": optimized_object_xyz},
                    "_video_refinement": {
                        "source_state_index": frame_index,
                        "fps": float(fps),
                        "object_features": optimized_object_features,
                        "background_features": optimized_background_features,
                    },
                }
            )
            if status_callback is not None:
                status_callback(
                    f"3D scene refinement: {frame_index + 1}/{len(target_paths)} frames"
                )

    checkpoint_path = output_dir / "optimized_dynamic_scene.pt"
    torch.save(
        {
            "version": 2,
            "object_count": object_count,
            "total_count": total_count,
            "frames": optimized_records,
        },
        checkpoint_path,
    )
    output_video_path = output_dir / "optimized_3d_scene.mp4"
    imageio.mimwrite(output_video_path, rendered_frames, fps=float(fps))
    print(
        f"[video-scene-refinement] Optimized 3D scene saved to {checkpoint_path}; "
        f"rendered playback saved to {output_video_path}",
        flush=True,
    )
    return {
        "refined_frames_dir": frames_dir,
        "output_video_path": output_video_path,
        "scene_checkpoint_path": checkpoint_path,
        "fps": float(fps),
        "frame_count": len(rendered_frames),
        "playback_states": playback_states,
    }
