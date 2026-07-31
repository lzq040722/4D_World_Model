import copy
import io
import base64
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from kornia.morphology import dilation
from typing import Iterable, Union
from scipy.interpolate import griddata as interp_grid
from scipy.ndimage import minimum_filter, maximum_filter
from torch.optim.lr_scheduler import ExponentialLR

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage
from PIL import Image
from einops import rearrange
from kornia.geometry import PinholeCamera
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds
from torchvision.transforms import ToTensor, ToPILImage
from util.midas_utils import dpt_transform, dpt_512_transform
from util.utils import functbl, save_depth_map, SimpleLogger, soft_stitching

from util.segment_utils import refine_disp_with_segments_2, save_sam_anns
from typing import List, Optional, Tuple, Union
from kornia.morphology import erosion
from syncdiffusion.syncdiffusion_model import SyncDiffusion
import os
from utils.loss import l1_loss
import matplotlib.pyplot as plt
from scipy.ndimage import label

from thirdparty.cinemagraphy.demo import eulerian_estimation
from helpmotion_in_model import SceneFlow, flow2img, save_image
import os, re, glob
from utils.trajectory import get_pcdGenPoses
def check(x, name):
    import torch
    if x is None:
        print(name, "None"); return
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    print(name, "nan:", torch.isnan(t).any().item(), "inf:", torch.isinf(t).any().item(),
          "shape:", tuple(t.shape))

BG_COLOR=(1, 0, 0)


def make_final_hints_xy(hints, H, W, yflip=False, as_column_list=True, dtype=np.float32):
    hints = np.asarray(hints)
    assert hints.ndim == 2 and hints.shape[0] == 4, f"expected (4,N), got {hints.shape}"
    N = hints.shape[1]
    if N == 0:
        return [], [], [], []


    sx = hints[0].astype(dtype, copy=False)
    sy = hints[1].astype(dtype, copy=False)
    ex = hints[2].astype(dtype, copy=False)
    ey = hints[3].astype(dtype, copy=False)


    if yflip:
        sy = (H - 1) - sy
        ey = (H - 1) - ey


    sx = np.clip(sx, 0, W - 1)
    ex = np.clip(ex, 0, W - 1)
    sy = np.clip(sy, 0, H - 1)
    ey = np.clip(ey, 0, H - 1)

    if as_column_list:

        to_col_list = lambda v: [np.asarray([v[i]], dtype=dtype) for i in range(N)]
        return to_col_list(sx), to_col_list(sy), to_col_list(ex), to_col_list(ey)
    else:

        return sx.astype(float).tolist(), sy.astype(float).tolist(), ex.astype(float).tolist(), ey.astype(float).tolist()


def has_motion_hints(hints):
    """Return whether the UI supplied at least one start/end motion pair."""
    if hints is None:
        return False
    hints = np.asarray(hints)
    return hints.ndim == 2 and hints.shape[0] == 4 and hints.shape[1] > 0

def save_image_incremental(img, base_dir, prefix="background"):
    out_dir = Path(base_dir) / "background"
    out_dir.mkdir(parents=True, exist_ok=True)


    existing = sorted(out_dir.glob(f"{prefix}_*.png"))
    next_idx = 0
    if existing:
        try:
            nums = [int(p.stem.split("_")[-1]) for p in existing
                    if p.stem.split("_")[-1].isdigit()]
            if nums:
                next_idx = max(nums) + 1
        except Exception:
            pass


    def to_uint8_rgb(x):

        if isinstance(x, Image.Image):
            return np.array(x.convert("RGB"), dtype=np.uint8)


        if torch.is_tensor(x):
            t = x.detach().cpu()

            if t.ndim == 4:
                t = t[0]

            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1,2,0)
            t = t.numpy()

        else:
            t = np.asarray(x)


        if t.ndim == 2:
            t = np.stack([t, t, t], axis=-1)


        if t.ndim == 3 and t.shape[-1] == 4:
            t = t[..., :3]

        assert t.ndim == 3 and t.shape[-1] in (1,3), f"Unexpected shape: {t.shape}"


        if t.shape[-1] == 1:
            t = np.repeat(t, 3, axis=-1)


        if np.issubdtype(t.dtype, np.floating):
            m = np.nanmax(np.abs(t))

            if m <= 1.0 + 1e-6:
                t = np.clip(t, 0.0, 1.0) * 255.0
            else:
                t = np.clip(t, 0.0, 255.0)
        else:
            t = np.clip(t, 0, 255)

        return t.astype(np.uint8)

    rgb = to_uint8_rgb(img)

    out_path = out_dir / f"{prefix}_{next_idx:04d}.png"
    Image.fromarray(rgb).save(out_path)
    return str(out_path)

def save_image_incremental_fore(img, base_dir, prefix="foreground"):
    out_dir = Path(base_dir) / "foreground"
    out_dir.mkdir(parents=True, exist_ok=True)


    existing = sorted(out_dir.glob(f"{prefix}_*.png"))
    next_idx = 0
    if existing:
        try:
            nums = [int(p.stem.split("_")[-1]) for p in existing
                    if p.stem.split("_")[-1].isdigit()]
            if nums:
                next_idx = max(nums) + 1
        except Exception:
            pass


    def to_uint8_rgb(x):

        if isinstance(x, Image.Image):
            return np.array(x.convert("RGB"), dtype=np.uint8)


        if torch.is_tensor(x):
            t = x.detach().cpu()

            if t.ndim == 4:
                t = t[0]

            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1,2,0)
            t = t.numpy()

        else:
            t = np.asarray(x)


        if t.ndim == 2:
            t = np.stack([t, t, t], axis=-1)


        if t.ndim == 3 and t.shape[-1] == 4:
            t = t[..., :3]

        assert t.ndim == 3 and t.shape[-1] in (1,3), f"Unexpected shape: {t.shape}"


        if t.shape[-1] == 1:
            t = np.repeat(t, 3, axis=-1)


        if np.issubdtype(t.dtype, np.floating):
            m = np.nanmax(np.abs(t))

            if m <= 1.0 + 1e-6:
                t = np.clip(t, 0.0, 1.0) * 255.0
            else:
                t = np.clip(t, 0.0, 255.0)
        else:
            t = np.clip(t, 0, 255)

        return t.astype(np.uint8)

    rgb = to_uint8_rgb(img)

    out_path = out_dir / f"{prefix}_{next_idx:04d}.png"
    Image.fromarray(rgb).save(out_path)
    return str(out_path)

def save_mask_incremental(mask, base_dir, prefix="mask"):
    out_dir = Path(base_dir) / "mask"
    out_dir.mkdir(parents=True, exist_ok=True)


    nums = [int(p.stem.split("_")[-1]) for p in out_dir.glob(f"{prefix}_*.png")
            if p.stem.split("_")[-1].isdigit()]
    idx = (max(nums) + 1) if nums else 0


    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().squeeze()
        if m.ndim == 3 and m.shape[0] in (1, 3):
            m = m[0]
        m = (m > 0).to(torch.uint8).numpy() * 255
    else:
        m = np.asarray(mask).squeeze()
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8) * 255

    path = out_dir / f"{prefix}_{idx:04d}.png"
    Image.fromarray(m).save(path)
    return str(path)

def camera_params(h_in, w_in, focal_length=None):
    H = int(h_in.item() if hasattr(h_in, "item") else h_in)
    W = int(w_in.item() if hasattr(w_in, "item") else w_in)
    print("camera_params H,W:",H,W)


    print("focal_length in camera params:", focal_length)
    aspect_ratio = W / H
    focal = (focal_length * aspect_ratio, focal_length)


    fov = (2*np.arctan(W / (2*focal[0])), 2*np.arctan(H / (2*focal[1])))

    K = np.array([
            [focal[0], 0., W/2],
            [0., focal[1], H/2],
            [0.,            0.,       1.],
        ]).astype(np.float32)


    print("W,H:", W,H, "fx,fy,cx,cy:", K[0,0],K[1,1],K[0,2],K[1,2])
    assert abs(K[0,2]-W/2)<1e-3 and abs(K[1,2]-H/2)<1e-3


    return K, fov

def render_PCD(src_img, src_mask, hints, src_depth, K, fov, render_poses, internel_render_poses):

    H=512
    W=512


    if torch.is_tensor(src_depth):
        src_depth = src_depth.detach().float().cpu().numpy()


    if src_depth.ndim == 3:
        src_depth = src_depth[0]
    elif src_depth.ndim == 4:
        src_depth = src_depth[0,0]


    src_mask = Image.fromarray(np.repeat(np.array(src_mask)[..., np.newaxis], 3, axis=-1).astype(np.uint8))
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    edgeN = 2
    edgemask = np.ones((H-2*edgeN, W-2*edgeN))
    edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))


    R0, T0 = render_poses[0,:3,:3], render_poses[0,:3,3:4]
    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*src_depth, y*src_depth, 1*src_depth), axis=0).reshape(3,-1))
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
    new_pts_colors2 = (np.array(src_img).reshape(-1,3).astype(np.float32)/255.)
    mask_pts_colors2 = (np.array(src_mask).reshape(-1,3).astype(np.float32)/255.)

    pts_coord_world, pts_colors, mask_pts_colors= new_pts_coord_world2.copy(), new_pts_colors2.copy(), mask_pts_colors2.copy()


    hint_start_world_coord = []
    for h in range(len(hints[0])):
        h_x = hints[0][h]
        h_y = hints[1][h]
        depth_x_y = src_depth[h_y, h_x]
        pixel_coords = np.array([[h_y], [h_x], [1]]) * depth_x_y
        cam_coords = np.linalg.inv(K).dot(pixel_coords)
        hint_pts_coord_world2 = (np.linalg.inv(R0).dot(cam_coords) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
        hint_start_world_coord.append(hint_pts_coord_world2)

    hint_end_world_coord = []
    for l in range(len(hints[0])):
        h_x = hints[2][l]
        h_y = hints[3][l]
        depth_x_y = src_depth[h_y, h_x]
        pixel_coords = np.array([[h_y], [h_x], [1]]) * depth_x_y
        cam_coords = np.linalg.inv(K).dot(pixel_coords)
        hint_pts_coord_world2 = (np.linalg.inv(R0).dot(cam_coords) - np.linalg.inv(R0).dot(T0)).astype(np.float32)
        hint_end_world_coord.append(hint_pts_coord_world2)


    yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
    traindata = {
        'camera_angle_x': fov[0],
        'camera_angle_y': fov[1],
        'W': W,
        'H': H,
        'pcd_points': pts_coord_world,
        'pcd_colors': pts_colors,
        'pcd_masks': mask_pts_colors,
        'frames': [],
    }

    iterable_align = range(len(render_poses))

    none_idx = []
    for i in iterable_align:
        for j in range(len(internel_render_poses)):
            idx = i * len(internel_render_poses) + j
            print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')


            Rw2i = render_poses[i,:3,:3]
            Tw2i = render_poses[i,:3,3:4]
            Ri2j = internel_render_poses[j,:3,:3]
            Ti2j = internel_render_poses[j,:3,3:4]

            Rw2j = np.matmul(Ri2j, Rw2i)
            Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j


            Rj2w = np.matmul(yz_reverse, Rw2j).T
            Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
            Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

            pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
            pixel_coord_camj = np.matmul(K, pts_coord_camj)

            valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0,
                                                        pixel_coord_camj[0]/pixel_coord_camj[2]>=0,
                                                        pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1,
                                                        pixel_coord_camj[1]/pixel_coord_camj[2]>=0,
                                                        pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
            if len(valid_idxj) == 0:
                none_idx.append(idx)
                continue
            pts_depthsj = pixel_coord_camj[-1:, valid_idxj]
            pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
            round_coord_camj = np.round(pixel_coord_camj).astype(np.int32)

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)

            imagej = interp_grid(pixel_coord_camj.transpose(1,0), pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
            imagej = edgemask[...,None]*imagej + (1-edgemask[...,None])*np.pad(imagej[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            depthj = interp_grid(pixel_coord_camj.transpose(1,0), pts_depthsj.T, grid, method='linear', fill_value=0).reshape(H,W)
            depthj = edgemask*depthj + (1-edgemask)*np.pad(depthj[1:-1,1:-1], ((1,1),(1,1)), mode='edge')

            maskj = np.zeros((H,W), dtype=np.float32)
            maskj[round_coord_camj[1], round_coord_camj[0]] = 1
            maskj = maximum_filter(maskj, size=(9,9))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*(-1)

            maskj = minimum_filter((imagej.sum(-1)!=-3)*1, size=(11,11))
            imagej = maskj[...,None]*imagej + (1-maskj[...,None])*0

            imagej_2 = interp_grid(pixel_coord_camj.transpose(1,0), mask_pts_colors[valid_idxj], grid, method='linear', fill_value=0).reshape(H,W,3)
            imagej_2 = edgemask[...,None]*imagej_2 + (1-edgemask[...,None])*np.pad(imagej_2[1:-1,1:-1], ((1,1),(1,1),(0,0)), mode='edge')

            maskj_2 = np.zeros((H,W), dtype=np.float32)
            maskj_2[round_coord_camj[1], round_coord_camj[0]] = 1
            maskj_2 = maximum_filter(maskj_2, size=(9,9))
            imagej_2 = maskj_2[...,None]*imagej_2 + (1-maskj[...,None])*(-1)

            maskj_2 = minimum_filter((imagej_2.sum(-1)!=-3)*1, size=(11,11))
            imagej_2 = maskj_2[...,None]*imagej_2 + (1-maskj_2[...,None])*0
            maskp = imagej_2
            mask = maskp[:,:,0]

            final_hint_start_x = []
            final_hint_start_y = []

            final_hint_end_x = []
            final_hint_end_y = []

            for hint_coord in hint_start_world_coord:
                pts_coord_camj = Rw2j.dot(hint_coord) + Tw2j
                pixel_coord_camj = np.matmul(K, pts_coord_camj)
                pixel_coord_camj /= pixel_coord_camj[2]

                final_hint_start_y.append(pixel_coord_camj[0])
                final_hint_start_x.append(pixel_coord_camj[1])

            for hint_coord in hint_end_world_coord:
                pts_coord_camj = Rw2j.dot(hint_coord) + Tw2j
                pixel_coord_camj = np.matmul(K, pts_coord_camj)
                pixel_coord_camj /= pixel_coord_camj[2]

                final_hint_end_y.append(pixel_coord_camj[0])
                final_hint_end_x.append(pixel_coord_camj[1])


            dbg_dir = "/home/mhj/mhj/WonderWorld_ours/input/mhj/cloud8_fix/MOM/render_debug"
            os.makedirs(dbg_dir, exist_ok=True)

            debug_img = np.round(imagej * 255.).astype(np.uint8)
            Image.fromarray(debug_img).save(
                os.path.join(dbg_dir, f"render_{idx:04d}.png")
            )


            depth_norm = depthj.copy()
            if depth_norm.max() > 0:
                depth_norm = depth_norm / depth_norm.max()
            depth_img = (depth_norm * 255).astype(np.uint8)

            Image.fromarray(depth_img).save(
                os.path.join(dbg_dir, f"depth_{idx:04d}.png")
            )

            traindata['frames'].append({
                'image': Image.fromarray(np.round(imagej*255.).astype(np.uint8)),
                'transform_matrix': Pc2w.tolist(),
                'mask': Image.fromarray(np.round(mask*255.).astype(np.uint8)),
                'final_hint_start_x' : final_hint_start_x,
                'final_hint_start_y' : final_hint_start_y,
                'final_hint_end_x' : final_hint_end_x,
                'final_hint_end_y' : final_hint_end_y,
                'T2C_flow' : [],
                'our_flow' : [],
            })

    return traindata, none_idx

def estimate_flow(train_data, viz_dir, args):
        frames = train_data["frames"]
        for idx, frame in enumerate(frames):
            flow = eulerian_estimation(args, frame)
            viz_flow2(flow, viz_dir)
            train_data['frames'][idx]['T2C_flow'].append(flow)

        return train_data

def viz_flow2(flow, out_dir, prefix="flow_viz", ext="png"):
    os.makedirs(out_dir, exist_ok=True)
    idx = _next_index(out_dir, prefix)
    fname = f"{prefix}_{idx:06d}.{ext}"
    save_path = os.path.join(out_dir, fname)

    img = flow2img(flow[0])
    Image.fromarray(img).save(save_path)
    return save_path

def optimize_motion(train_data, non_frame_idx, train_iteration, K, render_poses, internel_render_poses):
    print("frames_len =", len(train_data["frames"]))
    print("pose_pairs =", len(render_poses) * len(internel_render_poses))
    print("non_frame_idx_len =", len(non_frame_idx), "min/max =", (min(non_frame_idx), max(non_frame_idx)) if len(non_frame_idx)>0 else None)

    H=512
    W=512
    coord = train_data['pcd_points']

    pts_coord_world = coord.copy()


    yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])


    GT_list = []
    frame_idx = 0
    iterable_align = range(len(render_poses))
    for i in iterable_align:
        for j in range(len(internel_render_poses)):
            idx = i * len(internel_render_poses) + j
            print(f'{idx+1} / {len(render_poses)*len(internel_render_poses)}')
            if idx in non_frame_idx:
                continue
            frame = train_data["frames"][frame_idx]
            GT_flow = frame["T2C_flow"][0]
            frame_idx +=1

            x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
            grid = np.stack((x,y), axis=-1).reshape(-1,2)


            Rw2i = render_poses[i,:3,:3]
            Tw2i = render_poses[i,:3,3:4]
            Ri2j = internel_render_poses[j,:3,:3]
            Ti2j = internel_render_poses[j,:3,3:4]

            Rw2j = np.matmul(Ri2j, Rw2i)
            Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j


            Rj2w = np.matmul(yz_reverse, Rw2j).T
            Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
            Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

            pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
            pixel_coord_camj = np.matmul(K, pts_coord_camj)

            valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0,
                                                                        pixel_coord_camj[0]/pixel_coord_camj[2]>=0,
                                                                        pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1,
                                                                        pixel_coord_camj[1]/pixel_coord_camj[2]>=0,
                                                                        pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
            if len(valid_idxj) == 0:
                continue
            pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]

            GT_flow = GT_flow.permute(2, 3, 1, 0).squeeze().reshape(H*W,2)
            GT_flow_numpy = GT_flow.cpu().clone().numpy()

            GT_flow_pixel_coord_camj = interp_grid(grid, GT_flow_numpy, pixel_coord_camj.transpose(1,0), method='linear', fill_value=0)
            GT_flow_pixel_coord_camj_tensor = torch.tensor(GT_flow_pixel_coord_camj.T)
            GT_list.append(GT_flow_pixel_coord_camj_tensor)


    model = SceneFlow(coord)
    tensor_pts_coord_world = torch.from_numpy(pts_coord_world.copy())
    device = next(model.parameters()).device

    tensor_pts_coord_world = tensor_pts_coord_world.to(device)

    flow_optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    scene_flow = model().to(device)
    scheduler = ExponentialLR(flow_optimizer, gamma=0.97)


    for epoch in tqdm(range(train_iteration)):
        avg_loss = 0
        GT_num = 0
        for i in iterable_align:
            for j in range(len(internel_render_poses)):
                idx = i * len(internel_render_poses) + j
                if idx not in non_frame_idx:
                    GT_flow_pixel_coord_camj_tensor = GT_list[GT_num].to(device)
                    GT_num += 1
                else:
                    continue


                Rw2i = render_poses[i,:3,:3]
                Tw2i = render_poses[i,:3,3:4]
                Ri2j = internel_render_poses[j,:3,:3]
                Ti2j = internel_render_poses[j,:3,3:4]

                Rw2j = np.matmul(Ri2j, Rw2i)
                Tw2j = np.matmul(Ri2j, Tw2i) + Ti2j


                Rj2w = np.matmul(yz_reverse, Rw2j).T
                Tj2w = -np.matmul(Rj2w, np.matmul(yz_reverse, Tw2j))
                Pc2w = np.concatenate((Rj2w, Tj2w), axis=1)
                Pc2w = np.concatenate((Pc2w, np.array([[0,0,0,1]])), axis=0)

                pts_coord_camj = Rw2j.dot(pts_coord_world) + Tw2j
                pixel_coord_camj = np.matmul(K, pts_coord_camj)

                Rw2j_tensor = torch.tensor(Rw2j,dtype=torch.float32).to(device)
                Tw2j_tensor = torch.tensor(Tw2j,dtype=torch.float32).to(device)
                K_tensor = torch.tensor(K,dtype=torch.float32).to(device)

                scene_flow = model()
                flow_pts_coord_world = tensor_pts_coord_world+scene_flow


                flow_pts_coord_camj = torch.matmul(Rw2j_tensor,flow_pts_coord_world) + Tw2j_tensor
                flow_pixel_coord_camj = torch.matmul(K_tensor, flow_pts_coord_camj)
                valid_idxj = np.where(np.logical_and.reduce((pixel_coord_camj[2]>0,
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]>=0,
                                                            pixel_coord_camj[0]/pixel_coord_camj[2]<=W-1,
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]>=0,
                                                            pixel_coord_camj[1]/pixel_coord_camj[2]<=H-1)))[0]
                if len(valid_idxj) == 0:
                    continue
                pixel_coord_camj = pixel_coord_camj[:2, valid_idxj]/pixel_coord_camj[-1:, valid_idxj]
                flow_pixel_coord_camj = flow_pixel_coord_camj[:2, valid_idxj]/flow_pixel_coord_camj[-1:, valid_idxj]

                pixel_coord_camj_tensor = torch.tensor(pixel_coord_camj,dtype=torch.float32).to(device)

                new_flow_coord_camj = flow_pixel_coord_camj - pixel_coord_camj_tensor


                Ll1 = l1_loss(new_flow_coord_camj, GT_flow_pixel_coord_camj_tensor)
                avg_loss +=  Ll1
                loss = avg_loss/(idx+1)
                if GT_num == len(GT_list):
                    flow_optimizer.zero_grad()


                    loss.backward()
                    flow_optimizer.step()


                if epoch == (train_iteration-1):
                    new_flow_coord_camj_2 = new_flow_coord_camj.detach().cpu().numpy()
                    final_flow = interp_grid(pixel_coord_camj.transpose(1,0), new_flow_coord_camj_2.T, grid, method='linear', fill_value=0).reshape(H,W,2)
                    flow_world_tensor = torch.tensor(np.transpose(final_flow,(2,0,1))).unsqueeze(0)
                    train_data['frames'][idx]['our_flow'].append(flow_world_tensor)


        scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_lr(), 'loss:', loss)

    scene_flow = model()
    return train_data, scene_flow

class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, return_fragment_idx=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )


        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())
        if len(ret) == 1:
            ret = images
        return ret


class SoftmaxImportanceCompositor(torch.nn.Module):

    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None, softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        background_color = kwargs.get("background_color", self.background_color)

        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = - 1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(ptclds.shape[0], fragments.shape[0], fragments.shape[1], fragments.shape[2], fragments.shape[3])
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)


        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


class FrameSyn(torch.nn.Module):
    def __init__(self, config, inpainter_pipeline, depth_model, normal_estimator=None):
        super().__init__()


        self.inpainting_prompt = None
        self.adaptive_negative_prompt = None
        self.current_pc = None
        self.current_pc_sky = None
        self.current_pc_layer = None
        self.current_pc_latest = None
        self.current_pc_layer_latest = None
        self.current_visible_pc = None
        self.current_visible_pc_init = None
        self.inpainting_resolution = None
        self.border_mask = None
        self.border_size = None
        self.border_image = None
        self.run_dir = None
        self.hints_prev = None


        self.image_latest = torch.zeros(1, 3, 512, 512)

        self.flow_latest = torch.zeros(1, 2, 512, 512)
        self.extrinsics_latest = torch.zeros(1, 4, 4)
        self.intrinsics_latest = torch.zeros(1, 4, 4)

        self.sky_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_latest = torch.zeros(1, 1, 512, 512)
        self.inpaint_input_image_latest = ToPILImage()(torch.zeros(3, 512, 512))
        self.depth_latest = torch.zeros(1, 1, 512, 512)
        self.disparity_latest = torch.zeros(1, 1, 512, 512)
        self.post_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_disocclusion = torch.zeros(1, 1, 512, 512)

        self.kf_idx = 0
        self.images = []
        self.images_layer = []

        self.flows_layer = []
        self.extrinsics_layer = []
        self.intrinsics_layer = []

        self.inpaint_input_images = []
        self.disparities = []
        self.depths = []
        self.masks = []
        self.post_masks = []
        self.cameras = []
        self.cameras_archive = []


        self.config = config
        self.device = config["device"]

        self.inpainting_pipeline = inpainter_pipeline
        self.use_noprompt = False
        self.negative_inpainting_prompt = config['negative_inpainting_prompt']
        self.is_upper_mask_aggressive = False
        self.preservation_weight = config['preservation_weight']
        self.init_focal_length = config["init_focal_length"]

        self.decoder_learning_rate = config['decoder_learning_rate']
        self.dilate_mask_decoder_ft = config['dilate_mask_decoder_ft']

        self.depth_model = depth_model
        self.normal_estimator = normal_estimator
        self.depth_model_name = config['depth_model'].lower()
        self.depth_shift = config['depth_shift']
        self.very_far_depth = config['sky_hard_depth'] * 2


        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        self.points_3d_list = []
        self.colors_list = []
        self.floating_point_mask = None
        self.floating_point_mask_list = []
        self.sky_mask_list = []
        self.depth_cache = []
        self.floater_cluster_mask = torch.zeros(1, 1, 512, 512)
        self.user_clicks_xy = []

        self.current_pc_full = None
        self.current_pc_full_latest = None


        self.sam_mask=None
        self.hints=None

        self.prev_flow_valid = {"fore": False, "back": False}
        self.last_align = {
            "fore": {"R": None, "s": None},
            "back": {"R": None, "s": None},
        }

    @torch.no_grad()
    def set_frame_param(self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt):
        self.inpainting_resolution = inpainting_resolution
        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt


        self.border_mask = torch.ones(
            (1, 1, inpainting_resolution, inpainting_resolution)
        ).to(self.device)
        self.border_size = (inpainting_resolution - 512) // 2
        self.border_mask[:, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size] = 0
        self.border_image = torch.zeros(
            1, 3, inpainting_resolution, inpainting_resolution
        ).to(self.device)

    @torch.no_grad()
    def get_normal(self, image):


        normal = self.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format='pt',
        ).to(dtype=torch.float32)

        return normal

    def get_depth(self, image, archive_output=False, target_depth=None, mask_align=None, save_depth_to_cache=False, mask_farther=None, diffusion_steps=30, guidance_steps=8):
        assert self.depth_model is not None
        if self.depth_model_name == "midas":

            disparity = self.depth_model(dpt_transform(image))
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.depth_model_name == "midas_v3.1":
            img_transformed = dpt_512_transform(image)
            disparity = self.depth_model(img_transformed)
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        elif self.depth_model_name == "zoedepth":

            depth = self.depth_model(image)['metric_depth']
        elif self.depth_model_name == "marigold":

            image_input = (image*255).byte().squeeze().permute(1, 2, 0)
            image_input = Image.fromarray(image_input.cpu().numpy())
            depth = self.depth_model(
                image_input,
                denoising_steps=diffusion_steps,
                ensemble_size=1,
                processing_res=0,
                match_input_res=True,
                batch_size=0,
                color_map=None,
                show_progress_bar=True,
                depth_conditioning=self.config['depth_conditioning'],
                target_depth=target_depth,
                mask_align=mask_align,
                mask_farther=mask_farther,
                guidance_steps=guidance_steps,

                logger=self.logger,
            )

            depth = depth[None, None, :].to(dtype=torch.float32)
            depth /= 200
        elif self.depth_model_name in ["moge", "moge2", "moge_v2"]:

            img_input = image.squeeze(0).to(self.device).float().clamp(0, 1)
            with torch.inference_mode():
                out = self.depth_model.infer(img_input)


            depth = out.get("depth_metric", out.get("depth", out.get("depth_scale_invariant")))


            depth = torch.nan_to_num(depth, posinf=depth[torch.isfinite(depth)].max().item())


            d_min, d_max = depth.min(), depth.max()
            depth_relative = (depth - d_min) / (d_max - d_min + 1e-8)


            target_min, target_max = 0.001, 0.006
            depth = (depth_relative * (target_max - target_min)) + target_min


            depth = depth.to(torch.float32)
            if depth.ndim == 2:
                depth = depth[None, None]
            elif depth.ndim == 3:
                depth = depth.unsqueeze(0) if depth.shape[0] != 1 else depth.unsqueeze(1)


            elif torch.is_tensor(out):
                depth = out

            if depth is None:

                raise RuntimeError(f"MoGe output keys: {out.keys() if isinstance(out, dict) else 'Not a dict'}")


            if depth.ndim == 2:
                depth = depth[None, None]
            elif depth.ndim == 3:
                depth = depth[None, None] if depth.shape[0] != 1 else depth.unsqueeze(1)


            depth = depth.to(torch.float32).to(self.device)

        depth = depth + self.depth_shift
        disparity = 1 / depth

        if archive_output:
            self.depth_latest = depth
            self.disparity_latest = disparity

        if save_depth_to_cache:
            self.depth_cache.append(depth)

        return depth, disparity

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode = 'cv2_telea', self_guidance=False, style=None, inpainting_prompt=None, negative_prompt=None, mask_strategy=np.min, diffusion_steps=50):

        if self.inpainting_resolution > 512 and rendered_image.shape[-1] == 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size
            ] = inpaint_mask
            padded_rendered_image = self.border_image.clone()
            padded_rendered_image[
                :, :, self.border_size : self.inpainting_resolution-self.border_size, self.border_size : self.inpainting_resolution-self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_rendered_image = rendered_image


        img = (padded_rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = padded_inpainting_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (padded_inpainting_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img, _ = functbl[fill_mode](img, fill_mask_)


        mask_block_size = 8
        mask_boundary = mask.shape[0] // 2
        mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), mask_strategy)
        mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
        mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)

        if inpainting_prompt is not None:
            self.inpainting_prompt = inpainting_prompt
        if negative_prompt is None:
            negative_prompt = self.adaptive_negative_prompt + self.negative_inpainting_prompt if self.adaptive_negative_prompt != None else self.negative_inpainting_prompt

        inpainted_image = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else self.inpainting_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=diffusion_steps,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=self.inpainting_resolution,
            width=self.inpainting_resolution,
            self_guidance=self_guidance,
            inpaint_mask=~padded_inpainting_mask.bool(),
            rendered_image=padded_rendered_image,
        ).images[0]


        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)[None]

        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255

        self.post_mask_latest = post_mask
        self.inpaint_input_image_latest = init_image
        self.image_latest = inpainted_image

        return {"inpainted_image": inpainted_image,
                "padded_inpainting_mask": padded_inpainting_mask, "padded_rendered_image": padded_rendered_image}

    @torch.no_grad()
    def get_current_pc(self, is_detach=False, get_sky=False, combine=False, get_layer=False):

        if combine:
            if is_detach:
                return {k: v.detach() for k, v in self.get_combined_pc().items()}
            else:
                return self.get_combined_pc()

        elif get_sky:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_sky.items()}
            else:
                return self.current_pc_sky

        elif get_layer:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_layer.items()}
            else:
                return self.current_pc_layer

        else:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc.items()}
            else:
                return self.current_pc

    @torch.no_grad()
    def get_current_pc_latest(self, get_layer=False):
        if get_layer:
            return {k: v.detach() for k, v in self.current_pc_layer_latest.items()}
        else:
            return {k: v.detach() for k, v in self.current_pc_latest.items()}

    @torch.no_grad()
    def update_current_pc(self, points, colors, scene_flow, motion_mask, gen_sky=False, gen_layer=False, normals=None, focal_length=None):
        if gen_sky:
            if self.current_pc_sky is None:
                self.current_pc_sky = {"xyz": points, "rgb": colors, 'scene_flow':scene_flow, 'motion_mask':motion_mask}
            else:
                self.current_pc_sky["xyz"] = torch.cat([self.current_pc_sky["xyz"], points], dim=0)
                self.current_pc_sky["rgb"] = torch.cat([self.current_pc_sky["rgb"], colors], dim=0)
                self.current_pc_sky["scene_flow"] = torch.cat([self.current_pc_sky["scene_flow"], scene_flow], dim=0)
                self.current_pc_sky["motion_mask"] = torch.cat([self.current_pc_sky["motion_mask"], motion_mask], dim=0)
        elif gen_layer:
            if self.current_pc_layer is None:
                self.current_pc_layer = {"xyz": points, "rgb": colors, 'scene_flow':scene_flow, 'motion_mask':motion_mask}
            else:
                self.current_pc_layer["xyz"] = torch.cat([self.current_pc_layer["xyz"], points], dim=0)
                self.current_pc_layer["rgb"] = torch.cat([self.current_pc_layer["rgb"], colors], dim=0)
                self.current_pc_layer["scene_flow"] = torch.cat([self.current_pc_layer["scene_flow"], scene_flow], dim=0)
                self.current_pc_layer["motion_mask"] = torch.cat([self.current_pc_layer["motion_mask"], motion_mask], dim=0)
            self.current_pc_layer_latest = {"xyz": points, "rgb": colors, 'normals': normals, 'scene_flow':scene_flow, 'motion_mask':motion_mask}
        else:
            if self.current_pc is None:
                self.current_pc = {"xyz": points, "rgb": colors, 'scene_flow':scene_flow, 'motion_mask':motion_mask}
            else:
                self.current_pc["xyz"] = torch.cat([self.current_pc["xyz"], points], dim=0)
                self.current_pc["rgb"] = torch.cat([self.current_pc["rgb"], colors], dim=0)
                self.current_pc["scene_flow"] = torch.cat([self.current_pc["scene_flow"], scene_flow], dim=0)
                self.current_pc["motion_mask"] = torch.cat([self.current_pc["motion_mask"], motion_mask], dim=0)
            self.current_pc_latest = {"xyz": points, "rgb": colors, 'normals': normals, 'scene_flow':scene_flow, 'motion_mask':motion_mask}


    @torch.no_grad()
    def get_combined_pc(self):
        if self.current_pc_layer is None:
            pc = {"xyz": torch.cat([self.current_pc["xyz"], self.current_pc_sky["xyz"]], dim=0), "rgb": torch.cat([self.current_pc["rgb"], self.current_pc_sky["rgb"]], dim=0)}
        else:
            pc = {"xyz": torch.cat([self.current_pc["xyz"], self.current_pc_sky["xyz"], self.current_pc_layer["xyz"]], dim=0), "rgb": torch.cat([self.current_pc["rgb"], self.current_pc_sky["rgb"], self.current_pc_layer["rgb"]], dim=0)}
        return pc


    @torch.no_grad()
    def push_away_inconsistent_points(self, inconsistent_point_index, depth, mask):
        h, w = depth.shape[2:]
        depth = rearrange(depth.clone(), "b c h w -> (w h b) c")
        extract_mask = rearrange(mask, "b c h w -> (w h b) c")[:, 0].bool()
        depth_extracted = depth[extract_mask]
        if inconsistent_point_index.shape[0] > 0:
            assert depth_extracted.shape[0] >= inconsistent_point_index.max() + 1
        depth_extracted[inconsistent_point_index] = self.very_far_depth
        depth[extract_mask] = depth_extracted
        depth = rearrange(depth, "(w h b) c -> b c h w", w=w, h=h)
        return depth

    @torch.no_grad()
    def archive_latest(self, idx=0, vmax=0.006):
        if self.config['gen_layer']:
            self.images_layer.append(self.image_latest)
            self.images.append(self.image_latest_init)
            self.flows_layer.append(self.flow_latest)
            self.extrinsics_layer.append(self.extrinsics_latest)
            self.intrinsics_layer.append(self.intrinsics_latest)
        else:
            self.images.append(self.image_latest)


        self.masks.append(self.mask_latest)
        self.post_masks.append(self.post_mask_latest)
        self.inpaint_input_images.append(self.inpaint_input_image_latest)
        self.depths.append(self.depth_latest)
        self.disparities.append(self.disparity_latest)

        save_root = Path(self.run_dir) / "images"
        save_root.mkdir(exist_ok=True, parents=True)


        (save_root / "frames").mkdir(exist_ok=True, parents=True)
        (save_root / "frames_init").mkdir(exist_ok=True, parents=True)


        ToPILImage()(self.image_latest[0]).save(save_root / "frames" / f"{idx:03d}.png")
        if self.config['gen_layer']:
            ToPILImage()(self.image_latest_init[0]).save(save_root / "frames_init" / f"{idx:03d}.png")


        if idx == 0:
            with open(Path(self.run_dir) / "config.yaml", "w") as f:
                OmegaConf.save(self.config, f)

    @torch.no_grad()
    def increment_kf_idx(self):
        self.kf_idx += 1

    @torch.no_grad()
    def convert_to_3dgs_traindata(self, xyz_scale=1.0, remove_threshold=None, use_no_loss_mask=True):
        train_datas = []
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        current_pc = self.get_current_pc(is_detach=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_scene_flow = current_pc['scene_flow'].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_motion_mask = current_pc['motion_mask'].cpu().numpy()

        if remove_threshold is not None:
            remove_threshold_scaled = remove_threshold * xyz_scale
            mask = np.linalg.norm(pcd_points, axis=0) >= remove_threshold_scaled
            pcd_points = pcd_points[:, ~mask]
            pcd_colors = pcd_colors[~mask]


        frames = []

        for i, img in enumerate(self.images):
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = self.cameras[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}
        train_datas.append(train_data)


        current_pc = self.sky_pc_downsampled
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = pcd_points / np.linalg.norm(pcd_points, axis=1, keepdims=True)
        pcd_normals = pcd_normals.T

        pcd_scene_flow = current_pc['scene_flow'].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_motion_mask = current_pc['motion_mask'].cpu().numpy()

        frames = []

        for i, camera in enumerate(self.sky_cameras):
            self.current_camera = camera
            render_output = self.render(render_sky=True)

            if render_output['inpaint_mask'].mean() > 0:
                render_output['rendered_image'] = inpaint_cv2(render_output['rendered_image'], render_output['inpaint_mask'])
            no_loss_mask = render_output['inpaint_mask'][0]

            image = ToPILImage()(render_output['rendered_image'][0])
            save_root = Path(self.run_dir) / "images"


            transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data_sky = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}
        train_datas.append(train_data_sky)

        if self.config['gen_layer']:
            current_pc = self.get_current_pc(is_detach=True, get_layer=True)
            pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
            pcd_colors = current_pc["rgb"].cpu().numpy()

            frames = []

            for i, img in enumerate(self.images_layer):
                image = ToPILImage()(img[0])
                no_loss_mask = self.no_loss_masks_layer[i][0]  if use_no_loss_mask else None
                transform_matrix_pt3d = self.cameras[i].get_world_to_view_transform().get_matrix()[0]
                transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
                transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

                transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

                opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
                transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

                transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
                frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
                frames.append(frame)
            train_data_layer = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}
            train_datas.append(train_data_layer)

        return train_datas

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest(self, xyz_scale=1.0, points_3d=None, colors=None, use_no_loss_mask=False, use_only_latest_frame=True):
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))
        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()
        pcd_scene_flow = current_pc['scene_flow'].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_motion_mask = current_pc['motion_mask'].cpu().numpy()

        frames = []

        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': no_loss_mask}
            frames.append(frame)
        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}

        return train_data

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest_layer(self, xyz_scale=1.0, points_3d=None, colors=None, use_only_latest_frame=True):
        W, H = 512, 512
        camera_angle_x = 2*np.arctan(W / (2*self.init_focal_length))


        sam_points = self.user_clicks_xy

        current_pc = self.get_current_pc_latest(get_layer=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()
        pcd_scene_flow = current_pc['scene_flow'].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_motion_mask = current_pc['motion_mask'].cpu().numpy()

        frames = []
        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': None}
            frames.append(frame)

        train_data = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}

        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc['normals'].cpu().numpy()
        pcd_scene_flow = current_pc['scene_flow'].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_motion_mask = current_pc['motion_mask'].cpu().numpy()


        frames = []
        images = self.images_layer
        extrinsics_layer = self.extrinsics_layer
        intrinsics_layer = self.intrinsics_layer
        flows_layer = self.flows_layer

        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(torch.tensor([-1., 1, -1, 1], device=self.device))
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {'image': image, 'transform_matrix': transform_matrix, 'no_loss_mask': None}
            frames.append(frame)

        flow_data = []
        for i, img in enumerate(images):
            if i != len(images) - 1:
                continue
            flow = {'flow': flows_layer[i], 'extrinsics': extrinsics_layer[i], 'intrinsics': intrinsics_layer[i]}
            flow_data.append(flow)


        train_data_layer = {'frames': frames, 'pcd_points': pcd_points, 'pcd_colors': pcd_colors, 'pcd_normals': pcd_normals, 'camera_angle_x': camera_angle_x, 'W': W, 'H': H, 'pcd_scene_flow':pcd_scene_flow, 'pcd_motion_mask':pcd_motion_mask}
        return train_data, train_data_layer, flow_data

    @torch.no_grad()
    def get_knn_mask(self, pad_width=1):
        print("-- knn heuristic, removing floating points...")
        depth_map = self.depth_latest.squeeze().detach().cpu().numpy()
        height, width = depth_map.shape
        padded_depth_map = np.pad(depth_map, pad_width=pad_width, mode='constant', constant_values=0)
        cleaned_depth_map = np.zeros_like(depth_map)

        for dy in range(-pad_width, pad_width+1):
            for dx in range(-pad_width, pad_width+1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_diff = np.abs(padded_depth_map[pad_width+dy:height+pad_width+dy, pad_width+dx:width+pad_width+dx] - depth_map)
                cleaned_depth_map += (neighbor_diff > 0.00001)

        knn_mask = torch.from_numpy(cleaned_depth_map == 8)
        print("-- floating points ratio: {}".format(knn_mask.float().mean()))
        ToPILImage()(knn_mask.float()).save(self.run_dir / 'images' / 'knn_masks' / f"{self.kf_idx:02d}_knn_mask.png")

        return knn_mask


    @torch.no_grad()
    def render_pcd_to_image(self, points_3d, colors, kf_camera, H=512, W=512):
        try:
            pixel_coords_float, depth = kf_camera.project(points_3d)
        except ValueError:

            pixel_coords_float = kf_camera.project(points_3d)
            depth = points_3d[:, 2]

        pixel_xy = pixel_coords_float.round().long()


        valid_mask = (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < W) & \
                    (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < H)

        valid_xy = pixel_xy[valid_mask]
        valid_colors = colors[valid_mask]
        valid_depth = depth[valid_mask]


        image = torch.ones((H, W, 3), dtype=valid_colors.dtype, device=valid_colors.device)


        y_coords = valid_xy[:, 1]
        x_coords = valid_xy[:, 0]


        pixel_flat_indices = y_coords * W + x_coords


        sort_keys = -valid_depth
        sorted_indices = torch.argsort(sort_keys, descending=False)


        sorted_y = y_coords[sorted_indices]
        sorted_x = x_coords[sorted_indices]
        sorted_colors = valid_colors[sorted_indices]


        image[sorted_y, sorted_x] = sorted_colors

        return image


    @torch.no_grad()
    def update_current_pc_by_kf(self, valid_mask=None, sky_mask=None, gen_layer=False, image=None, depth=None, camera=None, flow=None, motion_mask=None, hints=None, first=False, fore=False, back=False):
        if image is None:
            image = self.image_latest
        if depth is None:
            depth = self.depth_latest
        if camera is None:
            camera = self.current_camera


        key = "back" if back else "fore"

        prev_can_align = self.prev_flow_valid[key]
        cur_flow_valid = (flow is not None)
        print("prev_can_align:",prev_can_align)

        kf_camera = convert_pytorch3d_kornia(camera, self.init_focal_length)
        focal_length= self.init_focal_length
        point_depth = rearrange(depth, "b c h w -> (w h b) c")
        normals = self.get_normal(image[0])
        normals[:, 1:] *= -1


        normals_world = kf_camera.rotation_matrix.inverse() @ rearrange(normals, 'b c h w -> b c (h w)')


        check(flow, "flow_raw")

        normals = rearrange(normals_world, 'b c (h w) -> b c h w', h=512)
        new_normals = rearrange(normals, "b c h w -> (w h b) c")
        new_points_3d = kf_camera.unproject(self.points, point_depth)
        if self.config['gen_layer']:
            if fore:
                print("foreground")
                if flow is not None:
                    print("Foreground flow is not None")

                    if self.config['use_mom'] and first:
                        print("first")
                        new_scene_flow = flow
                        H, W = 512, 512

                        new_scene_flow = new_scene_flow.view(H, W, 3)
                        new_scene_flow = new_scene_flow.permute(1, 0, 2)
                        new_scene_flow = new_scene_flow.reshape(-1, 3)
                        new_scene_flow[:, :2] *= -1

                        print("new_scene_flow.shape",new_scene_flow.shape)
                        print("new_scene_flow",new_scene_flow)
                    else:
                        flow= flow.detach().cpu()
                        flow = flow.squeeze(0).permute(2, 1, 0)

                        x = torch.arange(512).float() + 0.5
                        y = torch.arange(512).float() + 0.5
                        flow_ponints = torch.stack(torch.meshgrid((x), (y), indexing='ij'), -1) + flow


                        flow_ponints = rearrange(flow_ponints, "h w c -> (h w) c").to(self.device)
                        flow_points_3d = kf_camera.unproject(flow_ponints, point_depth)

                        new_scene_flow = flow_points_3d-new_points_3d
                        print("new_scene_flow.shape",new_scene_flow.shape)
                        print("new_scene_flow",new_scene_flow)

                    if (self.current_pc_layer_latest is not None) and prev_can_align:
                        print("[DEBUG] sky_mask type:", type(sky_mask))
                        try:
                            print("[DEBUG] sky_mask shape:", sky_mask.shape)
                        except Exception as e:
                            print("[DEBUG] sky_mask has no .shape:", e)


                        import numpy as np
                        if isinstance(sky_mask, np.ndarray):
                            print("[DEBUG] sky_mask dtype:", sky_mask.dtype)

                        sky_mask = rearrange(sky_mask, "b c h w -> (w h b) c")[:, 0].bool()
                        new_colors = rearrange(image, "b c h w -> (w h b) c")
                        new_motion_mask = rearrange(torch.from_numpy(motion_mask).unsqueeze(0).unsqueeze(0), "b c h w -> (w h b) c")[:, 0].bool().cuda()


                else :
                    print("Background flow is None")
                    flow_points_3d = new_points_3d
                    new_scene_flow = flow_points_3d-new_points_3d
            else:
                print("background")
                if flow is not None:
                    print("Background flow is not None")
                    if self.config['use_mom'] and first:
                        new_scene_flow = flow
                        H, W = 512, 512

                        new_scene_flow = new_scene_flow.view(H, W, 3)
                        new_scene_flow = new_scene_flow.permute(1, 0, 2)
                        new_scene_flow = new_scene_flow.reshape(-1, 3)
                        new_scene_flow[:, :2] *= -1

                        print("new_scene_flow.shape",new_scene_flow.shape)
                        print("new_scene_flow",new_scene_flow)
                    else:

                        flow= flow.detach().cpu()
                        flow = flow.squeeze(0).permute(2, 1, 0)

                        x = torch.arange(512).float() + 0.5
                        y = torch.arange(512).float() + 0.5
                        flow_ponints = torch.stack(torch.meshgrid((x), (y), indexing='ij'), -1) + flow


                        flow_ponints = rearrange(flow_ponints, "h w c -> (h w) c").to(self.device)
                        flow_points_3d = kf_camera.unproject(flow_ponints, point_depth)

                        new_scene_flow = flow_points_3d-new_points_3d

                        print("new_scene_flow.shape",new_scene_flow.shape)
                        print("new_scene_flow",new_scene_flow)

                    if (self.current_pc_latest is not None) and prev_can_align:
                        print("[DEBUG] sky_mask type:", type(sky_mask))
                        try:
                            print("[DEBUG] sky_mask shape:", sky_mask.shape)
                        except Exception as e:
                            print("[DEBUG] sky_mask has no .shape:", e)


                        import numpy as np
                        if isinstance(sky_mask, np.ndarray):
                            print("[DEBUG] sky_mask dtype:", sky_mask.dtype)

                        sky_mask = rearrange(sky_mask, "b c h w -> (w h b) c")[:, 0].bool()
                        new_colors = rearrange(image, "b c h w -> (w h b) c")
                        new_motion_mask = rearrange(torch.from_numpy(motion_mask).unsqueeze(0).unsqueeze(0), "b c h w -> (w h b) c")[:, 0].bool().cuda()


                else :
                    print("Foreground flow is None")
                    flow_points_3d = new_points_3d
                    new_scene_flow = flow_points_3d-new_points_3d
        else:
            if flow is not None:
                if self.config['use_mom'] and first:
                    new_scene_flow = flow
                    H, W = 512, 512

                    new_scene_flow = new_scene_flow.view(H, W, 3)
                    new_scene_flow = new_scene_flow.permute(1, 0, 2)
                    new_scene_flow = new_scene_flow.reshape(-1, 3)
                    new_scene_flow[:, :2] *= -1

                    print("new_scene_flow.shape",new_scene_flow.shape)
                    print("new_scene_flow",new_scene_flow)
                else:
                    flow= flow.detach().cpu()
                    flow = flow.squeeze(0).permute(2, 1, 0)

                    x = torch.arange(512).float() + 0.5
                    y = torch.arange(512).float() + 0.5
                    flow_ponints = torch.stack(torch.meshgrid((x), (y), indexing='ij'), -1) + flow


                    flow_ponints = rearrange(flow_ponints, "h w c -> (h w) c").to(self.device)
                    flow_points_3d = kf_camera.unproject(flow_ponints, point_depth)

                    new_scene_flow = flow_points_3d-new_points_3d

                    print("new_scene_flow.shape",new_scene_flow.shape)
                    print("new_scene_flow",new_scene_flow)
                    check(new_scene_flow, "new_scene_flow_before_match")

                if (self.current_pc_latest is not None) and prev_can_align:
                    print("[DEBUG] sky_mask type:", type(sky_mask))
                    if sky_mask is None:
                        sky_mask = torch.ones_like(image[:, :1, :, :], dtype=torch.bool, device=image.device)

                    sky_mask = rearrange(sky_mask, "b c h w -> (w h b) c")[:, 0].bool()
                    new_colors = rearrange(image, "b c h w -> (w h b) c")
                    new_motion_mask = rearrange(torch.from_numpy(motion_mask).unsqueeze(0).unsqueeze(0), "b c h w -> (w h b) c")[:, 0].bool().cuda()


            else:
                flow_points_3d = new_points_3d
                new_scene_flow = flow_points_3d-new_points_3d


        new_colors = rearrange(image, "b c h w -> (w h b) c")
        if motion_mask is not None:
            motion_mask = torch.from_numpy(motion_mask).cuda()
            motion_mask = motion_mask.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
            new_motion_mask = motion_mask.float() * 255
        else:
            motion_mask = torch.zeros(1, 3, 512, 512).cuda()
            new_motion_mask = motion_mask.float() * 255
        new_motion_mask = rearrange(new_motion_mask, "b c h w -> (w h b) c")

        if valid_mask is not None:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
            device = extract_mask.device
            new_scene_flow = new_scene_flow.to(device)
            print("new_scene_flow device:", new_scene_flow.device)
            print("extract_mask device:", extract_mask.device)
            print("new_points_3d device:", new_points_3d.device)
            new_points_3d = new_points_3d[extract_mask]
            new_colors = new_colors[extract_mask]
            new_normals = new_normals[extract_mask]
            new_scene_flow = new_scene_flow[extract_mask]
            new_motion_mask = new_motion_mask[extract_mask]


        extrinsics = kf_camera.extrinsics
        intrinsics = kf_camera.intrinsics
        self.extrinsics_latest = extrinsics
        self.intrinsics_latest = intrinsics

        self.update_current_pc(new_points_3d, new_colors, scene_flow=new_scene_flow, motion_mask=new_motion_mask, normals=new_normals, gen_layer=gen_layer, focal_length=focal_length)

        self.prev_flow_valid[key] = cur_flow_valid

        return new_points_3d, new_colors


from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


_SAM3_PROCESSOR = None

def get_sam3_processor():
    global _SAM3_PROCESSOR
    if _SAM3_PROCESSOR is None:
        model = build_sam3_image_model()
        _SAM3_PROCESSOR = Sam3Processor(model)
    return _SAM3_PROCESSOR

def sam3_segment_all_union_after(
    image_rgb_uint8: np.ndarray,
    prompt: Union[str, Iterable[str]],
    processor: "Sam3Processor | None" = None,
    split_commas: bool = True,
):
    assert image_rgb_uint8.ndim == 3 and image_rgb_uint8.shape[2] == 3
    H, W = image_rgb_uint8.shape[:2]

    if processor is None:
        processor = get_sam3_processor()

    pil = Image.fromarray(image_rgb_uint8, mode="RGB")


    state = processor.set_image(pil)


    if isinstance(prompt, str):
        if split_commas:
            prompts = [p.strip() for p in prompt.split(",") if p.strip()]
        else:
            prompts = [prompt]
    else:
        prompts = [str(p).strip() for p in prompt if str(p).strip()]

    if len(prompts) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, (0, 0, 0, 0), empty[None, ...]


    masks_list = []
    for p in prompts:
        output = processor.set_text_prompt(state=state, prompt=p)
        masks = output.get("masks", None)
        if masks is None:
            continue


        if torch.is_tensor(masks):
            m = masks.detach().cpu().numpy()
        else:
            m = np.asarray(masks)


        if m.ndim == 2:
            m = m[None, ...]
        elif m.ndim == 4:
            if m.shape[0] == 1:
                m = m[0, ...]
            if m.ndim == 4 and m.shape[1] == 1:
                m = m[:, 0, :, :]

        if m.ndim != 3 or m.shape[0] == 0:
            continue

        masks_list.append(m.astype(bool))

    if len(masks_list) == 0:
        empty = np.zeros((H, W), dtype=bool)
        return empty, (0, 0, 0, 0), empty[None, ...]

    masks_all = np.concatenate(masks_list, axis=0)


    union_mask = np.any(masks_all, axis=0)


    ys, xs = np.where(union_mask)
    if ys.size == 0:
        bbox = (0, 0, 0, 0)
    else:


        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

    return union_mask, bbox, masks_all


def _next_index(out_dir, prefix, exts=("png","jpg","jpeg")):
    os.makedirs(out_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.(?:{'|'.join(exts)})$")
    max_idx = -1
    for p in glob.glob(os.path.join(out_dir, f"{prefix}_*")):
        m = pat.match(os.path.basename(p))
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1

def viz_flow(train_data, viz_dir):
    for idx, frame in enumerate(train_data['frames']):
        our_flow = frame["our_flow"][0]
        viz_flow = flow2img(our_flow[0])
        our_flow_path = os.path.join(viz_dir, str(idx).zfill(3)+'.png')
        save_image(viz_flow, our_flow_path)

def viz_flow_test(flow, out_dir, prefix="flow_viz", ext="png"):
    os.makedirs(out_dir, exist_ok=True)
    idx = _next_index(out_dir, prefix)
    fname = f"{prefix}_{idx:06d}.{ext}"
    save_path = os.path.join(out_dir, fname)

    img = flow2img(flow[0])
    Image.fromarray(img).save(save_path)
    return save_path

def estimate_flow_test(frame, mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y,  args):
    viz_dir = os.path.join(args.input_dir, 'flow_viz')
    os.makedirs(viz_dir, exist_ok=True)
    print("final_hint_start_x-3d",final_hint_start_x)
    print("final_hint_end_x-3d",final_hint_end_x)
    print("final_hint_start_y-3d",final_hint_start_y)
    print("final_hint_end_y-3d",final_hint_end_y)
    frame = {'image': frame, 'mask':mask, 'final_hint_start_x':final_hint_start_x, 'final_hint_start_y':final_hint_start_y, 'final_hint_end_x':final_hint_end_x, 'final_hint_end_y':final_hint_end_y}


    flow = eulerian_estimation(args, frame)


    viz_flow_test(flow, viz_dir)
    return flow

class KeyframeGen(FrameSyn):
    def __init__(self, config, inpainter_pipeline, depth_model, mask_generator,
                 segment_model=None, segment_processor=None, normal_estimator=None,
                 rotation_path=None, inpainting_resolution=None, socketio=None, client_id=None):
        super().__init__(config, inpainter_pipeline=inpainter_pipeline, depth_model=depth_model, normal_estimator=normal_estimator)


        self.rendered_image_latest = torch.zeros(1, 3, 512, 512)
        self.rendered_depth_latest = torch.zeros(1, 1, 512, 512)
        self.no_loss_mask_latest = torch.zeros(1, 1, 512, 512).bool()
        self.no_loss_mask_latest_layer = torch.zeros(1, 1, 512, 512).bool()
        self.current_camera = None

        self.rendered_images = []
        self.rendered_depths = []
        self.no_loss_masks = []
        self.no_loss_masks_layer = []


        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Gen-{dt_string}"
        self.logger = SimpleLogger(self.run_dir / "log.txt")
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.sky_hard_depth = config['sky_hard_depth']
        self.sky_erode_kernel_size = config['sky_erode_kernel_size']
        self.is_upper_mask_aggressive = False

        self.rotation_range_theta = config['rotation_range']
        self.interp_frames = config['frames']
        self.camera_speed = config["camera_speed"]
        self.camera_speed_multiplier_rotation = config["camera_speed_multiplier_rotation"]


        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'images' / "knn_masks").mkdir(exist_ok=True, parents=True)

        (self.run_dir / 'images' / "depth_should_be").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "depth_conditioned").mkdir(exist_ok=True, parents=True)

        (self.run_dir / 'images' / "layer").mkdir(exist_ok=True, parents=True)
        (self.run_dir / 'images' / "disparity_gradient").mkdir(exist_ok=True, parents=True)


        self.scene_cameras_idx = []
        self.center_camera_idx = None
        self.generate_cameras(rotation_path)
        self.cameras_users = []
        self.inpainting_resolution = inpainting_resolution
        self.sam_mask = torch.empty(0, dtype=torch.bool).cuda()

        self.socketio = socketio
        self.client_id = client_id


        self.GT_list = []

    @torch.no_grad()
    def get_camera_at_origin(self, big_view=False):
        if big_view:
            K = torch.zeros((1, 4, 4), device=self.device)
            K[0, 0, 0] = 500
            K[0, 1, 1] = 500
            K[0, 0, 2] = 768
            K[0, 1, 2] = 256
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.zeros((1, 3), device=self.device)
            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        else:
            K = torch.zeros((1, 4, 4), device=self.device)
            K[0, 0, 0] = self.init_focal_length
            K[0, 1, 1] = self.init_focal_length
            K[0, 0, 2] = 256
            K[0, 1, 2] = 256
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.zeros((1, 3), device=self.device)
            camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return camera

    @torch.no_grad()
    def recompose_image_latest_and_set_current_pc(self, scene_name=None, args=None, points_xy= None, hints=None, sam_prompt=None, ok_event=None):
        print("recompose called")
        self.set_current_camera(self.get_camera_at_origin(), archive_camera=True)
        sem_map = self.update_sky_mask()
        render_output = self.render(render_sky=True)
        self.image_latest = soft_stitching(render_output["rendered_image"], self.image_latest, self.sky_mask_latest)

        ground_mask = self.generate_ground_mask(sem_map=sem_map)[None, None]
        depth_should_be_ground = self.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        with torch.no_grad():
            depth_guided, _ = self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be_ground, mask_align=(ground_mask & ground_outputable_mask),
                                             diffusion_steps=30, guidance_steps=8)
        self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())


        H, W= 512, 512
        K, fov = camera_params(H, W, self.init_focal_length)

        if self.config['gen_layer']:
            self.generate_layer(pred_semantic_map=sem_map, scene_name=scene_name)
            depth_should_be = self.depth_latest_init
            mask_to_align_depth = ~(self.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = self.mask_disocclusion.bool() & (depth_should_be < 0.006)
            with torch.no_grad():
                self.depth, self.disparity = self.get_depth(self.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth,
                                                            diffusion_steps=30, guidance_steps=8)

            self.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                                             existing_mask=~(self.mask_disocclusion).bool().squeeze().cpu().numpy(),
                                             existing_disp=self.disparity_latest_init.squeeze().cpu().numpy())
            wrong_depth_mask = self.depth_latest<self.depth_latest_init
            self.depth_latest[wrong_depth_mask] = self.depth_latest_init[wrong_depth_mask] + 0.0001
            self.depth_latest = self.mask_disocclusion * self.depth_latest + (1-self.mask_disocclusion) * self.depth_latest_init
            self.update_sky_mask()
            if self.config['load_gen']:
                background_mask = None
                foreground_mask = None
                background_flow = None
                foreground_flow = None
                fore = False
            else:


                mask_dis = self.mask_disocclusion


                if mask_dis.dim() == 3:
                    mask_dis = mask_dis.squeeze(0)

                mask_dis_np = mask_dis.detach().cpu().numpy()


                mask_dis_np = (mask_dis_np      > 0).astype(np.uint8) * 255

                save_mask_incremental(mask_dis_np, args.input_dir, prefix="mask_disocclusion")


                print("sam_prompt in recompose: ",sam_prompt)
                final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = make_final_hints_xy(hints, H, W)
                start_keyframe_path = Path(args.input_dir) / "start_keyframe.png"
                pil_image_latest = Image.open(start_keyframe_path).convert("RGB").resize((W, H))
                image_for_mask = np.asarray(pil_image_latest, dtype=np.uint8)


                save_image_incremental(image_for_mask, args.input_dir)
                save_image_incremental_fore(self.image_latest_init, args.input_dir)
                if hints is None or len(hints) == 0:
                    mask = None
                else:
                    mask, bbox, _ = sam3_segment_all_union_after(
                        image_for_mask,
                        prompt=sam_prompt,
                        split_commas=True,
                    )
                    save_mask_incremental(mask, args.input_dir, prefix="sam3_mask")


                if mask is None:
                    common_mask = None
                else:
                    print("multiple_disocclusion")
                    sam_np = mask

                    if sam_np.ndim == 3:
                        sam_np = sam_np[..., 0]
                    sam_np = (sam_np > 0)

                    print("sam_np sum:", sam_np.sum())


                    common_mask = sam_np & mask_dis_np


                    save_mask_incremental(common_mask.astype(np.uint8) * 255, args.input_dir, prefix="common_sam_disocc")


                    print("About to emit, socketio:", self.socketio, "client_id:", self.client_id)
                    if self.socketio and self.client_id:
                        print("Emitting show-mask-preview to client", self.client_id)
                        print("Emitting show-mask-preview to client", self.client_id)
                        print("Emitting show-mask-preview to client", self.client_id)
                        pil_img = Image.fromarray(image_for_mask)
                        alpha_channel = np.full((H, W), 255, dtype=np.uint8)
                        alpha_channel[sam_np] = int(255 * 0.5)
                        pil_alpha = Image.fromarray(alpha_channel, mode='L')
                        pil_img_rgba = pil_img.convert('RGBA')
                        pil_img_rgba.putalpha(pil_alpha)
                        buffered = io.BytesIO()
                        pil_img_rgba.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        print("Image base64 length:", len(img_str))
                        self.socketio.emit('show-mask-preview', {'image': img_str}, room=self.client_id)
                        if ok_event:
                            ok_event.clear()
                            self.socketio.emit('server-state', 'Check the mask and click OK to proceed...', room=self.client_id)
                            ok_event.wait()


                mask_dis = self.mask_disocclusion
                if mask_dis.dim() == 3:
                    mask_dis = mask_dis.squeeze(0)
                elif mask_dis.dim() == 4:
                    mask_dis = mask_dis[0, 0]

                mask_dis_np = mask_dis.detach().cpu().numpy().astype(bool)


                if mask is None:
                    foreground_mask = None
                else:
                    sam_np = mask
                    if sam_np.ndim == 3:
                        sam_np = sam_np[..., 0]
                    sam_np = (sam_np > 0)

                intersection = np.logical_and(sam_np, mask_dis_np).sum()
                union = np.logical_or(sam_np, mask_dis_np).sum()

                iou = intersection / (union + 1e-6)
                print("mask IoU:", iou)

                IOU_THRESHOLD = 0.7

                if iou > IOU_THRESHOLD:
                    print("Use common foreground (intersection)")
                    foreground_mask = sam_np & mask_dis_np
                    background_mask = None
                else:
                    print("Use SAM mask only")
                    background_mask = sam_np
                    foreground_mask = None


                hints_list = hints.tolist()


                pcdgenpath='lookaround5'
                render_poses = get_pcdGenPoses(pcdgenpath)

                internel_render_poses = get_pcdGenPoses('hemispherei')


                if background_mask is not None:
                    print("background_flow")
                    if self.config['use_mom']:
                        train_data, none_idx = render_PCD(pil_image_latest, background_mask, hints_list, self.depth_latest, K, fov, render_poses, internel_render_poses)
                        viz_dir1 = os.path.join(args.input_dir, 'bo_Flow_viz')
                        os.makedirs(viz_dir1, exist_ok=True)
                        train_data = estimate_flow(train_data, viz_dir1, args)

                        with torch.enable_grad():
                            train_data, background_flow = optimize_motion(train_data, none_idx, 200, K, render_poses, internel_render_poses)
                        viz_dir = os.path.join(args.input_dir, 'ao_flow_viz')
                        os.makedirs(viz_dir, exist_ok=True)
                        viz_flow(train_data, viz_dir)
                        background_flow = background_flow.permute(1, 0)

                    else:
                        background_flow=estimate_flow_test(pil_image_latest, background_mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)
                    fore=False
                else:
                    background_flow=None

                if foreground_mask is not None:
                    print("foreground_flow")
                    if self.config['use_mom']:
                        train_data, none_idx = render_PCD(pil_image_latest, foreground_mask, hints_list, self.depth_latest_init, K, fov, render_poses, internel_render_poses)
                        viz_dir1 = os.path.join(args.input_dir, 'bo_Flow_viz')
                        os.makedirs(viz_dir1, exist_ok=True)
                        train_data = estimate_flow(train_data, viz_dir1, args)
                        with torch.enable_grad():
                            train_data, foreground_flow = optimize_motion(train_data, none_idx, 200, K, render_poses, internel_render_poses)
                        viz_dir = os.path.join(args.input_dir, 'ao_flow_viz')
                        os.makedirs(viz_dir, exist_ok=True)
                        viz_flow(train_data, viz_dir)
                        foreground_flow = foreground_flow.permute(1, 0)
                    else:
                        foreground_flow=estimate_flow_test(pil_image_latest, foreground_mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)
                    fore=True
                else:
                    foreground_flow=None

            if self.config['load_gen']:
                first=False
            else:
                first=True


            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest, flow=background_flow, motion_mask=background_mask, hints=hints, first=first, fore=fore, back=True)


            self.update_current_pc_by_kf(image=self.image_latest_init, depth=self.depth_latest_init, valid_mask=self.mask_disocclusion, flow=foreground_flow, motion_mask=foreground_mask, hints=hints, first=first, gen_layer=True, fore=fore, back=False)
        else:
            print("not gen_layer")
            print("sam_prompt in recompose: ",sam_prompt)
            final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = make_final_hints_xy(hints, H, W)
            hints_list = hints.tolist()
            start_keyframe_path = Path(args.input_dir) / "start_keyframe.png"
            pil_image_latest = Image.open(start_keyframe_path).convert("RGB").resize((W, H))
            image_for_mask = np.asarray(pil_image_latest, dtype=np.uint8)
            save_image_incremental(image_for_mask, args.input_dir)
            pil_image_latest = ToPILImage()(self.image_latest[0].detach().cpu().clamp(0., 1.))
            image_for_mask2 = np.asarray(pil_image_latest.convert("RGB"), dtype=np.uint8)
            save_image_incremental(image_for_mask2, args.input_dir, prefix="input_image")

            if not has_motion_hints(hints):
                mask = None
                flow = None
            else:
                mask, bbox, _ = sam3_segment_all_union_after(
                    image_for_mask,
                    prompt=sam_prompt,
                    split_commas=True,
                )

            pcdgenpath='lookaround5'
            render_poses = get_pcdGenPoses(pcdgenpath)

            internel_render_poses = get_pcdGenPoses('hemispherei')

            if has_motion_hints(hints):
                if self.config['use_mom']:
                    train_data, none_idx = render_PCD(pil_image_latest, mask, hints_list, self.depth_latest, K, fov, render_poses, internel_render_poses)
                    viz_dir1 = os.path.join(args.input_dir, 'bo_Flow_viz')
                    os.makedirs(viz_dir1, exist_ok=True)
                    train_data = estimate_flow(train_data, viz_dir1, args)

                    with torch.enable_grad():
                        train_data, flow = optimize_motion(train_data, none_idx, 200, K, render_poses, internel_render_poses)
                    viz_dir = os.path.join(args.input_dir, 'ao_flow_viz')
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_flow(train_data, viz_dir)
                    flow = flow.permute(1, 0)
                else:
                    flow = estimate_flow_test(pil_image_latest, mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)
            first=True
            self.update_current_pc_by_kf(image=self.image_latest, depth=self.depth_latest, valid_mask=~self.sky_mask_latest, flow=flow, motion_mask=mask, hints=hints, first=first)

        self.archive_latest()

    @torch.no_grad()
    def compute_ground_depth(self, camera_height = 0.0003):
        focal_length = self.init_focal_length
        x_res, y_res = 512, 512
        y_principal = 256


        y_grid = torch.arange(y_res).view(1, 1, y_res, 1)


        denominator = torch.where(y_grid - y_principal != 0, y_grid - y_principal, torch.tensor(1e-10))
        depth_map = (camera_height * focal_length) / denominator


        depth_map = depth_map.expand(-1, -1, -1, x_res)

        return depth_map.to(self.device)

    def generate_sky_pointcloud(self, syncdiffusion_model:SyncDiffusion=None, image=None, mask=None, gen_sky=False, style=None,sky_flow=False):
        image_height = 512
        image_width = 6144
        w_start = 256
        stride = 8
        anchor_view_idx = w_start // 8 // stride
        layers_panorama = 2
        num_inference_steps = 50
        guidance_scale = 7.5
        sync_weight = 80.0
        sync_decay_rate = 0.98
        sync_freq = 3
        sync_thres = 50

        example_name = self.config["example_name"]

        def linear_blend(images, overlap=100):


            alpha = np.linspace(0, 1, overlap).reshape(overlap, 1, 1)

            for i, img in enumerate(images):
                img_new = np.array(img)
                if i != 0:
                    overlap_img2 = img_new[512-overlap:, :, :]
                    top_img = img_new[:512-overlap, :, :]
                    blend_overlap = overlap_img1 * (1 - alpha) + overlap_img2 * alpha


                    blended_image = np.concatenate((top_img, blend_overlap, bottom_img), axis=0)
                    img_old = blended_image
                else:
                    img_old = img_new

                overlap_img1 = img_old[:overlap, :, :]
                bottom_img = img_old[overlap:, :, :]

            blended_image = (blended_image).astype(np.uint8)
            return Image.fromarray(blended_image)

        imgs = []
        gen_layer_0 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_0.png')) or gen_sky
        gen_layer_1 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_1.png')) or gen_layer_0 or gen_sky
        gen_layer_2 = (not os.path.exists(f'./examples/sky_images/{example_name}/sky_2.png')) or gen_layer_1 or gen_sky

        for layer in range(layers_panorama):
            if layer == 0:
                if gen_layer_0:
                    init_image = torch.zeros((1, 3, image_height, image_width))
                    init_image[:, :, :, w_start:w_start+image_height] = image
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_init_image.png")

                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, :, w_start:w_start+image_height] = 1-mask
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask.png")


                    mask_image_eroded = dilation(mask_image,
                                    kernel=torch.ones(10, 10).cuda()
                                    )
                    init_image = inpaint_cv2(init_image, mask_image_eroded)
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_inpainted_init_image.png")


                    mask_ = (mask_image[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    mask_block_size = 8
                    mask_ = skimage.measure.block_reduce(mask_, (mask_block_size, mask_block_size), np.min)
                    mask_ = mask_.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
                    mask_image = ToTensor()(mask_).unsqueeze(0).to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask_blocky.png")
                else:
                    img = Image.open(f'./examples/sky_images/{example_name}/sky_0.png')

                    imgs.append(img)
                    continue
            else:
                if gen_layer_1:
                    init_image = imgs[-1]
                    init_image = ToTensor()(init_image).unsqueeze(0).to(self.device)
                    toprows = init_image[:, :, :100, :]
                    remaining = init_image[:, :, 100:, :]
                    init_image = torch.cat((remaining, toprows), dim=-2)
                    ToPILImage()(init_image[0]).save(self.run_dir / f"{layer:02d}_init_image.png")

                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, -100:, :] = 0
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(self.run_dir / f"{layer:02d}_mask.png")
                else:
                    img = Image.open(f'./examples/sky_images/{example_name}/sky_1.png')

                    imgs.append(img)
                    continue

            print(f"[INFO] generating sky layer {layer} ...")
            prompts = f"sky, blue sky, horizon, distant hills. style: {style}" if layer == 0 else f"sky, blue sky, cloud. style: {style}"

            img = syncdiffusion_model.sample(
                prompts = prompts,
                negative_prompts = 'tree, text',
                height = image_height,
                width = image_width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                sync_weight = sync_weight,
                sync_decay_rate = sync_decay_rate,
                sync_freq = sync_freq,
                sync_thres = sync_thres,
                stride = stride,
                loop_closure = True,
                condition = True,
                inpaint_mask=mask_image,
                rendered_image=init_image,
                anchor_view_idx=anchor_view_idx,
            )


            new_img = ToTensor()(img).unsqueeze(0).to(self.device)
            mask_image_ = mask_image.expand(-1, 3, -1, -1).bool()
            loss = F.mse_loss(new_img[~mask_image_], init_image[~mask_image_]).cpu().item()
            print(f"[INFO] Sky Loss: {loss}")


            if layer == 0:
                new_img_ = torch.cat((new_img[:, :, :, w_start:], new_img[:, :, :, :w_start]), dim=-1)
                img = ToPILImage()(new_img_[0])
                img.save(self.run_dir / f"{layer:02d}_sky_leftmost.png")
            os.makedirs(f'./examples/sky_images/{example_name}', exist_ok=True)
            img.save(f'./examples/sky_images/{example_name}/sky_{layer}.png')
            imgs.append(img)
        img = linear_blend(imgs)


        image_height =  img.size[-1]
        equatorial_radius = 0.02

        camera_angle_x = 2*np.arctan(512 / (2*self.init_focal_length))
        min_latitude = -camera_angle_x / 2 - (image_height / 512 - 1) * camera_angle_x
        max_latitude = camera_angle_x / 2

        latitude = torch.linspace(min_latitude, max_latitude, image_height)
        longitude_offset = -camera_angle_x / 2
        longitude = torch.linspace(longitude_offset, longitude_offset + 2 * np.pi, image_width)

        lat, lon = torch.meshgrid(latitude, longitude, indexing='ij')


        x = -equatorial_radius * torch.cos(lat) * torch.sin(lon)
        z = equatorial_radius * torch.cos(lat) * torch.cos(lon)
        y = -equatorial_radius * torch.sin(lat)

        points = torch.stack((x, y, z), -1)


        points_flat = points.reshape(-1, 3)


        new_points_3d = points_flat.to(self.device)


        image_latest = ToTensor()(img).unsqueeze(0).to(self.device)
        colors = rearrange(image_latest, "b c h w -> (h w b) c")


        sky_rows_idx = torch.where(mask.any(dim=1))[0]
        max_idx = sky_rows_idx.max().item()
        ground_threshold = -0.0003 if max_idx <= 255 else -0.003
        mask_above_ground = new_points_3d[:, 1] >= ground_threshold
        new_points_3d = new_points_3d[mask_above_ground]
        colors = colors[mask_above_ground]


        if sky_flow:
            raise RuntimeError("sky_flow is included")
        else:
            motion_mask = torch.zeros(1, 1, 512, 512)
            new_motion_mask = motion_mask.float() * 255
            scene_flow = torch.zeros(1, 3, 512, 512)
            new_scene_flow = scene_flow.unsqueeze(0).unsqueeze(0).float() * 255

        self.update_current_pc(new_points_3d, colors,new_scene_flow,new_motion_mask, gen_sky=True)


        self.depth_latest[:] = self.sky_hard_depth
        self.disparity_latest[:] = 1. / self.sky_hard_depth
        self.depth_latest = self.depth_latest.to(self.device)
        self.disparity_latest = self.disparity_latest.to(self.device)


        image_height_down, image_width_down = int(image_height / 2), int(image_width / 2)
        img_down = img.resize((image_width_down, image_height_down), Image.Resampling.LANCZOS)
        latitude_down = torch.linspace(min_latitude, max_latitude, image_height_down)
        longitude_offset = -camera_angle_x / 2
        longitude_down = torch.linspace(longitude_offset, longitude_offset + 2 * np.pi, image_width_down)

        lat_down, lon_down = torch.meshgrid(latitude_down, longitude_down, indexing='ij')

        x_down = -equatorial_radius * torch.cos(lat_down) * torch.sin(lon_down)
        z_down = equatorial_radius * torch.cos(lat_down) * torch.cos(lon_down)
        y_down = -equatorial_radius * torch.sin(lat_down)

        points_down = torch.stack((x_down, y_down, z_down), -1)
        points_flat_down = points_down.reshape(-1, 3)
        new_points_3d_down = points_flat_down.to(self.device)

        image_latest_down = ToTensor()(img_down).unsqueeze(0).to(self.device)
        colors_down = rearrange(image_latest_down, "b c h w -> (h w b) c")

        mask_above_ground = new_points_3d_down[:, 1] >= ground_threshold
        new_points_3d_down = new_points_3d_down[mask_above_ground]
        colors_down = colors_down[mask_above_ground]

        if isinstance(img_down, Image.Image):
            W, H = img_down.size
        elif isinstance(img_down, torch.Tensor):
            H, W = img_down.shape[-2:]
        else:
            H, W = img_down.shape[:2]

        if sky_flow:
            raise RuntimeError("sky_flow is included")
        else:
            new_scene_flow_down = torch.zeros_like(new_points_3d_down)
            new_motion_mask_down = torch.zeros_like(colors_down)

        self.sky_pc_downsampled = {"xyz": new_points_3d_down, "rgb": colors_down, "motion_mask": new_motion_mask_down, "scene_flow": new_scene_flow_down}

        self.generate_sky_cameras()
        print('No using sky top for efficiency.')
        return

        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.init_focal_length
        K[0, 1, 1] = self.init_focal_length
        K[0, 0, 2] = 1280
        K[0, 1, 2] = 1280
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        new_camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((2560, 2560),), device=self.device)

        delta = -torch.tensor(torch.pi) / 2

        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, torch.cos(delta), -torch.sin(delta)], [0, torch.sin(delta), torch.cos(delta)]],
            device=self.device,
        )
        new_camera.R[0] = rotation_matrix @ new_camera.R[0]

        self.current_camera = new_camera

        render_output = self.render(render_sky=True, big_view=True)


        _, _, image_width, image_height = render_output["rendered_image"].shape

        if gen_layer_2:
            print(f"[INFO] generating sky layer 3 ...")
            img = syncdiffusion_model.sample(
                prompts = f"sky, blue sky, cloud. style: {style}",
                negative_prompts = 'tree, text',
                height = image_height,
                width = image_width,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                sync_weight=sync_weight,
                sync_decay_rate = sync_decay_rate,
                sync_freq = sync_freq,
                sync_thres = sync_thres,
                stride = stride,
                loop_closure = False,
                condition=True,
                inpaint_mask=render_output["inpaint_mask"],
                rendered_image=render_output["rendered_image"],
                anchor_view_idx=0,
            )
            os.makedirs(f'./sky_img/{example_name}', exist_ok=True)
            img.save(f'./sky_img/{example_name}/sky_2.png')
        else:
            img = Image.open(f'./sky_img/{example_name}/sky_2.png')


        radius = render_output["inpaint_mask"].sum(dim=-2).max().item() // 2
        center_x, center_y = image_width // 2, image_height // 2
        img = ToTensor()(img).unsqueeze(0).to(self.device)
        max_latitude = min_latitude
        min_latitude = -np.pi / 2

        points, colors = [], []


        points, colors = [], []

        i, j = torch.meshgrid(torch.arange(image_width), torch.arange(image_height), indexing='ij')

        dist = torch.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)

        mask = dist <= radius

        theta = min_latitude - (dist[mask] / radius) * (min_latitude - max_latitude)
        phi = torch.arctan2(i[mask] - center_x, j[mask] - center_y)

        x = -equatorial_radius * torch.cos(theta) * torch.cos(phi)
        y = -equatorial_radius * torch.sin(theta)
        z = equatorial_radius * torch.cos(theta) * torch.sin(phi)

        points = torch.stack([x, y, z], dim=1)
        colors = img[:, :, mask].permute(2, 0, 1).squeeze()
        points, colors = points.to(self.device), colors.to(self.device)
        self.update_current_pc(points, colors, gen_sky=True)

    @torch.no_grad()
    def get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0, big_view=False):
        view_matrix = torch.tensor(view_matrix, device=self.device, dtype=torch.float).reshape(4, 4)
        xy_negate_matrix = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=self.device, dtype=torch.float)
        view_matrix_negate_xy = view_matrix @ xy_negate_matrix
        R = view_matrix_negate_xy[:3, :3].unsqueeze(0)
        T = view_matrix_negate_xy[3, :3].unsqueeze(0)
        camera = self.get_camera_at_origin(big_view=big_view)
        camera.R = R
        camera.T = T / xyz_scale
        return camera

    @torch.no_grad()
    def update_sky_mask(self):
        sky_mask_latest, sem_seg = self.generate_sky_mask(self.image_latest, return_sem_seg=True)
        self.sky_mask_latest = sky_mask_latest[None, None, :]
        return sem_seg

    @torch.no_grad()
    def generate_sky_mask(self, input_image=None, return_sem_seg=False):
        if input_image is not None:
            image = ToPILImage()(input_image.squeeze())
        else:
            image = ToPILImage()(self.image_latest.squeeze())

        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                segment_output, target_sizes=[image.size[::-1]])[0]
        sky_mask = pred_semantic_map == 2
        if self.sky_erode_kernel_size > 0:
            sky_mask = erosion(sky_mask.float()[None, None],
                            kernel=torch.ones(self.sky_erode_kernel_size, self.sky_erode_kernel_size).to(self.device)
                            ).squeeze() > 0.5
        if return_sem_seg:
            return sky_mask, pred_semantic_map
        else:
            return sky_mask

    @torch.no_grad()
    def generate_ground_mask(self, sem_map=None, input_image=None):
        if sem_map is None:
            if input_image is not None:
                image = ToPILImage()(input_image.squeeze())
            else:
                image = ToPILImage()(self.image_latest.squeeze())

            segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
            segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                    segment_output, target_sizes=[image.size[::-1]])[0]
            sem_map = pred_semantic_map

        ground_mask = (sem_map == 3) | (sem_map == 6) | (sem_map == 9) | (sem_map == 11) | (sem_map == 13) | (sem_map == 26) | (sem_map == 29) | (sem_map == 46) | (sem_map == 128)
        if self.config['ground_erode_kernel_size'] > 0:
            ground_mask = erosion(ground_mask.float()[None, None],
                            kernel=torch.ones(self.config['ground_erode_kernel_size'], self.config['ground_erode_kernel_size']).to(self.device)
                            ).squeeze() > 0.5
        return ground_mask

    @torch.no_grad()
    def generate_grad_magnitude(self, disparity):
        vmin, vmax = disparity.min(), disparity.max()
        normalized_disparity = (disparity - vmin) / (vmax - vmin)
        cmap = plt.get_cmap('viridis')
        rgb_image = cmap(normalized_disparity)
        rgb_image = rgb_image[...,1]
        disparity = np.uint8(rgb_image * 255)

        ToPILImage()(disparity).save(self.run_dir / 'images' / 'disparity_gradient' / f'{self.kf_idx}_normalized_disparity.png')


        grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)


        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        threshold = 10
        mask = torch.from_numpy(grad_magnitude > threshold)
        return mask

    @torch.no_grad()
    def generate_layer(self, pred_semantic_map=None, scene_name=None):
        self.image_latest_init = copy.deepcopy(self.image_latest)
        self.depth_latest_init = copy.deepcopy(self.depth_latest)
        self.disparity_latest_init = copy.deepcopy(self.disparity_latest)
        if pred_semantic_map is None:
            image = ToPILImage()(self.image_latest.squeeze())

            segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
            segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                                    segment_output, target_sizes=[image.size[::-1]])[0]

        unique_elements = torch.unique(pred_semantic_map)
        masks = {str(element.item()): (pred_semantic_map == element) for element in unique_elements}


        disparity_np = self.disparity_latest.squeeze().cpu().numpy()
        grad_magnitude_mask = self.generate_grad_magnitude(disparity_np)
        mask_disocclusion = np.full((512, 512), False, dtype=bool)

        dilation_kernel=torch.ones(9, 9).to(self.device)
        for id, mask in masks.items():

            if id in ['3', '4', '6', '9', '11', '13', '26', '29', '46', '52', '128']:
                continue

            mask = dilation((mask).float()[None, None],
                            kernel=dilation_kernel).squeeze().cpu() > 0.5

            if id in ['49','76', '83', '87']:
                mask_disocclusion |= mask.numpy()
                continue
            labeled_array, num_features = label(mask)
            for i in range(1, num_features+1):

                mask_i = labeled_array==i
                disp_pixels = disparity_np[mask_i]
                disparity_mean = disp_pixels.mean()

                if disparity_mean < np.percentile(disparity_np, 60):
                    continue

                grad_magnitude_segment = grad_magnitude_mask[mask_i]

                if grad_magnitude_segment.float().mean() < 0.02:
                    continue

                segment_boundary = np.where(mask_i, grad_magnitude_mask, 0)
                if disparity_np[segment_boundary!=0].mean() > np.percentile(disp_pixels, 70):
                    continue

                if mask_i.mean() < 0.001:
                    continue

                mask_i_erosion = erosion(torch.from_numpy(mask_i).float()[None, None],
                            kernel=dilation_kernel.cpu()).squeeze() > 0.5
                disp_pixels = disparity_np[mask_i_erosion]
                p20 = np.percentile(disp_pixels, 20)
                p80 = np.percentile(disp_pixels, 80)
                if 1/p20 - 1/p80 > 0.0003 and mask_i.mean() > 0.05:
                    continue

                save_prompt = False

                mask_disocclusion |= mask_i

        inpainting_prompt = scene_name if scene_name is not None else 'road, building'
        print("Base layer inpainting_prompt: ", inpainting_prompt)
        mask_disocclusion = torch.from_numpy(mask_disocclusion)[None, None]


        self.mask_disocclusion = erosion(mask_disocclusion.float().to(self.device),
                                         kernel=dilation_kernel)
        inpaint_mask = self.mask_disocclusion > 0.5
        self.inpaint(self.image_latest, inpaint_mask=inpaint_mask, inpainting_prompt=inpainting_prompt, negative_prompt='tree, plant', mask_strategy=np.max, diffusion_steps=50)
        inpainter_output = self.image_latest

        stitch_mask = erosion(mask_disocclusion.float().to(self.device),
                            kernel=torch.ones(5, 5).to(self.device))
        self.image_latest = soft_stitching(inpainter_output, self.image_latest_init, stitch_mask, sigma=1, blur_size=3)
        ToPILImage()(grad_magnitude_mask.float()).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_grad_magnitude_mask.png')
        ToPILImage()((self.image_latest.cpu() * (~mask_disocclusion).float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_mask_disocclusion.png')
        ToPILImage()((self.image_latest_init * inpaint_mask.float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_inpaint_mask.png')
        ToPILImage()(self.image_latest_init[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_image_init.png')
        ToPILImage()(self.image_latest[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_remove_disocclusion.png')


    @torch.no_grad()
    def transform_all_cam_to_current_cam(self, center=False):

        if self.cameras != []:
            if not center:
                inv_current_camera_RT = self.cameras[-1].get_world_to_view_transform().inverse().get_matrix()
            else:
                inv_current_camera_RT = self.cameras[self.center_camera_idx].get_world_to_view_transform().inverse().get_matrix()

            for cam in self.cameras:
                cam_RT = cam.get_world_to_view_transform().get_matrix()
                new_cam_RT = inv_current_camera_RT @ cam_RT
                cam.R = new_cam_RT[:, :3, :3]
                cam.T = new_cam_RT[:, 3, :3]


    @torch.no_grad()
    def set_current_camera(self, camera, archive_camera=False):
        self.current_camera = camera
        if archive_camera:
            self.cameras_archive.append(copy.deepcopy(camera))

    @torch.no_grad()
    def generate_9_cameras(self, center_camera, distance = 0.00001):

        cameras = []
        cameras.append(center_camera)


        offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]

        for offset in offsets:

            new_camera = copy.deepcopy(center_camera)


            right = new_camera.R[0, :, 0]
            forward = new_camera.R[0, :, 1]
            delta_position = offset[0] * right + offset[1] * forward
            delta_position = delta_position / torch.norm(delta_position) * distance


            new_camera.T[0] = new_camera.T[0] + delta_position

            cameras.append(new_camera)

        return cameras

    @torch.no_grad()
    def set_cameras(self, rotation_path):
        move_left_count = 0
        move_right_count = 0
        for rotation in rotation_path:
            new_camera = copy.deepcopy(self.cameras[-1])

            if rotation == 0:
                forward_speed_multiplier = -1.0
                right_multiplier = 0
                camera_speed = self.camera_speed


                if move_left_count != 0 or move_right_count != 0:

                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0

            elif abs(rotation) == 2:

                if rotation > 0:
                    move_left_count += 1

                    if move_right_count != 0:
                        new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                        move_right_count = 0
                else:
                    move_right_count += 1

                    if move_left_count != 0:
                        new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                        move_left_count = 0

                forward_speed_multiplier = 0
                right_multiplier = 0
                camera_speed = 0
                theta = torch.tensor(self.rotation_range_theta * rotation / 2)
                rotation_matrix = torch.tensor(
                    [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

            elif abs(rotation) == 1:

                if move_left_count != 0 or move_right_count != 0:

                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0

                theta_frame = torch.tensor(self.rotation_range_theta / (self.interp_frames + 1)) * rotation
                sin = torch.sum(torch.stack([torch.sin(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                cos = torch.sum(torch.stack([torch.cos(i*theta_frame) for i in range(1, self.interp_frames+2)]))
                forward_speed_multiplier = -1.0 / (self.interp_frames + 1) * cos.item()
                right_multiplier = -1.0 / (self.interp_frames + 1) * sin.item()
                camera_speed = self.camera_speed * self.camera_speed_multiplier_rotation

                theta = torch.tensor(self.rotation_range_theta * rotation)
                rotation_matrix = torch.tensor(
                    [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

            elif rotation == 3:
                continue

            move_dir = torch.tensor([[-right_multiplier, 0.0, -forward_speed_multiplier]], device=self.device)


            new_camera.T += camera_speed * move_dir
            self.cameras.append(copy.deepcopy(new_camera))

        return new_camera

    @torch.no_grad()
    def generate_cameras(self, rotation_path):
        print("-- generating 360-degree cameras...")

        camera = self.get_camera_at_origin()
        self.cameras.append(copy.deepcopy(camera))
        self.scene_cameras_idx.append(len(self.cameras) - 1)
        self.transform_all_cam_to_current_cam()

        self.set_cameras(rotation_path)
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        print("-- generated 360-degree cameras!")

    @torch.no_grad()
    def generate_sky_cameras(self):
        print("-- generating sky cameras...")
        cameras_cache = copy.deepcopy(self.cameras)
        init_len = len(self.cameras)


        for i in tqdm(range(1)):
            delta = -torch.tensor(torch.pi) / (8) * (i + 1)
            for camera_id in range(init_len):
                self.center_camera_idx = camera_id
                self.transform_all_cam_to_current_cam(True)
                new_camera = copy.deepcopy(self.cameras[camera_id])

                rotation_matrix = torch.tensor(
                    [[1, 0, 0], [0, torch.cos(delta), -torch.sin(delta)], [0, torch.sin(delta), torch.cos(delta)]],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

                self.cameras.append(copy.deepcopy(new_camera))
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        self.sky_cameras = copy.deepcopy(self.cameras)
        self.cameras = cameras_cache
        print("-- generated sky cameras!")

    @torch.no_grad()
    def set_kf_param(self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt):
        super().set_frame_param(inpainting_resolution=inpainting_resolution,
                                inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)

    @torch.no_grad()
    def refine_disp_with_segments(self, save_intermediates=False, keep_threshold_disp_range=10, no_refine_mask=None, existing_mask=None, existing_disp=None):
        print('Refining disparity with segments...')
        if save_intermediates:
            (self.run_dir / 'refine_intermediates').mkdir(parents=True, exist_ok=True)
        image = ToPILImage()(self.image_latest.squeeze())
        image_np = np.array(image)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)
        min_mask_area = 100
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]

        if save_intermediates:
            save_sam_anns(masks, self.run_dir / 'refine_intermediates' / f"kf{self.kf_idx:02}_SAM.png")

        disparity_np = self.disparity_latest.squeeze().cpu().numpy()

        refined_disparity = refine_disp_with_segments_2(disparity_np, sorted_mask, keep_threshold=keep_threshold_disp_range, no_refine_mask=no_refine_mask,
                                                        existing_mask=existing_mask, existing_disp=existing_disp)

        if save_intermediates:
            save_depth_map(1/refined_disparity, self.run_dir / 'refine_intermediates' / f"kf{self.kf_idx:02}_p1_SAM")

        refined_depth = 1 / refined_disparity

        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)

        self.depth_latest[0, 0] = refined_depth
        self.disparity_latest[0, 0] = refined_disparity

        print('Refining done!')
        return refined_depth, refined_disparity

    @torch.no_grad()
    def generate_visible_pc(self):
        camera = self.current_camera
        raster_settings = PointsRasterizationSettings(
            image_size = 512,
            radius = 0.003,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        points, colors = self.get_combined_pc()["xyz"], self.get_combined_pc()["rgb"]
        point_cloud = Pointclouds(points=[points], features=[colors])
        images, fragment_idx = renderer(point_cloud, return_fragment_idx=True)
        fragment_idx = fragment_idx[..., :1]

        n_kf1_points = points.shape[0]
        fragment_idx = fragment_idx.reshape(-1)
        visible_points_idx = (fragment_idx < n_kf1_points) & (fragment_idx >= 0)
        fragment_idx = fragment_idx[visible_points_idx]

        if self.current_visible_pc is None:
            self.current_visible_pc = {"xyz": points[fragment_idx], "rgb": colors[fragment_idx]}
        else:
            self.current_visible_pc = {"xyz": torch.cat([self.current_visible_pc["xyz"], points[fragment_idx]], dim=0), "rgb": torch.cat([self.current_visible_pc["rgb"], colors[fragment_idx]], dim=0)}

    @torch.no_grad()
    def render(self, archive_output=False, camera=None, render_visible=False, render_sky=False, big_view=False, render_fg=False):
        camera = self.current_camera if camera is None else camera
        raster_settings = PointsRasterizationSettings(
            image_size = 1536 if big_view else 512,
            radius = 0.003,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        if render_visible:
            points, colors = self.current_visible_pc["xyz"], self.current_visible_pc["rgb"]
        elif render_sky:
            points, colors = self.current_pc_sky["xyz"], self.current_pc_sky["rgb"]
        elif render_fg:
            points, colors = self.current_pc["xyz"], self.current_pc["rgb"]
        else:
            points, colors = self.get_combined_pc()["xyz"], self.get_combined_pc()["rgb"]

        point_cloud = Pointclouds(points=[points], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        if archive_output:
            self.rendered_image_latest = rendered_image
            self.rendered_depth_latest = rendered_depth
            self.mask_latest = inpaint_mask

        return {
            "rendered_image": rendered_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": inpaint_mask,
        }

    @torch.no_grad()
    def archive_latest(self, idx=None):
        if idx is None:
            idx = self.kf_idx
        vmax = 0.006
        super().archive_latest(idx=idx, vmax=vmax)
        self.rendered_images.append(self.rendered_image_latest)
        self.rendered_depths.append(self.rendered_depth_latest)
        self.sky_mask_list.append(~self.sky_mask_latest.bool())


def save_point_cloud_as_ply(points, filename="output.ply", colors=None):

    assert points.dim() == 2 and points.size(1) == 3, "Input tensor should be of shape [N, 3]."

    if colors is not None:
        assert colors.dim() == 2 and colors.size(1) == 3, "Color tensor should be of shape [N, 3]."
        assert points.size(0) == colors.size(0), "Points and colors tensors should have the same number of entries."


    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.size(0)}",
        "property float x",
        "property float y",
        "property float z"
    ]


    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])

    header.append("end_header")


    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")

        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"


            if colors is not None:

                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"

            f.write(line + "\n")

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

    pt3d_to_kornia = torch.diag(torch.tensor([-1., -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)


def inpaint_cv2(rendered_image, mask_diff):
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype(np.uint8)
    mask_cv2 = mask_diff[0, 0].cpu().numpy()
    mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
    return inpainting.unsqueeze(0)
