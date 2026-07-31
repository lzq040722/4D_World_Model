import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
import threading
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from typing import Iterable, Union

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
from moge.model.v2 import MoGeModel

from models.models import KeyframeGen
from util.chatGPT4 import TextpromptGen
from util.utils import prepare_scheduler, soft_stitching
from util.utils import load_example_yaml, convert_pt3d_cam_to_3dgs_cam
from util.segment_utils import create_mask_generator_repvit

from arguments import GSParams
from gaussian_renderer import render, render_MLP
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from utils.trajectory import get_pcdGenPoses
from random import randint
import time
import cv2
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from kornia.morphology import dilation
import warnings
import os
import copy

from thirdparty.cinemagraphy.demo import eulerian_estimation
from torch.optim.lr_scheduler import StepLR


YZ_REVERSE = np.diag([1, -1, -1]).astype(np.float32)

import numpy as np
from hashgrid import HashEncoderMotionModel
from kornia.geometry import PinholeCamera


def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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

def estimate_flow(frame, mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y,  args):
    frame = {'image': frame, 'mask':mask, 'final_hint_start_x':final_hint_start_x, 'final_hint_start_y':final_hint_start_y, 'final_hint_end_x':final_hint_end_x, 'final_hint_end_y':final_hint_end_y}
    flow = eulerian_estimation(args, frame)
    return flow





def train_hashgrid(pc, model, scheduler_gamma=0.2, scheduler_step=100, iterations=100, lr=1e-2, device='cuda', freeze_mlp=False):
    means3D = pc.get_xyz_all
    scene_flow = pc.get_scene_flow_all

    assert means3D.shape[0] == scene_flow.shape[0]

    pos_all = means3D.detach().clone().to(device).requires_grad_(True)

    flow_all = scene_flow.detach().clone().to(device).requires_grad_(False)


    with torch.no_grad():
        if not (hasattr(model, "center") and hasattr(model, "bound_xyz")):
            pos_min = pos_all.min(0).values
            pos_max = pos_all.max(0).values
            center = 0.5 * (pos_min + pos_max)
            half_extent = 0.5 * (pos_max - pos_min)
            bound_xyz = half_extent.clamp_min(1e-6)

            model.center = center
            model.bound_xyz = bound_xyz
        else:
            pass


    model.train()

    pos_extent = pos_all.max(0).values - pos_all.min(0).values

    if not hasattr(model, "flow_scale"):
        current_mag = flow_all.norm(dim=1).mean()
        WW_VEL_MEAN = torch.tensor([
            0.0006032090168446302,
            6.930906238267198e-05,
            8.437281394435558e-06
        ], device=scene_flow.device)

        target_mean = WW_VEL_MEAN * 10.0
        target_mag = target_mean.norm()
        model.flow_scale = (target_mag / current_mag.clamp_min(1e-12)).item()
    else:
        pass

    flow_train = flow_all * model.flow_scale

    flow_mean = flow_all.mean(0)
    flow_std  = flow_all.std(0)
    flow_abs_mean = flow_all.abs().mean(0)
    flow_max = flow_all.abs().max(0).values

    flow_mean = flow_train.mean(0)
    flow_std  = flow_train.std(0)
    flow_abs_mean = flow_train.abs().mean(0)
    flow_max = flow_train.abs().max(0).values


    if freeze_mlp:
        for p in model.mlp.parameters():
            p.requires_grad = False


    else:
        for p in model.mlp.parameters():
            p.requires_grad = True


    for p in model.mlp.parameters():

        pass
    for p in model.encoder.parameters():

        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    for step in range(iterations):
        optimizer.zero_grad()
        pred = model(pos_all)
        loss = ((pred - flow_train) ** 2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == iterations - 1:
            pred = model(pos_all).detach()

            current_lr = scheduler.get_last_lr()[0]

    model.eval()
    return model




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




warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

view_matrix_fixed = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.2, 0.5, 1]
])
theta = np.radians(-3)
rotation_matrix_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
])
view_matrix_fixed = np.dot(view_matrix_fixed, rotation_matrix_x)
view_matrix_fixed = view_matrix_fixed.flatten().tolist()

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')
latest_frame = None
latest_frame_just = None
latest_viz = None
keep_rendering = True
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
undo = False
save = False
delete = False
exclude_sky = False


sam_enabled = False
sam_points2d = []
sam_pending = False

show_start_frame = True
start_frame_bgr = None


motion_model = None


clicks_buf_xy = []
clicks_on_ok_xy = []
scale_factor=10.0
scale_factor_just = 10000.0
capture = False
stop = False

sam_prompt = "water"
sam_prompt_lock = threading.Lock()


click_lock = threading.Lock()

def clicks_reset_for_new_loop():
    global clicks_buf_xy, clicks_on_ok_xy
    with click_lock:
        clicks_buf_xy = []
        clicks_on_ok_xy = []

def clicks_add(x, y, W=512, H=512, max_points=100, dedup_tol=4):
    global clicks_buf_xy
    x = max(0, min(W-1, int(round(x))))
    y = max(0, min(H-1, int(round(y))))
    with click_lock:
        for (px,py) in clicks_buf_xy:
            if abs(px-x) <= dedup_tol and abs(py-y) <= dedup_tol:
                return
        clicks_buf_xy.append((x,y))
        if len(clicks_buf_xy) > max_points:
            clicks_buf_xy = clicks_buf_xy[-max_points:]

def clicks_fix_on_ok():
    global clicks_on_ok_xy
    with click_lock:
        clicks_on_ok_xy = list(clicks_buf_xy)

def clicks_get_fixed():
    with click_lock:
        return list(clicks_on_ok_xy)

def points_reset_for_new_loop():
    global points_xy
    with click_lock:
        points_xy = np.empty((0, 2), dtype=int)


def split_points_and_hints(clicks):
    import numpy as np

    if len(clicks) == 0:
        return np.empty((0, 2), dtype=int), np.empty((4, 0), dtype=int)


    if len(clicks) % 2 == 1:
        points_xy = np.array([clicks[0]], dtype=int)
        tail = clicks[1:]
    else:
        points_xy = np.empty((0, 2), dtype=int)
        tail = clicks

    n = len(tail) // 2
    if n == 0:
        return points_xy, np.empty((4, 0), dtype=int)

    starts = np.array(tail[0:2*n:2], dtype=int)
    ends   = np.array(tail[1:2*n:2], dtype=int)

    hints = np.stack(
        [starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1]],
        axis=0
    )


    return points_xy, hints

ok_event = threading.Event()


start_event = threading.Event()
gen_event = threading.Event()

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

render_poses = get_pcdGenPoses('hemispherei')

internel_render_poses = get_pcdGenPoses('lookaround5')


def run(config):
    global client_id, view_matrix, scene_name, latest_frame,latest_frame_just, keep_rendering, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user, undo, save, delete, exclude_sky, view_matrix_delete
    global start_frame_bgr, motion_model, ax_len, show_start_frame, sam_prompt


    seeding(config["seed"])
    example = config['example_name']

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to('cuda')

    mask_generator = create_mask_generator_repvit()

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True,
        ).to(config["device"])
    inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())

    rotation_path = config['rotation_path'][:config['num_scenes']]


    assert len(rotation_path) == config['num_scenes']

    if config['depth_model']=='marigold':
        depth_model = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.bfloat16).to(config["device"])
        depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
        depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    elif config['depth_model']=='moge':
        depth_model = MoGeModel.from_pretrained(
            "Ruicheng/moge-2-vitl-normal"
        ).to(config["device"])
        depth_model.eval()

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v0-1",
        torch_dtype=torch.bfloat16,
        variant="fp16",
        use_safetensors=True,
    ).to(config["device"])

    kf_gen = KeyframeGen(config=config, inpainter_pipeline=inpainter_pipeline, mask_generator=mask_generator, depth_model=depth_model,
                            segment_model=segment_model, segment_processor=segment_processor, normal_estimator=normal_estimator,
                            rotation_path=rotation_path, inpainting_resolution=config['inpainting_resolution_gen']).to(config["device"])

    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text, outdoor = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None), yaml_data.get('outdoor', False)

    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    start_keyframe_path = Path(args.input_dir) / "start_keyframe.png"
    start_keyframe.save(start_keyframe_path)
    start_frame_bgr = np.array(start_keyframe, dtype=np.uint8)[..., ::-1].copy()

    if config['load_gen'] is False:

        socketio.emit('server-state', 'Waiting for OK...', room=client_id)
        kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config['device'])


        ok_event.wait()


        click_points = clicks_get_fixed()
        PASS_NO_MOTION = (click_points is None) or (len(click_points) == 0)

        if PASS_NO_MOTION:
            points_xy = np.empty((0, 2), dtype=int)
            hints     = np.empty((4, 0), dtype=int)
        else:
            points_xy, hints = split_points_and_hints(click_points)


        if config['gen_sky_image'] or (not os.path.exists(f'examples/sky_images/{example}/sky_0.png') and not os.path.exists(f'examples/sky_images/{example}/sky_1.png')):
            syncdiffusion_model = SyncDiffusion(config['device'], sd_version='2.0-inpaint')
        else:
            syncdiffusion_model = None

        sky_mask = kf_gen.generate_sky_mask().float()

        kf_gen.generate_sky_pointcloud(syncdiffusion_model, image=kf_gen.image_latest, mask=sky_mask, gen_sky=config['gen_sky_image'], style=style_prompt, sky_flow=False)
        kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name, args=args, points_xy=points_xy, hints=hints, sam_prompt=sam_prompt)
    pt_gen = TextpromptGen(kf_gen.run_dir, isinstance(control_text, list))

    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = content_prompt
    socketio.emit('scene-prompt', scene_name, room=client_id)


    kf_gen.increment_kf_idx()
    if config['load_gen'] is False:


        if config['gen_sky'] or not os.path.exists(f'examples/sky_images/{example}/finished_3dgs_sky_tanh2.ply'):
            traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
            if config['gen_layer']:
                traindata, traindata_sky, traindata_layer = traindatas
            else:
                traindata, traindata_sky = traindatas
            gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
            opt = GSParams()
            opt.max_screen_size = 100
            opt.scene_extent = 1.5
            opt.densify_from_iter = 200
            opt.prune_from_iter = 200
            opt.densify_grad_threshold = 1.0
            opt.iterations = 399
            scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_sky"
            train_gaussian(gaussians, scene, opt, save_dir, initialize_scaling=False)
            gaussians.save_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')
        else:
            gaussians = GaussianModel(sh_degree=0)
            gaussians.load_ply_with_filter(f'examples/sky_images/{example}/finished_3dgs_sky_tanh.ply')


        gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
        gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
        gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')

    if config['load_gen']:
        model_dir  = os.path.join(args.input_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(os.path.join(model_dir, 'finished_3dgs.ply'))
        gaussians.visibility_filter_all = torch.load(os.path.join(model_dir, 'visibility_filter_all.pth'), weights_only=False).to('cuda')
        gaussians.is_sky_filter = torch.load(os.path.join(model_dir, 'is_sky_filter.pth'), weights_only=False).to('cuda')
        gaussians.delete_mask_all = torch.load(os.path.join(model_dir, 'delete_mask_all.pth'), weights_only=False).to('cuda')

        motion_model = torch.load(os.path.join(model_dir, 'motion_model.pth'), weights_only=False).to('cuda')
        if motion_model:
            pass
        show_start_frame=False
    opt = GSParams()


    if config['load_gen'] is False:
        if config['gen_layer']:

            traindata, traindata_layer, flow_data = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{0:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)

        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)


        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        i = 0
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i:02d}"
        train_gaussian(gaussians, scene, opt, save_dir)
    i = 0


    if motion_model==None:
                motion_model = HashEncoderMotionModel().to('cuda')
    if config['load_gen'] is False:
        motion_model = train_hashgrid(gaussians, motion_model)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)


    def llm_prompt_generation(event):
        global scene_dict, style_prompt, pt_gen, change_scene_name_by_user, scene_name
        while True:
            event.wait()
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False
            event.clear()

    if config['use_gpt']:
        llm_event = threading.Event()
        llm_thread = threading.Thread(target=llm_prompt_generation, args=(llm_event, ))
        llm_thread.daemon = True
        llm_thread.start()


    gaussians_tmp = copy.deepcopy(gaussians)
    while True:
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        i += 1


        socketio.emit('scene-prompt', scene_name, room=client_id)
        socketio.emit('server-state', 'Waiting to generate new scenes...', room=client_id)

        while keep_rendering:
            time.sleep(0.05)
            if delete:
                current_pt3d_cam_delete = kf_gen.get_camera_by_js_view_matrix(view_matrix_delete, xyz_scale=xyz_scale)
                tdgs_cam_delete = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam_delete, xyz_scale=xyz_scale)
                gaussians.delete_points(tdgs_cam_delete)
                delete = False
            if save:
                model_dir  = os.path.join(args.input_dir, "model")
                os.makedirs(model_dir, exist_ok=True)
                gaussians.save_ply_all_with_filter(os.path.join(model_dir, 'finished_3dgs.ply'))
                torch.save(gaussians.visibility_filter_all, os.path.join(model_dir, 'visibility_filter_all.pth'))
                torch.save(gaussians.is_sky_filter, os.path.join(model_dir, 'is_sky_filter.pth'))
                torch.save(gaussians.delete_mask_all, os.path.join(model_dir, 'delete_mask_all.pth'))
                torch.save(kf_gen.flows_layer, os.path.join(model_dir, 'flows_layer.pth'))
                if motion_model:
                    torch.save(motion_model, os.path.join(model_dir, 'motion_model.pth'))
                gaussians.yield_splat_data(os.path.join(model_dir, f'{example}_finished_3dgs.splat'))
                save = False


        if undo:
            gaussians = copy.deepcopy(gaussians_tmp)
            undo = False
        else:
            gaussians_tmp = copy.deepcopy(gaussians)

        socketio.emit('server-state', 'Generating new scene...', room=client_id)


        if config['use_gpt']:
            llm_event.set()

        if config['use_gpt']:
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]


        kf_gen.set_kf_param(inpainting_resolution=config['inpainting_resolution_gen'],
                            inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)
        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)


        if exclude_sky:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)

            side_sky_height = 128


            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            mask_using_full_render[:, :, :side_sky_height, :] = 1

            mask_using_nosky_render = 1 - mask_using_full_render

            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render

            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

            exclude_sky = False
        else:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)

            side_sky_height = 128
            sky_cond_width = 40

            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)
            fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
            foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1)>150
            foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            if foreground_cols_idx.numel() > 0:
                min_index = foreground_cols_idx.min().item()
                max_index = foreground_cols_idx.max().item()
                mask_using_full_render[:, :, :, min_index:max_index+1] = 1
            mask_using_full_render[:, :, :sky_cond_width, :] = 1
            mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
            mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1

            mask_using_nosky_render = 1 - mask_using_full_render

            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render

            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

        kf_gen.inpaint(outpaint_condition_image, inpaint_mask=outpaint_mask, fill_mask=fill_mask, inpainting_prompt=inpainting_prompt, mask_strategy=np.max, diffusion_steps=50)

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)

        depth_should_be = render_pkg['median_depth'][0:1].unsqueeze(0) / xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)

        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)

        with torch.no_grad():
            depth_guide_joint, _ = kf_gen.get_depth(kf_gen.image_latest, target_depth=depth_should_be_joint, mask_align=joint_mask, archive_output=True,
                                                    diffusion_steps=30, guidance_steps=8)

        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        kf_gen.image_latest = recomposed

        clicks_reset_for_new_loop()
        points_reset_for_new_loop()
        img = kf_gen.image_latest[0].detach().cpu().clamp(0,1)
        img = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        start_frame_bgr = img[..., ::-1].copy()
        show_start_frame = True

        ok_event.clear()
        socketio.emit('server-state', 'Waiting for OK...', room=client_id)
        ok_event.wait()


        click_points = clicks_get_fixed()
        PASS_NO_MOTION = (click_points is None) or (len(click_points) == 0)

        if PASS_NO_MOTION:
            points_xy = np.empty((0, 2), dtype=int)
            hints     = np.empty((4, 0), dtype=int)
        else:
            points_xy, hints = split_points_and_hints(click_points)
        first=False
        H, W= 512, 512

        if config['gen_layer']:
            kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)


            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
            with torch.no_grad():
                kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(kf_gen.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth,
                                                                  diffusion_steps=30, guidance_steps=8)
            kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                                             existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
                                             existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy())
            wrong_depth_mask = kf_gen.depth_latest<kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            kf_gen.depth_latest = kf_gen.mask_disocclusion * kf_gen.depth_latest + (1-kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            kf_gen.update_sky_mask()
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)

            valid_px_mask = outpaint_mask.bool() & (~kf_gen.sky_mask_latest.bool())
            sky_mask = (~kf_gen.sky_mask_latest.bool())

            if PASS_NO_MOTION:

                background_mask   = None
                foreground_mask   = None
                background_flow   = None
                foreground_flow   = None
            else:

                kf_camera = convert_pytorch3d_kornia(kf_gen.current_camera, kf_gen.init_focal_length)


                if (hints is not None) and (
                        (isinstance(hints, np.ndarray) and hints.size > 0) or
                        (torch.is_tensor(hints) and hints.numel() > 0) or
                        (hasattr(hints, '__len__') and len(hints) > 0)
                    ):
                        new_hints = hints
                else :
                    new_hints = new_hints.detach().cpu().numpy()

                final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = make_final_hints_xy(new_hints, H, W)

                pil_image_latest = ToPILImage()(kf_gen.image_latest_init[0].detach().cpu().clamp(0., 1.))
                image_for_mask = np.asarray(pil_image_latest.convert("RGB"), dtype=np.uint8)


                if new_hints is None or len(new_hints) == 0:
                    mask = None
                else:
                    mask, bbox, _ = sam3_segment_all_union_after(
                        image_for_mask,
                        prompt=sam_prompt,
                        split_commas=True,
                    )


                mask_dis = kf_gen.mask_disocclusion
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

                IOU_THRESHOLD = 0.7

                if iou > IOU_THRESHOLD:
                    foreground_mask = sam_np & mask_dis_np
                    background_mask = None
                else:
                    background_mask = sam_np
                    foreground_mask = None


                if background_mask is not None:
                    background_flow=estimate_flow(pil_image_latest, background_mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)
                    fore=False
                else:
                    background_flow=None

                if foreground_mask is not None:
                    foreground_flow=estimate_flow(pil_image_latest, foreground_mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)
                    fore=True
                else:
                    foreground_flow=None

            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask, sky_mask=sky_mask, flow=background_flow, motion_mask=background_mask, first=first, fore=fore, back=True)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest_init, depth=kf_gen.depth_latest_init, valid_mask=kf_gen.mask_disocclusion*outpaint_mask, flow=foreground_flow, motion_mask=foreground_mask, hints=hints, gen_layer=True, fore=fore, back=False, sky_mask=sky_mask)
        else:
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)

            if PASS_NO_MOTION:
                new_hints = None
                flow   = None
                mask   = None
            else:
                pil_image_latest = ToPILImage()(kf_gen.image_latest[0].detach().cpu().clamp(0., 1.))
                image_for_flow = np.asarray(pil_image_latest.convert("RGB"), dtype=np.uint8)

                kf_camera = convert_pytorch3d_kornia(kf_gen.current_camera, kf_gen.init_focal_length)


                if (hints is not None) and (
                        (isinstance(hints, np.ndarray) and hints.size > 0) or
                        (torch.is_tensor(hints) and hints.numel() > 0) or
                        (hasattr(hints, '__len__') and len(hints) > 0)
                    ):
                        new_hints = hints
                else :
                    new_hints = new_hints.detach().cpu().numpy()

                final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = make_final_hints_xy(new_hints, H, W)

                if new_hints is None or len(new_hints) == 0:
                    mask = None
                    flow = None
                else:
                    mask, bbox, _ = sam3_segment_all_union_after(
                        image_for_flow,
                        prompt=sam_prompt,
                        split_commas=True,
                    )
                    flow=estimate_flow(pil_image_latest, mask, final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y, args)

            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask, flow=flow, motion_mask=mask, hints=new_hints)
        kf_gen.archive_latest()

        if config['gen_layer']:
            traindata, traindata_layer, flow_data = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{i+1:02d}"
            train_gaussian(gaussians, scene, opt, save_dir)


        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

        if traindata['pcd_points'].shape[-1] == 0:

            motion_model = HashEncoderMotionModel().to('cuda')

            motion_model = train_hashgrid(gaussians, motion_model)
            gaussians.set_inscreen_points_to_visible(tdgs_cam)

            kf_gen.increment_kf_idx()
            keep_rendering = True
            continue

        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        x = torch.sum(fg_mask_0p5_nosky == 1, dim=2)>0
        x_idx = torch.nonzero(x, as_tuple=True)[1]
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :x_idx.max().item(), min_index:max_index+1] = 1


        mask_using_nosky_render = 1 - mask_using_full_render
        image_tmp = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render


        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i+1:02d}"
        train_gaussian(gaussians, scene, opt, save_dir)


        if motion_model==None:
            motion_model = HashEncoderMotionModel().to('cuda')

        motion_model = train_hashgrid(gaussians, motion_model, freeze_mlp=True)

        gaussians.set_inscreen_points_to_visible(tdgs_cam)


        kf_gen.increment_kf_idx()
        keep_rendering = True
        empty_cache()


def train_gaussian(gaussians: GaussianModel, scene: Scene, opt: GSParams, save_dir: Path, initialize_scaling=True):
    global latest_frame, iter_number, view_matrix, latest_viz, latest_frame_just
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)


    for iteration in iterable_gauss:

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])


        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration == opt.iterations:

            time.sleep(0.1)
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                image = render_pkg['render']


            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image
        loss.backward()
        if iteration == opt.iterations:
            pass


        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

        with torch.no_grad():


            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)


                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                    gaussians.compute_3D_filter(cameras=trainCameras)


            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

def start_server(port):
    socketio.run(app, host='0.0.0.0', port=port)

@socketio.on('connect')
def handle_connect():
    global client_id
    client_id = request.sid

@socketio.on('disconnect')
def handle_disconnect():
    global client_id
    client_id = None

@socketio.on('start')
def handle_start(data):
    start_event.set()

@socketio.on('gen')
def handle_gen(data):
    global view_matrix, keep_rendering
    keep_rendering = False
    view_matrix = data

@socketio.on('render-pose')
def handle_render_pose(data):
    global view_matrix_wonder, keep_rendering
    view_matrix_wonder = data

@socketio.on('scene-prompt')
def handle_new_prompt(data):
    assert isinstance(data, str)
    global scene_name, change_scene_name_by_user
    scene_name = data
    change_scene_name_by_user = True

@socketio.on('undo')
def handle_undo():
    global undo
    undo = True

@socketio.on('save')
def handle_save():
    global save
    save = True

@socketio.on('capture')
def handle_save():
    global capture
    capture = True

@socketio.on('stop')
def handle_save():
    global stop
    stop = True

@socketio.on('delete')
def handle_delete(data):
    global delete, view_matrix_delete
    delete = True
    view_matrix_delete = data

@socketio.on('fill_hole')
def handle_fill_hole():
    global exclude_sky
    exclude_sky = True


@socketio.on('sam-toggle')
def on_sam_toggle(msg):
    global sam_enabled
    sam_enabled = bool(msg.get('enabled', False))

    emit('server-state', f"SAM {'ON' if sam_enabled else 'OFF'}", room=request.sid)

@socketio.on('sam-click')
def on_sam_click(msg):
    uv   = msg.get('uv')
    xy   = msg.get('xy')
    size = msg.get('size')
    if size is None:
        W = H = 512
    else:
        W, H = int(size[0]), int(size[1])

    if xy is not None:
        x, y = int(xy[0]), int(xy[1])
    else:

        u, v = float(uv[0]), float(uv[1])
        x, y = int(round(u*(W-1))), int(round(v*(H-1)))

    clicks_add(x, y, W=W, H=H)

    return {'ok': True}

@socketio.on('ok-start')
def handle_ok_start():
    global show_start_frame
    clicks_fix_on_ok()
    show_start_frame = False
    ok_event.set()
    emit('server-state', "OK clicked. Rendering enabled.", room=request.sid)

@socketio.on('set-scale')
def on_set_scale(msg):
    global scale_factor
    try:
        v = float(msg.get('value', scale_factor))
        v = max(0.0, min(v, 1e6))
        scale_factor = v
        emit('scale-state', {'value': scale_factor}, room=request.sid)
        emit('server-state', f'scale_factor = {scale_factor}', room=request.sid)
    except Exception as e:
        emit('server-state', f'scale set failed: {e}', room=request.sid)

@socketio.on('set-sam-prompt')
def on_set_sam_prompt(msg):
    global sam_prompt
    try:
        v = msg.get('value', sam_prompt)
        sam_prompt = str(v)


        emit('server-state', f"sam_prompt = '{sam_prompt}'", room=request.sid)

    except Exception as e:
        emit('server-state', f'sam prompt set failed: {e}', room=request.sid)


@socketio.on('connect')
def handle_connect():
    global client_id
    client_id = request.sid
    emit('scale-state', {'value': scale_factor}, room=request.sid)
    emit('scale-state2', {'value': scale_factor_just}, room=request.sid)


def render_current_scene():
    global latest_frame,latest_frame_just, client_id, iter_number, latest_viz, kf_gen, gaussians, opt, background, view_matrix_wonder, save
    global sam_enabled, sam_points2d, sam_pending
    global show_start_frame, start_frame_bgr, view_matrix_fixed, xyz_scale, motion_model, ax_len, scale_factor, capture, scale_factor_just, stop

    fnum=0
    start_ts = None
    capture_frames = []
    normal_dir  = os.path.join(args.input_dir, "normal")
    video_dir  = os.path.join(args.input_dir, "video")
    normal_flow_dir  = os.path.join(args.input_dir, "normal_flow")
    birdeye_dir = os.path.join(args.input_dir, "birdeye")
    birdeye_flow_dir = os.path.join(args.input_dir, "birdeye_flow")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(normal_flow_dir, exist_ok=True)
    os.makedirs(birdeye_dir, exist_ok=True)
    os.makedirs(birdeye_flow_dir, exist_ok=True)
    while True:
        time.sleep(0.03)
        try:
            fnum+=1
            if fnum==100:
                fnum=0
            if show_start_frame:

                if start_frame_bgr is not None:
                    latest_frame = start_frame_bgr

                latest_viz = None

            else:

                with torch.no_grad():
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                        kf_gen.get_camera_by_js_view_matrix(view_matrix_wonder, xyz_scale=xyz_scale),
                        xyz_scale=xyz_scale
                    )

                    render_pkg_flow = render(tdgs_cam, gaussians, opt, background, flow_render=True, render_visible=True)


                    render_pkg = render_MLP(tdgs_cam, gaussians, motion_model, fnum, opt, background, render_visible=True, scale_factor=scale_factor)

                rendered_img = render_pkg['render']


                rendered_img_flow = render_pkg_flow['render']


                rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
                rendered_image = (rendered_image * 255).astype(np.uint8)[..., ::-1]
                latest_frame = rendered_image


                rendered_img_flow = rendered_img_flow.permute(1, 2, 0).detach().cpu().numpy()
                rendered_img_flow = (rendered_img_flow * 255).astype(np.uint8)[..., ::-1]
                latest_frame_flow = rendered_img_flow


                with torch.no_grad():
                    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(
                        kf_gen.get_camera_by_js_view_matrix(view_matrix_fixed, xyz_scale=xyz_scale, big_view=True),
                        xyz_scale=xyz_scale
                    )
                    tdgs_cam.image_width = 1536
                    render_pkg_flow = render(tdgs_cam, gaussians, opt, background, flow_render=True, render_visible=True)

                    render_pkg = render_MLP(tdgs_cam, gaussians,motion_model,fnum, opt, background, render_visible=True, scale_factor=scale_factor)


                rendered_img = render_pkg['render']


                rendered_img_flow = render_pkg_flow['render']


                rendered_image = rendered_img.permute(1, 2, 0).detach().cpu().numpy()
                rendered_image = (rendered_image * 255).astype(np.uint8)[..., ::-1]
                latest_viz = rendered_image


                rendered_img_flow = rendered_img_flow.permute(1, 2, 0).detach().cpu().numpy()
                rendered_img_flow = (rendered_img_flow * 255).astype(np.uint8)[..., ::-1]
                latest_viz_flow = rendered_img_flow


                if capture and start_ts is None:
                    start_ts = now_ts()
                    capture_frames = []

                if capture:
                    socketio.emit('server-state', 'Capture image...', room=client_id)

                    ts = now_ts()

                    frame_path = os.path.join(normal_dir, f"{ts}_frame.jpg")
                    flow_path = os.path.join(normal_flow_dir, f"{ts}_flow.jpg")
                    viz_path = os.path.join(birdeye_dir, f"{ts}_viz.jpg")
                    viz_flow_path = os.path.join(birdeye_flow_dir, f"{ts}_flow.jpg")

                    cv2.imwrite(frame_path, latest_frame)
                    cv2.imwrite(flow_path, latest_frame_flow)
                    cv2.imwrite(viz_path, latest_viz)
                    cv2.imwrite(viz_flow_path, latest_viz_flow)

                    capture_frames.append(frame_path)

                if stop and start_ts is not None:
                    capture = False
                    stop = False

                    end_ts = now_ts()

                    if len(capture_frames) > 0:
                        first = cv2.imread(capture_frames[0])
                        h, w = first.shape[:2]

                        video_path = os.path.join(video_dir, f"{end_ts}.mp4")

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video = cv2.VideoWriter(video_path, fourcc, 20, (w, h))

                        for f in capture_frames:
                            img = cv2.imread(f)
                            if img is not None:
                                video.write(img)

                        video.release()

                    start_ts = None
                    capture_frames = []


                if save:
                    ToPILImage()(rendered_img).save(kf_gen.run_dir / 'rendered_img.png')

        except Exception as e:
            pass


        if latest_frame is not None and client_id is not None:


            image_bytes = cv2.imencode('.jpg', latest_frame)[1].tobytes()
            socketio.emit('frame', image_bytes, room=client_id)
            socketio.emit('iter-number', f'Iter: {iter_number}', room=client_id)


        if latest_viz is not None and client_id is not None:


            image_bytes = cv2.imencode('.jpg', latest_viz)[1].tobytes()
            socketio.emit('viz', image_bytes, room=client_id)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config"
    )
    parser.add_argument(
        "--port",
        default=7777,
        type=int,
        help="Port for the server",
    )
    parser.add_argument('--input_dir', type=str, help='input folder that contains src images', required=False)
    parser.add_argument('--train_iteration', type=int, default=50)
    parser.add_argument('-c', '--config', type=str, default='thirdparty/cinemagraphy/config.yaml', help='config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--cinema_ckpt", type=str, default='thirdparty/cinemagraphy/ckpts',help='specific weights file to reload')

    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--no_load_opt", action='store_true', help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true', help='do not load scheduler when reloading')
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)


    server_thread = threading.Thread(target=start_server, args=(args.port,))
    server_thread.start()


    render_thread = threading.Thread(target=render_current_scene)
    render_thread.start()

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            import ipdb
            ipdb.post_mortem()
    else:
        run(config)
