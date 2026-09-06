# WonderPlay 当前 Pipeline

本文档只描述 `/root/autodl-tmp/EOWorld/Wonderplay` 当前代码中的实际运行流程。

## 1. 入口与运行目录

Stage 1 的入口是：

```bash
python WonderPlay_new/run_genesis.py --config examples/configs/venice.yaml --prefix example
```

## 2. Stage 1：基础场景生成

Stage 1 先构建单张图像对应的基础 3D 场景。当前 venice 配置使用：

```yaml
examples_dir: "examples/imgs/venice"
depth_model: marigold
gen_sky: False
gen_layer: False
skip_interp: True
imgto3D: False
motion_type: "environment"
```

流程入口是 `KeyframeGen`，主要完成以下内容：

```text
输入图像
→ 语义分割 / sky mask / ground mask
→ Marigold depth
→ Marigold normals
→ 深度修正与局部 inpaint
→ 从图像、深度、法线生成当前点云 current_pc_latest
```

`current_pc_latest` 是后续 3DGS 训练和环境运动绑定使用的点云字典，核心字段包括：

```text
xyz          # [N, 3]，点云三维位置
rgb          # [N, 3]，点颜色
normals      # [N, 3]，法线
scene_flow   # [N, 3]，每个点的三维运动向量
motion_mask  # [N, 1]，每个点是否参与环境运动
```

在普通静态点云生成时，`scene_flow` 默认为 0，`motion_mask` 默认为 False。环境运动分支会在 3DGS 训练前填充这两个字段。

## 3. 环境运动配置

当前环境运动由 `examples/configs/venice.yaml` 控制：

```yaml
motion_type: "environment"

environment_motion:
  scale_factor: 2.0
  sam_prompt: "water"
  fixed_hints:
    - [311, 361, 254, 403]
    - [369, 382, 285, 451]
    - [431, 401, 355, 472]
```

含义如下：

```text
sam_prompt   # 给 SAM3 的文本分割 prompt，用于得到运动区域 mask
fixed_hints  # 给 Cinemagraphy 的 2D 运动箭头，格式为 [start_x, start_y, end_x, end_y]
scale_factor # render_MLP 渲染阶段的运动强度缩放
```

`motion_type: "environment"` 时，`run_genesis.py` 会在基础 3DGS 训练前调用：

```python
prepare_environment_motion_fields(save_dir_sim, config)
```

`save_dir_sim` 对应：

```text
3d_result/wonderplay/venice/example/simulation
```

## 4. SAM3 运动区域分割

环境运动准备函数 `prepare_environment_motion_fields()` 的第一步是 SAM3 分割：

```text
sam_prompt
→ build_sam3_image_model()
→ Sam3Processor.set_image()
→ Sam3Processor.set_text_prompt()
→ 多个 prompt 的 mask 做 union
→ 得到 motion_mask_2d
```

如果 `sam_prompt` 中有逗号，会被拆成多个 prompt 分别分割，然后把所有 mask 合并。

输出文件：

```text
simulation/sam3_mask.png
```

`motion_mask_2d` 的分辨率按当前代码使用 512 x 512，对应输入图像和深度的工作分辨率。

## 5. Cinemagraphy 2D Motion Field

SAM3 mask 得到之后，环境运动准备函数会调用 Cinemagraphy 估计二维运动场：

```text
fixed_hints
→ make_final_hints_xy()
→ estimate_flow()
→ thirdparty/cinemagraphy/demo.py::eulerian_estimation()
→ flow_2d
```

`make_final_hints_xy()` 会把 YAML 中的：

```text
[[sx, sy, ex, ey], ...]
```

转换为 Cinemagraphy 接收的列向量格式：

```text
final_hint_start_x
final_hint_start_y
final_hint_end_x
final_hint_end_y
```

Cinemagraphy 使用的主要路径是：

```text
WonderPlay_new/thirdparty/cinemagraphy/config.yaml
WonderPlay_new/thirdparty/cinemagraphy/ckpts/model_150000.pth
```

`estimate_flow()` 的输入包括：

```text
image_pil       # 当前关键帧 RGB 图像
depth_latest    # 当前关键帧深度
motion_mask_2d  # SAM3 得到的运动区域
fixed_hints     # YAML 中写死的运动箭头
```

输出 `flow_2d` 是当前视角下的二维 motion field。

## 6. 2D Flow 到 3D Scene Flow

二维 motion field 生成后，代码通过 `KeyframeGen.attach_flow_to_current_pc_latest()` 将其绑定到当前点云：

```python
scene_flow, point_motion_mask = kf_gen.attach_flow_to_current_pc_latest(
    flow=flow_2d,
    motion_mask=motion_mask_2d,
    valid_mask=~kf_gen.sky_mask_latest,
    depth=kf_gen.depth_latest,
)
```

该函数的核心逻辑是：

```text
原始像素坐标 + depth
→ unproject 得到原始 3D 点

原始像素坐标 + 2D flow + depth
→ unproject 得到流动后的 3D 点

流动后的 3D 点 - 原始 3D 点
→ scene_flow
```

对应代码结构：

```python
new_points_3d = kf_camera.unproject(self.points, point_depth)
flow_points_3d = kf_camera.unproject(flow_points, point_depth)
scene_flow = flow_points_3d - new_points_3d
```

`valid_mask=~kf_gen.sky_mask_latest` 会把 sky 区域排除掉。最终写回：

```text
current_pc_latest["scene_flow"]
current_pc_latest["motion_mask"]
```

如果 `current_pc` 与 `current_pc_latest` 的点数一致，也会同步写入 `current_pc`。

## 7. 3DGS 训练数据构造

环境运动字段绑定完成后，`run_genesis.py` 再调用：

```python
traindata_layer = kf_gen.convert_to_3dgs_traindata_latest(
    xyz_scale=xyz_scale,
    use_no_loss_mask=False,
)
```

该函数会把当前点云转换成 3DGS 使用的训练数据：

```text
pcd_points       # [3, N]，点位置，乘以 xyz_scale
pcd_colors       # [N, 3]，点颜色
pcd_normals      # [N, 3]，点法线
pcd_scene_flow   # [3, N]，三维运动场，乘以 xyz_scale
pcd_motion_mask  # [N, 1]，运动点 mask
frames           # 训练相机和图像
camera_angle_x   # 相机水平视角
W, H             # 当前为 512 x 512
```

`scene/dataset_readers.py` 读取这些字段后构造 `BasicPointCloud`：

```python
BasicPointCloud(
    points=pcd_points,
    colors=pcd_colors,
    normals=pcd_normals,
    scene_flow=pcd_scene_flow,
    motion_mask=pcd_motion_mask,
)
```

`BasicPointCloud` 是从训练数据传入 `GaussianModel.create_from_pcd()` 的中间点云结构。它只保存点云属性，不负责训练和渲染。

## 8. GaussianModel 中的运动字段

`GaussianModel.create_from_pcd()` 会把 `BasicPointCloud` 中的运动字段写入 Gaussian：

```text
_scene_flow   # 当前 Gaussian 点的三维运动向量
_motion_mask  # 当前 Gaussian 点的运动 mask
```

对于有 previous Gaussian 的场景，代码会同时维护：

```text
_scene_flow_prev
_motion_mask_prev
```

渲染和 HashGrid 训练阶段使用的是拼接后的全量字段：

```python
gaussians.get_scene_flow_all
gaussians.get_motion_mask_all
gaussians.get_xyz_all
```

这些字段会跟随 densify、clone、split、prune、PLY 保存和 PLY 读取一起保持点数对齐。

PLY 中对应的字段包括：

```text
dx, dy, dz  # scene_flow
mx, my, mz  # motion_mask
```

## 9. 基础 Gaussian 训练

环境运动分支仍然会训练基础 3DGS 场景：

```text
sky_gaussians
→ base gaussians
→ Scene(traindata_layer, gaussians, opt)
→ train_gaussian()
```

在 `motion_type: "environment"` 下，基础 Gaussian 训练完成后进入环境运动渲染分支，不再执行 object 3DGS、InstantMesh 物体重建、Genesis 物理模拟。

环境分支会保存基础 Gaussian：

```text
3d_result/wonderplay/venice/example/3d_results/gaussians.ply
```

## 10. HashGrid 运动场训练

环境运动渲染函数是：

```python
environment_motion_rendering(
    gaussians,
    scene,
    save_dir_sim,
    config,
    video_gen_fps=8,
    sky_gaussians=sky_gaussians,
)
```

该函数先创建 LivingWorld 风格的 HashGrid 运动模型：

```python
from hashgrid import HashEncoderMotionModel
motion_model = HashEncoderMotionModel().to("cuda")
```

然后调用：

```python
train_hashgrid(gaussians, motion_model, iterations=100)
```

`train_hashgrid()` 使用的监督数据是：

```text
输入：gaussians.get_xyz_all
目标：gaussians.get_scene_flow_all
```

训练流程：

```text
Gaussian 3D 坐标
→ HashGrid encoding + MLP
→ 预测 pred_flow
→ 与 scene_flow 做 MSE 形式的监督
→ 得到连续的全局 3D motion field
```

训练时会根据点云范围设置：

```text
model.center
model.bound_xyz
```

并根据 LivingWorld 的参考速度均值设置：

```text
model.flow_scale
```

默认训练参数：

```text
iterations: 100
lr: 1e-2
scheduler: StepLR(step_size=100, gamma=0.2)
loss: ((pred_flow - flow_train) ** 2).sum()
```

## 11. LivingWorld Render MLP 渲染

动态环境渲染使用：

```python
gaussian_renderer/living_world_render.py::render_MLP()
```

输入包括：

```text
viewpoint_camera
gaussians
motion_model
t
opt
background
scale_factor
```

核心流程：

```text
读取 Gaussian 位置、颜色、不透明度、尺度、旋转、scene_flow、motion_mask
→ 根据 visibility/delete/is_sky 等 filter 取出可渲染点
→ 根据 motion_mask 取出运动点
→ 用 HashGrid MLP 查询运动点速度
→ pre_euler_integral() 做 forward/backward Euler 积分
→ 按时间 t/T 混合 forward/backward opacity
→ 将静态点和动态复制点一起送入 Gaussian rasterizer
→ 输出 RGB、depth、opacity 等 render package
```

`scale_factor` 来自：

```yaml
environment_motion.scale_factor
```

它在 `render_MLP()` 阶段影响运动幅度。

## 12. Stage 1 环境分支输出

环境分支的主要输出目录是：

```text
3d_result/wonderplay/venice/example/simulation
```

其中基础文件包括：

```text
simulation/gt.png
simulation/text_prompt.txt
simulation/sam3_mask.png
```

环境运动可视化输出：

```text
simulation/environment_motion/debug_static_frame_0000.png
simulation/environment_motion/frame_0000.png
simulation/environment_motion/frame_0001.png
...
simulation/environment_motion/environment_motion.mp4
```

Stage 2 使用的轨迹目录：

```text
simulation/traj_00
```

目录结构：

```text
simulation/traj_00/render_video.mp4
simulation/traj_00/render_video.gif
simulation/traj_00/render_depths.mp4
simulation/traj_00/render_flows.mp4
simulation/traj_00/render_flows_arrows.mp4

simulation/traj_00/frames/frame_00000000.png
simulation/traj_00/frames/frame_00000001.png
...

simulation/traj_00/masks/frame_00000000.png
simulation/traj_00/masks/frame_00000001.png
...

simulation/traj_00/depths/depth_00000000.png
simulation/traj_00/depths/depth_00000001.png
...

simulation/traj_00/flows_actual/flow_00000000.npy
simulation/traj_00/flows_actual/flow_00000001.npy
...

simulation/traj_00/flows/flow_00000000.png
simulation/traj_00/flows/flow_00000001.png
...

simulation/traj_00/flows_arrows/frame_00000000.png
simulation/traj_00/flows_arrows/frame_00000001.png
...
```

`flows_actual/*.npy` 使用 Stage 2 读取的格式：

```text
shape: [2, H, W]
range: [0, 1]
decode: flow = flow_actual * 2 * 512 - 512
```

环境分支的 `flows_actual` 由渲染后的相邻 RGB 帧通过 Farneback optical flow 计算：

```text
frame_00000000 → zero flow
frame_t-1 + frame_t → cv2.calcOpticalFlowFarneback()
→ [2, H, W]
→ np.clip((flow / 512 + 1) * 0.5, 0, 1)
```

## 13. Stage 2：视频模型细化

Stage 2 的入口是：

```bash
python WonderPlay_new/run_video_model.py \
  --input_folder 3d_result/wonderplay/venice/example/simulation \
  --output_folder <stage2_output_root> \
  --traj_id 0
```

`--input_folder` 指向 Stage 1 的 `simulation` 目录。代码会读取：

```text
input_folder/gt.png
input_folder/text_prompt.txt
input_folder/traj_00/render_video.mp4
input_folder/traj_00/flows_actual/*.npy
```

`run_video_model.py` 会从 `input_folder` 路径中解析：

```text
simulation_name = path_parts[-3]
simulation_id   = path_parts[-2]
traj_str        = traj_00
```

例如：

```text
input_folder = 3d_result/wonderplay/venice/example/simulation

simulation_name = venice
simulation_id   = example
traj_str        = traj_00
```

Stage 2 的中间输出目录是：

```text
<stage2_output_root>/venice/example/traj_00
```

### 13.1 Warp Noise

`main_warp_noise(args)` 会读取：

```text
traj_00/flows_actual/*.npy
```

读取后执行：

```text
flow_actual
→ flow_actual * 2 * 512 - 512
→ 取第 1 到第 48 个 flow
→ resize 到 720 x 720
→ 乘以 720 / 512
→ 按 crop_start 裁剪成 480 x 720
→ video_models.noise_warp.get_noise_from_video()
→ noises.npy
```

同时会把 Stage 1 的第一帧处理为视频模型输入图：

```text
input_folder/gt.png
→ resize 到 hold 480 x 720
→ crop_start:crop_start+480
→ <stage2_output_root>/venice/example/traj_00/venice.png
```

并复制输入视频：

```text
input_folder/traj_00/render_video.mp4
→ <stage2_output_root>/venice/example/traj_00/input.mp4
```

### 13.2 Cut-and-Drag / SDEdit

`main_cut_and_drag(args)` 会读取 prompt：

```text
--text_prompt 参数优先
否则读取 input_folder/text_prompt.txt
```

视频模型固定使用：

```text
model_name: I2V5B_final_i38800_nearest_lora_weights
guidance_scale: 6
noise_downtemp_interp: nearest
mask_strength: -1
```

默认参数：

```text
crop_start: 120
num_inference_steps: 25
degradation: 0.4
sdedit_strengths: [0.75, 0.8, 0.85, 0.9]
```

每个 `sdedit_strength` 的输出路径是：

```text
<stage2_output_root>/venice/example/traj_00/sdedit_0.750/without_mask/output.mp4
<stage2_output_root>/venice/example/traj_00/sdedit_0.800/without_mask/output.mp4
<stage2_output_root>/venice/example/traj_00/sdedit_0.850/without_mask/output.mp4
<stage2_output_root>/venice/example/traj_00/sdedit_0.900/without_mask/output.mp4
```

## 14. Object 分支

当配置为：

```yaml
motion_type: "object"
```

Stage 1 会继续执行物体分支：

```text
基础 3DGS 场景
→ object 3DGS
→ InstantMesh / image-to-3D object
→ Genesis physics simulation
→ simulation_efficient()
→ traj_00 输出
```

物体分支的 `traj_00` 输出同样包含：

```text
render_video.mp4
frames/
masks/
depths/
flows/
flows_actual/
flows_arrows/
```

其中 `flows_actual/*.npy` 来自 Gaussian rasterizer 输出的 optical flow，并使用与 Stage 2 匹配的归一化格式：

```text
shape: [2, H, W]
range: [0, 1]
decode: flow = flow_actual * 2 * 512 - 512
```
