# EOWorld: WonderPlay + LivingWorld 最小改动集成方案

## 核心原则

**最小改动，完全复用 LivingWorld 的逻辑**

### ✅ 完全保留 LivingWorld 的实现
1. ✅ SAM3 分割（或用 WonderPlay 现有的 RepViT-SAM）
2. ✅ Eulerian flow estimation
3. ✅ HashGrid + MLP 训练
4. ✅ render_MLP 渲染
5. ✅ train_hashgrid 完整逻辑

### ✅ 唯一的修改：写死参数
- **sam_prompt**: `"water"` （写死在代码中）
- **scale_factor**: `2.0` （写死在代码中）
- **不需要前端交互**：从 config 读取固定的 motion hints

---

## 项目结构

```
/root/autodl-tmp/EOWorld/Wonderplay/
├── WonderPlay_new/
│   ├── hashgrid.py                  # 从 LivingWorld 复制
│   ├── run_genesis.py               # 修改：添加环境运动分支
│   ├── gaussian_renderer/
│   │   └── __init__.py              # 修改：添加 render_MLP
│   └── thirdparty/
│       └── cinemagraphy/            # 从 LivingWorld 复制（Eulerian flow）
└── examples/configs/
    └── venice.yaml                  # 修改：添加 motion_type
```

---

## 需要复制的文件

### 1. 核心文件
```bash
# HashGrid 模型
cp /root/autodl-tmp/LivingWorld/hashgrid.py \
   /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/

# Cinemagraphy (Eulerian flow)
cp -r /root/autodl-tmp/LivingWorld/thirdparty/cinemagraphy \
      /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/thirdparty/

# flow_viz (光流可视化)
cp /root/autodl-tmp/LivingWorld/flow_viz.py \
   /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/
```

### 2. 从 LivingWorld/run.py 复制的函数

在 `run_genesis.py` 中添加：

```python
# ===== 从 LivingWorld/run.py 复制 =====

# 1. 运动 hints 处理 (line 75-105)
def make_final_hints_xy(hints, H, W, yflip=False, as_column_list=True, dtype=np.float32):
    hints = np.asarray(hints)
    assert hints.ndim == 2 and hints.shape[0] == 4
    # ... 完整实现
    return to_col_list(sx), to_col_list(sy), to_col_list(ex), to_col_list(ey)

# 2. Eulerian flow 估计 (line 107-131)
def estimate_flow(frame, depth, mask, final_hint_start_x, final_hint_start_y, 
                  final_hint_end_x, final_hint_end_y, args):
    frame = {
        'image': frame,
        'depth': depth,
        'mask': mask,
        'final_hint_start_x': final_hint_start_x,
        # ...
    }
    from thirdparty.cinemagraphy.demo import eulerian_estimation
    flow = eulerian_estimation(args, frame)
    return flow

# 3. HashGrid 训练 (line 137-227)
def train_hashgrid(pc, model, scheduler_gamma=0.2, scheduler_step=100, 
                   iterations=100, lr=1e-2, device='cuda', freeze_mlp=False):
    means3D = pc.get_xyz_all
    scene_flow = pc.get_scene_flow_all
    
    # center 和 bound_xyz 计算
    if not (hasattr(model, "center") and hasattr(model, "bound_xyz")):
        pos_min = pos_all.min(0).values
        pos_max = pos_all.max(0).values
        center = 0.5 * (pos_min + pos_max)
        half_extent = 0.5 * (pos_max - pos_min)
        bound_xyz = half_extent.clamp_min(1e-6)
        model.center = center
        model.bound_xyz = bound_xyz
    
    # flow_scale 归一化
    if not hasattr(model, "flow_scale"):
        current_mag = flow_all.norm(dim=1).mean()
        WW_VEL_MEAN = torch.tensor([0.0006032, 6.9309e-05, 8.4372e-06])
        target_mag = (WW_VEL_MEAN * 10.0).norm()
        model.flow_scale = (target_mag / current_mag.clamp_min(1e-12)).item()
    
    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    for step in range(iterations):
        # ... 完整训练逻辑
    
    model.eval()
    return model
```

### 3. 从 LivingWorld/gaussian_renderer/__init__.py 复制

```python
# ===== 添加到 WonderPlay_new/gaussian_renderer/__init__.py =====

# 1. flow_to_rgb (line 24-49)
def flow_to_rgb(flow, clip=None):
    u, v = flow[...,0], flow[...,1]
    mag = torch.sqrt(u*u + v*v)
    ang = torch.atan2(v, u)
    # ... HSV to RGB 转换
    return rgb

# 2. pre_euler_integral (line 52-74)
def pre_euler_integral(xyz, model, T, smooth):
    f = torch.empty((T, *xyz.shape), device=xyz.device)
    b = torch.empty((T, *xyz.shape), device=xyz.device)
    f[0] = b[0] = xyz
    
    with torch.no_grad():
        for i in range(1, T):
            v = model(f[i-1])
            f[i] = f[i-1] + v * smooth
            # ... backward 积分
    return f, b

# 3. render_MLP (line 87-300+)
def render_MLP(viewpoint_camera, pc, motion_model, t, opt, bg_color, 
               scaling_modifier=1.0, override_color=None, 
               render_visible=False, exclude_sky=False, 
               scale_factor=2.0):
    # 完整的 MLP 渲染逻辑
    # 包括缓存、Euler 积分、双向渲染等
    pass
```

---

## 修改 run_genesis.py 主循环

在模拟循环中添加环境运动分支：

```python
# ========== 在物理模拟之后添加 ==========
motion_type = config.get('motion_type', 'object')

if motion_type == "environment":
    print("========== Environment Motion (LivingWorld) ==========")
    
    # 1. 写死的参数
    SAM_PROMPT = "water"
    SCALE_FACTOR = 2.0
    
    # 2. 从 config 读取固定的 motion hints
    # 格式: [[start_x, start_y, end_x, end_y], ...]
    fixed_hints = config['environment_motion'].get('fixed_hints', [])
    
    if len(fixed_hints) > 0:
        hints = np.array(fixed_hints).T  # 转为 [4, N]
        H, W = 512, 512
        
        # 处理 hints
        final_hint_start_x, final_hint_start_y, final_hint_end_x, final_hint_end_y = \
            make_final_hints_xy(hints, H, W)
        
        # 获取图像和深度
        pil_image_latest = ToPILImage()(kf_gen.image_latest[0].detach().cpu().clamp(0., 1.))
        image_for_mask = np.asarray(pil_image_latest.convert("RGB"), dtype=np.uint8)
        
        # 使用 SAM 分割（使用现有的 mask_generator 或 SAM3）
        # 这里假设使用 WonderPlay 现有的 mask_generator
        # 如果需要 SAM3，可以从 LivingWorld 复制
        mask = segment_region(image_for_mask, SAM_PROMPT, mask_generator)
        
        # Eulerian flow 估计
        flow = estimate_flow(
            pil_image_latest,
            kf_gen.depth_latest,
            mask,
            final_hint_start_x, final_hint_start_y,
            final_hint_end_x, final_hint_end_y,
            args
        )
        
        # 更新 Gaussian 的 scene_flow
        # 从 2D flow 投影到 3D（需要实现或从 LivingWorld 复制）
        update_gaussian_scene_flow(gaussians, flow, mask, camera)
    else:
        print("Warning: No motion hints provided, using zero flow")
        # 设置零 flow
        gaussians._scene_flow_all = torch.zeros_like(gaussians.get_xyz_all)
    
    # 设置 motion_mask（所有点都参与运动）
    gaussians._motion_mask_all = torch.ones(
        gaussians.get_xyz_all.shape[0], 1, 
        dtype=torch.bool, device='cuda'
    )
    
    # 训练 HashGrid
    from WonderPlay_new.hashgrid import HashEncoderMotionModel
    motion_model = HashEncoderMotionModel().to('cuda')
    motion_model = train_hashgrid(gaussians, motion_model, iterations=100)
    
    # 渲染循环
    for frame_idx in range(num_frames):
        render_pkg = render_MLP(
            viewpoint_camera=camera,
            pc=gaussians,
            motion_model=motion_model,
            t=frame_idx,
            opt=opt,
            bg_color=background,
            scale_factor=SCALE_FACTOR,
            render_visible=True
        )
        # 保存或显示 render_pkg['render']
        save_frame(render_pkg['render'], frame_idx)

else:
    # ===== 保持原有的物理模拟分支不变 =====
    print("========== Object Motion (WonderPlay Genesis) ==========")
    # ... 原有的 Genesis 物理模拟代码
```

---

## 修改配置文件

`examples/configs/venice.yaml`:

```yaml
# 在文件末尾添加
motion_type: "environment"  # "object" 或 "environment"

# 环境运动参数（仅当 motion_type="environment" 时使用）
environment_motion:
  scale_factor: 2.0
  sam_prompt: "water"  # 实际上写死在代码中了
  # 固定的运动方向 hints
  # 格式: [[start_x, start_y, end_x, end_y], ...]
  fixed_hints:
    - [100, 200, 150, 250]  # 示例：从 (100,200) 到 (150,250)
    - [200, 200, 250, 250]  # 可以添加多个
```

---

## 安装依赖

```bash
pip install tinycudann
```

---

## 测试步骤

### 1. 测试物体运动（验证不破坏原有功能）
```bash
cd /root/autodl-tmp/EOWorld/Wonderplay

# 修改 venice.yaml: motion_type: "object"
python WonderPlay_new/run_genesis.py \
  --config examples/configs/venice.yaml \
  --prefix test_object
```
**预期**: 船按原来的方式运动（Genesis 物理模拟）

### 2. 测试环境运动（验证新功能）
```bash
# 修改 venice.yaml: motion_type: "environment"
python WonderPlay_new/run_genesis.py \
  --config examples/configs/venice.yaml \
  --prefix test_environment
```
**预期**: 水面有波动效果

---

## 关键要点

1. ✅ **完全复用 LivingWorld 代码**：不做简化，不自己创新
2. ✅ **写死参数**：sam_prompt="water", scale_factor=2.0
3. ✅ **最小改动**：只添加分支逻辑，不修改原有物理模拟
4. ✅ **保持兼容**：scene_flow 和 motion_mask 使用 LivingWorld 的方式

---

## 估计工作量

- 复制文件: 15分钟
- 复制函数: 30分钟
- 添加分支逻辑: 30分钟
- 测试调试: 1小时
- **总计**: 约2小时
