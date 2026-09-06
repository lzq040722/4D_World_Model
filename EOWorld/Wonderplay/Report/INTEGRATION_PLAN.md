# EOWorld: WonderPlay + LivingWorld 集成方案 (最小改动)

## 更新说明 (2024-08-06)
- ✅ 新建独立项目 EOWorld (`/root/autodl-tmp/EOWorld/Wonderplay/`)
- ✅ 保持原 WonderPlay 和 LivingWorld 代码不变
- ✅ 添加 SAM3 分割细节说明
- ✅ 完整的运动 mask 生成流程

## 目标
在 EOWorld/Wonderplay 基础上添加 LivingWorld 的环境运动功能，保持最小改动，先实现单场景运动。

## 核心思路
- **物体运动**: 使用原有的 Genesis 物理模拟 (已有)
- **环境运动**: 添加 HashGrid + MLP 运动场 (新增)
- **双分支选择**: 通过 config.yaml 中的 `motion_type` 控制

---

## 项目结构

```
/root/autodl-tmp/
├── Wonderplay/          # 原项目，不修改
├── LivingWorld/         # 原项目，不修改
└── EOWorld/             # 新项目 (集成版本)
    └── Wonderplay/      # 从 Wonderplay 复制而来
        ├── WonderPlay_new/
        │   ├── hashgrid.py           # 新增：从 LivingWorld 复制
        │   ├── run_genesis.py        # 修改：添加环境运动分支
        │   └── gaussian_renderer/
        │       └── __init__.py       # 修改：添加 render_MLP
        └── examples/configs/
            └── venice.yaml           # 修改：添加 motion_type
```

## 文件变动清单

### 1. 需要从 LivingWorld 复制的文件

```bash
# 核心运动模型
cp /root/autodl-tmp/LivingWorld/hashgrid.py /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/

# SAM3 相关 (如果 Wonderplay 没有的话)
# Wonderplay 已经有 RepViT-SAM，需要确认是否也有 SAM3
# 检查：ls /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/ | grep sam

# 需要的依赖
pip install tinycudann
```

### 2. 需要修改的文件

#### (1) `examples/configs/venice.yaml`
添加运动类型配置：

```yaml
# 在文件末尾添加
motion_type: "object"  # "object" 或 "environment"

# 环境运动参数 (当 motion_type="environment" 时使用)
environment_motion:
  scale_factor: 2.0  # 运动强度 (0.5-10.0)
  sam_prompt: "water"
  # 固定的运动方向 (可选，用于测试)
  fixed_hints: [[100, 200, 150, 250]]  # [start_x, start_y, end_x, end_y]
```

#### (2) `WonderPlay_new/gaussian_renderer/__init__.py`
添加 `render_MLP` 函数，从 LivingWorld 复制以下内容：
- `flow_to_rgb()` 函数 (line 24-49)
- `pre_euler_integral()` 函数 (line 52-74)
- `render_MLP()` 函数 (line 87-300+)

**注意**: render_MLP 依赖 `scene_flow` 和 `motion_mask`，这些需要通过 SAM3 生成。

#### (3) `WonderPlay_new/run_genesis.py`
在模拟和渲染部分添加分支逻辑。

---

## 实现步骤

### Step 1: 准备工作
```bash
# 1. 复制 HashGrid 模型
cp /root/autodl-tmp/LivingWorld/hashgrid.py /root/autodl-tmp/EOWorld/Wonderplay/WonderPlay_new/

# 2. 安装依赖
pip install tinycudann

# 3. 验证导入
cd /root/autodl-tmp/EOWorld/Wonderplay
python -c "from WonderPlay_new.hashgrid import HashEncoderMotionModel; print('OK')"
```

### Step 2: 添加 render_MLP 到渲染器
在 `WonderPlay_new/gaussian_renderer/__init__.py` 末尾添加：

```python
def render_MLP(viewpoint_camera, pc, motion_model, t, opt, bg_color, 
               scaling_modifier=1.0, override_color=None, 
               render_visible=False, exclude_sky=False, 
               scale_factor=2.0):
    """
    使用 HashGrid MLP 模型渲染环境运动
    """
    # 从 LivingWorld 复制完整实现
    # 包含 pre_euler_integral 的调用
    pass
```

### Step 3: 使用现有分割模型生成运动 mask

**简化方案**: 使用 WonderPlay 现有的 RepViT-SAM，不需要额外集成 SAM3。

WonderPlay 已经有的分割功能：
- ✅ `create_mask_generator_repvit()` - 已有
- ✅ 可以直接用于分割运动区域（水、云、烟等）

只需要添加简单的 mask 生成函数：

```python
# 在 run_genesis.py 中添加
def generate_motion_mask(image_rgb, sam_prompt, mask_generator):
    """
    使用现有的 RepViT-SAM 生成运动 mask
    
    Args:
        image_rgb: RGB图像 [H, W, 3]
        sam_prompt: 文本提示（可选，用于指导分割）
        mask_generator: WonderPlay 现有的 mask generator
        
    Returns:
        motion_mask: [H, W] bool array
    """
    # 使用 WonderPlay 现有的 AMG (Automatic Mask Generation)
    masks = mask_generator.generate(image_rgb)
    
    # 简单策略：选择最大的 mask 作为运动区域
    # 或者根据位置/语义选择（如选择下半部分的 mask 代表水）
    if len(masks) > 0:
        # 选择面积最大的 mask
        largest_mask = max(masks, key=lambda x: x['area'])
        motion_mask = largest_mask['segmentation']
    else:
        # 如果没有检测到，使用全图
        motion_mask = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
    
    return motion_mask
```

**或者更简单**: 直接用固定的区域作为运动 mask（比如图像下半部分代表水面）：

```python
def generate_water_mask_simple(H, W):
    """生成简单的水面 mask（下半部分）"""
    mask = np.zeros((H, W), dtype=bool)
    mask[H//2:, :] = True  # 下半部分
    return mask
```

### Step 4: 修改 run_genesis.py 主循环添加环境运动分支

在模拟步骤后添加分支逻辑：

```python
# ========== 在 simulation 循环中添加 ==========
motion_type = config.get('motion_type', 'object')

if motion_type == "environment":
    # ===== LivingWorld 路径：环境运动 =====
    
    # 1. 获取运动参数
    env_config = config['environment_motion']
    scale_factor = env_config.get('scale_factor', 2.0)
    
    # 2. 生成运动 mask（使用 WonderPlay 现有功能或简化方案）
    H, W = 512, 512
    
    # 方案A：使用现有的 mask_generator
    # image_rgb = kf_gen.image_latest[0].permute(1,2,0).cpu().numpy()
    # image_rgb = (image_rgb * 255).astype(np.uint8)
    # motion_mask = generate_motion_mask(image_rgb, "water", mask_generator)
    
    # 方案B：简单固定区域（推荐用于测试）
    motion_mask = generate_water_mask_simple(H, W)
    
    # 3. 创建简单的 scene_flow（基于固定方向）
    # 为每个 Gaussian 点分配一个速度向量
    means3D = gaussians.get_xyz_all
    scene_flow = torch.zeros_like(means3D)
    
    # 给在 motion_mask 区域内的点分配运动
    # 这里简化处理：假设所有点都在运动区域
    # 运动方向：沿着 y 轴（forward）
    scene_flow[:, 1] = 0.001  # 小的向前运动
    
    # 存储到 gaussians
    gaussians._scene_flow_all = scene_flow
    gaussians._motion_mask_all = torch.ones(means3D.shape[0], 1, dtype=torch.bool, device='cuda')
    
    # 4. 训练 HashGrid 运动模型
    from WonderPlay_new.hashgrid import HashEncoderMotionModel
    motion_model = HashEncoderMotionModel().to('cuda')
    motion_model = train_hashgrid(gaussians, motion_model, iterations=100)
    
    # 5. 渲染
    for frame_idx in range(num_frames):
        render_pkg = render_MLP(
            viewpoint_camera=camera,
            pc=gaussians,
            motion_model=motion_model,
            t=frame_idx,
            opt=opt,
            bg_color=background,
            scale_factor=scale_factor
        )
        # 保存 render_pkg['render']
        
else:
    # ===== WonderPlay 原有路径：物体物理模拟 =====
    # 保持原有的 Genesis 物理模拟代码不变
    render_pkg = render_w_shift_flow(
        viewpoint_camera=camera,
        gaussians=gaussians,
        obj_states=simulation_result,
        opt=opt,
        bg_color=background
    )
```

**关键简化点**：
1. ✅ 不需要 SAM3，用简单的固定区域 mask
2. ✅ 不需要复杂的 flow estimation，用固定的运动方向
3. ✅ scene_flow 直接手动设置，不需要从 2D 投影

### Step 4: 复制训练函数
从 LivingWorld 的 `run.py` 复制 `train_hashgrid` 函数到 `run_genesis.py`:

```python
def train_hashgrid(pc, model, scheduler_gamma=0.2, scheduler_step=100, 
                   iterations=100, lr=1e-2, device='cuda', freeze_mlp=False):
    """
    训练 HashGrid 运动模型
    从 LivingWorld/run.py line 137-227 复制
    """
    means3D = pc.get_xyz_all
    scene_flow = pc.get_scene_flow_all
    # ... (完整实现)
    return model
```

---

## 关键代码位置

### LivingWorld 中需要复制的代码：

1. **HashGrid 模型**: `/root/autodl-tmp/LivingWorld/hashgrid.py`
2. **render_MLP**: `/root/autodl-tmp/LivingWorld/gaussian_renderer/__init__.py` (line 87-300)
3. **train_hashgrid**: `/root/autodl-tmp/LivingWorld/run.py` (line 137-227)
4. **flow_to_rgb**: `/root/autodl-tmp/LivingWorld/gaussian_renderer/__init__.py` (line 24-49)
5. **pre_euler_integral**: `/root/autodl-tmp/LivingWorld/gaussian_renderer/__init__.py` (line 52-74)

### WonderPlay 中需要修改的位置：

1. **配置文件**: `examples/configs/venice.yaml` - 添加 motion_type
2. **渲染器**: `WonderPlay_new/gaussian_renderer/__init__.py` - 添加 render_MLP
3. **主程序**: `WonderPlay_new/run_genesis.py` - 添加分支逻辑

---

## 测试方案

### 测试1: 验证物体运动 (原有功能)
```bash
cd /root/autodl-tmp/EOWorld/Wonderplay
python WonderPlay_new/run_genesis.py \
  --config examples/configs/venice.yaml \
  --prefix test_object
```
预期：船应该按原来的方式运动（物理模拟）

### 测试2: 验证环境运动 (新功能)
修改 `examples/configs/venice.yaml`:
```yaml
motion_type: "environment"
environment_motion:
  scale_factor: 2.0
```

运行：
```bash
cd /root/autodl-tmp/EOWorld/Wonderplay
python WonderPlay_new/run_genesis.py \
  --config examples/configs/venice.yaml \
  --prefix test_environment
```
预期：水面应该有波动效果

---

## 坐标系说明

### 不需要转换！
- **物体分支**: Genesis 内部自己处理坐标转换 (pt3d → genesis)
- **环境分支**: HashGrid 直接在 PyTorch3D 坐标系工作
- **渲染输出**: 两者都输出到 3DGS 坐标系

### 为什么不冲突？
```
物体路径:
  PyTorch3D (场景) → Genesis (模拟) → PyTorch3D (结果) → 3DGS (渲染)
                    ↑ 内部转换     ↑ 内部转换

环境路径:
  PyTorch3D (场景) → HashGrid (预测) → 3DGS (渲染)
                    ↑ 直接使用 PyTorch3D 坐标
```

---

## 光流输出

两个分支都能输出光流：

```python
if motion_type == "environment":
    # LivingWorld: render_MLP 返回的 render_pkg 包含光流
    flow_rgb = render_pkg['render']  # 彩色光流可视化
    
elif motion_type == "object":
    # WonderPlay: render_w_shift_flow 返回的 render_pkg 包含光流
    flow_rgb = render_pkg['render']  # 彩色光流可视化
```

**格式完全一致**，可以直接对比！

---

## 下一步行动（简化版）

1. ✅ 复制 `hashgrid.py`
2. ⬜ 复制 `render_MLP` 等函数到渲染器
3. ⬜ 复制 `train_hashgrid` 函数到主程序
4. ⬜ 添加简单的 `generate_water_mask_simple` 函数
5. ⬜ 修改 `venice.yaml` 添加配置
6. ⬜ 添加分支逻辑到主循环
7. ⬜ 测试物体运动（验证不破坏原有功能）
8. ⬜ 测试环境运动（验证新功能）

**关键简化**：
- ❌ 不需要 SAM3
- ❌ 不需要复杂的 flow estimation
- ❌ 不需要 hints 交互
- ✅ 用固定区域 mask（下半部分=水面）
- ✅ 用固定运动方向（向前流动）
- ✅ 直接手动设置 scene_flow

---

## 注意事项

1. **不要动原有的物理模拟代码**
2. **环境运动只是新增一个并行路径**
3. **两个路径共享场景重建部分**
4. **scale_factor=2.0 作为水流的起始值**
5. **暂时不考虑场景扩展，只做单场景运动**

---

## 估计工作量

- 文件复制: 10分钟
- 代码修改: 30分钟
- 测试调试: 30分钟
- **总计**: 约1小时

如果遇到问题，最可能的是：
1. tinycudann 安装问题 → 按官方文档重装
2. 坐标系混乱 → 检查是否正确使用 visibility_filter
3. 渲染结果不对 → 检查 scale_factor 是否合理
