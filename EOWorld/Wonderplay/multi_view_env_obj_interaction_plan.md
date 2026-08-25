# LivingWorld Expansion + WonderPlay Env-Obj Interaction 修改说明

## 1. 目标

新增 LivingWorld-style 多视角交互入口，实现如下循环：

```text
输入视角
  -> motion annotation
  -> 当前视角 env-obj interaction
  -> 用户移动到新视角
  -> LivingWorld expansion
  -> 新视角 motion annotation
  -> 新视角 env-obj interaction
  -> 重复
```

实现原则：

- 以现有 Wonderplay 代码为基准。
- 复用已有模块。
- 保留 Wonderplay V1 interaction。
- 保留 LivingWorld expansion 的交互式工作流。

## 2. 实现约束

### 2.1 Gaussian object/environment split

修改要求：

- 维护 `object_gaussian` 和 `environment_gaussian` 两个逻辑分量。
- expansion 只更新 environment Gaussian。
- 最终渲染与 interaction 使用 `object_gaussian + environment_gaussian`。
- 如果继续使用 `GaussianModel`，则以固定 `object_count` 进行 split。

### 2.2 motion_model 与 Gaussian state

修改要求：

```text
Expansion
  -> new environment points
  -> motion annotation propagation
  -> attach_flow_to_current_pc_latest()
  -> train_hashgrid()
```

### 2.3 FrameSyn 与 ImageEdit backend

修改要求：

- 保留 LivingWorld 的扩展状态机。
- 保留 Wonderplay 的 ImageEdit backend。
- `kf_gen.inpaint(...)` 使用 ImageEdit 实现。
- 以 backend 方式组织 inpaint 实现，便于控制流程复用。

### 2.4 run_multiview_interaction.py 职责

修改要求：

```text
run_multiview_interaction.py
  -> MultiViewController
      -> ExpansionManager
      -> InteractionManager
      -> PromptManager
      -> UIServer
```

入口负责启动与路由，核心流程进入 controller / manager。

### 2.5 Simulator state

修改要求：

- 复用 object asset、physics config 和 scene mesh。
- 每轮 interaction 重置 simulator state。

### 2.6 motion annotation 顺序

修改要求：

```text
Initial Reconstruction
  -> Initial Gaussian
  -> Motion annotation
  -> Flow binding
  -> Motion model
  -> Interaction
```

## 3. 入口结构

保留两个入口：

```text
WonderPlay_new/run_genesis.py
WonderPlay_new/run_multiview_interaction.py
```

`run_genesis.py` 保持现有离线行为。  
`run_multiview_interaction.py` 承担多视角交互流程。

## 4. 运行流程

### 4.1 启动阶段

```text
parse config
load models
create KeyframeGen
load input image
generate sky point cloud
show input view
wait for input-view motion annotation
bind input-view motion
build initial Gaussian
build initial motion_model
build object reconstruction and simulator
run interaction for view_000
enter expansion loop
```

### 4.2 单轮扩展阶段

```text
render current scene
build inpaint mask
ImageEdit inpaint
depth refine
update current point cloud
motion annotation
attach_flow_to_current_pc_latest(...)
train / finetune hashgrid
train Gaussian update for environment
run current-view interaction
```

### 4.3 当前视角 interaction

`run_interaction_pipeline(...)` 支持 `viewpoint_camera`。

默认值保持当前逻辑：

```python
viewpoint_camera = scene.getTrainCameras().copy()[0]
```

多视角入口传入浏览器当前 camera。

## 5. Gaussian 管理

多视角扩展显式维护：

```text
object_gaussian (fixed)
environment_gaussian (mutable)
```

要求：

- object Gaussian 只构建一次。
- expansion 只更新 environment Gaussian。
- `GaussianModel(previous_gaussian=...)` 仅用于环境侧增量更新。
- object/environment split 使用固定 `object_count` 维护。

## 6. Prompt 与 GPT

保留 LivingWorld prompt 状态机：

```text
scene_dict
style_prompt
background_prompt
adaptive_negative_prompt
control_text
pt_gen
change_scene_name_by_user
use_gpt
scene_name
```

prompt 规则：

1. 优先读取 Wonderplay config 中的 `content_prompt / style_prompt / negative_prompt / background / control_text`。
2. 配置缺失时兼容 LivingWorld `examples/examples.yaml`。
3. 每轮扩展前调用 `pt_gen.generate_prompt(...)`。
4. `use_gpt=True` 时调用 `pt_gen.wonder_next_scene(...)`。
5. 前端 `scene-prompt` 事件更新当前 prompt 状态。

## 7. Inpaint backend

inpaint backend 统一使用 Wonderplay 当前 ImageEdit 实现。

约定：

- 保留 `kf_gen.inpaint(...)` 调用入口。
- `kf_gen.inpaint(...)` 内部对接 ImageEdit pipeline。
- 扩展控制继续沿用 LivingWorld 交互状态机。

## 8. Simulator 生命周期

每轮 interaction 采用以下复用策略：

```text
reuse:
  object asset
  physics config
  scene mesh

reset:
  simulator state
```

## 9. Config 修改

新增配置块：

```yaml
multi_view_interaction:
  enabled: true
  port: 17778
  output_per_view: true
  view_dir_template: "view_{view_id:03d}"
  run_input_view_interaction: true
  save_each_view: true
  max_views: -1
  load_gen: false
  inpaint_backend: imageedit
  prompt_source: config
  use_gpt: false
  enable_undo: true
  enable_delete: true
  enable_save: true
```

继续复用的字段：

```yaml
content_prompt: "<scene, object, environment...>"
style_prompt: "<style>"
negative_prompt: ""
background: ""
control_text: null
text_prompt: "<video refinement prompt>"

motion_type: "interaction"
interaction:
  direction: "env2obj"
  velocity_scale: 0.5
  num_frames: 50
  environment_scale_factor: 1.0

environment_motion:
  scale_factor: 1.0
  sam_prompt: "water"
  fixed_hints:
    - [x0, y0, x1, y1]
```

约定：

- `fixed_hints` 作为离线入口的 fallback。
- 新多视角入口使用前端 arrows 覆盖 `fixed_hints`。
- `load_gen=True` 预留第二阶段恢复入口。

`load_gen=True` 的保存目录约定：

```text
<input_dir>/model/
  finished_3dgs.ply
  visibility_filter_all.pth
  is_sky_filter.pth
  delete_mask_all.pth
  flows_layer.pth
  motion_model.pth
```

## 10. 代码文件修改清单

| 文件 | 修改内容 | 目的 |
|---|---|---|
| `WonderPlay_new/run_multiview_interaction.py` | 新增多视角交互入口 | 启动多视角交互流程 |
| `WonderPlay_new/multiview_controller.py` | 新增主控制器 | 拆分 expansion / interaction / prompt / save |
| `WonderPlay_new/interaction_v1.py` | 抽出 V1 helper | 复用现有 env-obj interaction |
| `WonderPlay_new/models/models.py` | 扩展 `recompose_image_latest_and_set_current_pc(...)` 参数 | 支持输入视角 motion annotation 和新视角更新 |
| `WonderPlay_new/gaussian_renderer/living_world_render.py` | 支持当前 `viewpoint_camera` | 支持当前视角 interaction |
| `WonderPlay_new/run_genesis.py` | 最小共享抽取或 import 调整 | 保持旧入口行为 |
| `WonderPlay_new/splat-main-new/*` | 复制并适配前端事件 | 支持多视角交互 |
| `examples/base-config.yaml` 与 `examples/configs/*.yaml` | 增加 `multi_view_interaction` 配置块 | 支持新入口参数 |

## 11. 执行顺序

1. 新增 `interaction_v1.py`。
2. 新增 `run_multiview_interaction.py`。
3. 新增 `multiview_controller.py`。
4. 复制并适配 `splat-main-new` 前端。
5. 接入 ImageEdit / Marigold / SAM3 / OneFormer 初始化。
6. 接入 prompt/GPT 状态机。
7. 实现输入视角 motion annotation + flow binding。
8. 实现新视角 expansion，并只更新 environment Gaussian。
9. 每轮 expansion 后调用现有 V1 interaction。
10. 补齐 `load_gen=True` 恢复逻辑。

## 12. 验证目标

- 旧入口 `run_genesis.py` 行为保持一致。
- 输入视角先完成 motion annotation，再进入 interaction。
- 新视角 expansion 不重复加入 object Gaussian。
- 每轮 expansion 都先 attach flow，再 train hashgrid。
- 每轮 interaction 使用当前 `viewpoint_camera`。
- simulator 每轮从干净状态启动。
- prompt / GPT / scene-prompt 在多视角循环中保持可用。
