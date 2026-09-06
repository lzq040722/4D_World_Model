# Multi-view Env-Obj Interaction Implementation Plan (Revised)

## 1. Goal

在保持 WonderPlay Env-Obj Interaction V1 的基础上，引入
LivingWorld-style 多视角扩展能力，实现动态可扩展 4D world。

整体流程：

``` text
Input view
 -> motion annotation
 -> env-obj interaction rendering
 -> camera movement
 -> LivingWorld-style scene expansion
 -> new-view motion annotation
 -> env-obj interaction rendering
 -> repeat
```

核心原则：复用已有 pipeline，仅连接两个系统并同步状态。

------------------------------------------------------------------------

# 2. FrameSyn 与 ImageEdit

FrameSyn 负责组织 LivingWorld 新视角扩展流程：

``` text
Current Gaussian Rendering
        |
        v
Image Inpainting
        |
        v
Depth Estimation
        |
        v
Point Cloud Update
        |
        v
Gaussian Update
```

当前融合方案中保留 FrameSyn 扩展流程，仅替换其中的图像补全模型：

``` text
FrameSyn
   |
   v
WonderPlay ImageEdit Pipeline
   |
   v
Expanded RGB Image
   |
   v
Depth + Point Cloud Update
```

修改位置：

``` text
models/models.py
FrameSyn.inpaint()
```

后续 depth、point cloud、Gaussian 更新流程保持不变。

------------------------------------------------------------------------

# 3. Gaussian Expansion 策略

WonderPlay interaction 依赖 object/environment Gaussian 划分：

``` text
object Gaussian
+
environment Gaussian
```

因此扩展过程需要保持：

``` text
object Gaussian
    unchanged

environment Gaussian
    expanded
```

扩展后的结构：

``` text
object Gaussian
+
environment Gaussian(old)
+
environment Gaussian(new)
```

需要避免 object Gaussian 重复加入导致 interaction renderer 中 split
错误。

------------------------------------------------------------------------

# 4. Motion Field 更新

新增 environment Gaussian 后，需要同步更新 motion information：

``` text
Scene Expansion
        |
        v
New Environment Points
        |
        v
Motion Annotation Binding
        |
        v
attach_flow_to_current_pc_latest()
        |
        v
scene_flow update
        |
        v
train_hashgrid()
        |
        v
Interaction
```

------------------------------------------------------------------------

# 5. 新入口设计

新增：

``` text
WonderPlay_new/run_multiview_interaction.py
```

保持：

``` text
WonderPlay_new/run_genesis.py
```

作为原始离线入口。

新入口负责：

-   socket interaction loop
-   camera view switching
-   scene expansion orchestration
-   调用 WonderPlay V1 interaction

推荐结构：

``` text
MultiView Controller
        |
        +----------------+
        |                |
Expansion Manager   Interaction Manager
```

------------------------------------------------------------------------

# 6. Current-view Interaction

当前 interaction 使用固定 train camera，需要支持用户当前视角。

增加：

``` python
viewpoint_camera=None
```

规则：

``` text
viewpoint_camera != None:
    使用当前浏览视角

viewpoint_camera == None:
    保持旧逻辑
```

------------------------------------------------------------------------

# 7. Simulator 状态

保留：

-   object asset
-   physics configuration
-   scene information

需要明确 simulation state 管理方式，避免不同视角之间状态污染。

第一阶段沿用已有 Genesis 初始化逻辑。

------------------------------------------------------------------------

# 8. 新入口 Pipeline

## Startup

``` text
Load config
 -> Load models
 -> Initial reconstruction
 -> Build Gaussian world
 -> Build simulator
 -> Input-view motion annotation
 -> Initial interaction
 -> Expansion loop
```

## Per-view Loop

``` text
Render current scene
 -> User selects new view
 -> Render current Gaussian
 -> Generate expansion mask
 -> ImageEdit inpainting
 -> Depth estimation
 -> Point cloud update
 -> Environment Gaussian expansion
 -> Motion field update
 -> Interaction rendering
```

------------------------------------------------------------------------

# 9. 风险检查

## Object / Environment Split

检查：

``` text
object_count
environment_count
total_count
scene_flow_count
motion_mask_count
```

保证新增点属于 environment。

------------------------------------------------------------------------

## Geometry Consistency

验证：

``` text
ImageEdit
 -> Depth
 -> Point Cloud
```

保持：

-   camera geometry
-   scene layout
-   existing structure

------------------------------------------------------------------------

# 10. Implementation Order

1.  新增 run_multiview_interaction.py。
2.  接入 LivingWorld socket loop。
3.  保留 WonderPlay ImageEdit pipeline。
4.  接入 FrameSyn expansion。
5.  验证 point cloud update。
6.  验证 environment Gaussian 增量更新。
7.  更新 motion field。
8.  调用 WonderPlay V1 interaction。
9.  测试连续多视角扩展。

------------------------------------------------------------------------

# Final Goal

``` text
LivingWorld multi-view expansion

+

WonderPlay object-environment interaction

=

Expandable Interactive 4D World
```

------------------------------------------------------------------------

# 11. Function Checklist

下面是后续实现时需要对照修改的函数清单。

| 文件 | 函数 | 要实现的功能 |
|---|---|---|
| `WonderPlay_new/run_multiview_interaction.py` | `main()` / `run()` | 启动多视角交互入口，加载配置、初始化模型、启动 socket loop、驱动整条多视角流程 |
| `WonderPlay_new/run_multiview_interaction.py` | `handle_gen()` | 接收前端“生成新视角”请求，触发当前视角 expansion |
| `WonderPlay_new/run_multiview_interaction.py` | `handle_render_pose()` | 接收前端当前相机位姿，更新当前浏览视角 |
| `WonderPlay_new/run_multiview_interaction.py` | `on_sam_click()` | 接收 motion arrows / SAM 点击点，保存当前视角 motion annotation |
| `WonderPlay_new/run_multiview_interaction.py` | `handle_ok_start()` | 确认当前视角标注完成，进入后续 inpaint / flow / interaction 流程 |
| `WonderPlay_new/run_multiview_interaction.py` | `on_set_sam_prompt()` | 更新当前视角的 SAM prompt |
| `WonderPlay_new/run_multiview_interaction.py` | `on_set_scale()` | 更新 motion / render 强度参数 |
| `WonderPlay_new/run_multiview_interaction.py` | `render_current_scene()` | 渲染当前 Gaussian world，供浏览器预览和视角切换使用 |
| `WonderPlay_new/multiview_controller.py` | `MultiViewController` | 组织 expansion、interaction、prompt、save 四类状态与流程 |
| `WonderPlay_new/multiview_controller.py` | `ExpansionManager` | 负责当前视角 expansion、inpaint、depth、point cloud update、environment Gaussian update |
| `WonderPlay_new/multiview_controller.py` | `InteractionManager` | 负责当前视角 env-obj interaction 调用、simulator reset、render 输出 |
| `WonderPlay_new/multiview_controller.py` | `PromptManager` | 负责 prompt state、`use_gpt`、`scene-prompt` 更新逻辑 |
| `WonderPlay_new/interaction_v1.py` | `prepare_environment_motion_fields()` | 复用环境 motion 估计、flow binding 逻辑 |
| `WonderPlay_new/interaction_v1.py` | `validate_interaction_config()` | 校验 env2obj / obj2env 配置约束 |
| `WonderPlay_new/interaction_v1.py` | `train_interaction_motion_model()` | 根据当前 Gaussian state 训练 / 微调 hashgrid motion model |
| `WonderPlay_new/interaction_v1.py` | `get_interaction_query_points()` | 获取 env2obj 查询点 |
| `WonderPlay_new/interaction_v1.py` | `generate_object_motion_hints()` | 由 Genesis object motion 生成 obj2env motion hints |
| `WonderPlay_new/interaction_v1.py` | `collect_interaction_states()` | 每轮 interaction 重新采样 simulator state 并收集状态序列 |
| `WonderPlay_new/interaction_v1.py` | `run_interaction_pipeline()` | 对当前视角执行 env-obj interaction 主流程 |
| `WonderPlay_new/interaction_v1.py` | `interaction_rendering()` | 生成当前视角 interaction video、depth、mask、flow 输出 |
| `WonderPlay_new/models/models.py` | `FrameSyn.inpaint()` | 让扩展阶段使用 Wonderplay ImageEdit backend 完成 inpaint |
| `WonderPlay_new/models/models.py` | `recompose_image_latest_and_set_current_pc()` | 支持输入视角 motion annotation，并把当前视角 flow 绑定到 point cloud |
| `WonderPlay_new/models/models.py` | `update_current_pc_by_kf()` | 将当前视角图像、深度、flow、mask 更新到 current_pc / current_pc_latest |
| `WonderPlay_new/models/models.py` | `get_camera_by_js_view_matrix()` | 将浏览器视角转换为当前相机，用于当前视角 expansion 和 interaction |
| `WonderPlay_new/models/models.py` | `convert_to_3dgs_traindata_latest()` / `convert_to_3dgs_traindata_latest_layer()` | 将当前视角新增点云转成 3DGS 训练数据 |
| `WonderPlay_new/gaussian_renderer/living_world_render.py` | `render_MLP()` | 复用 environment motion model 进行动态渲染 |
| `WonderPlay_new/gaussian_renderer/living_world_render.py` | `render_interaction_mlp()` | 渲染 object + environment 的当前视角 interaction 结果 |
| `WonderPlay_new/gaussian_renderer/__init__.py` | `render_interaction()` | 渲染静态 / 动态 object 与 environment 的组合结果 |
| `WonderPlay_new/splat-main-new/main_stream.js` | 前端事件绑定 | 发送 `gen` / `render-pose` / `sam-click` / `ok-start` / `scene-prompt` 等事件 |
| `WonderPlay_new/splat-main-new/index_stream.html` | 交互界面 | 提供当前视角预览、motion 标注、prompt 编辑、生成控制 |

### 11.1 需完成的功能清单

1. 输入视角可完成 motion annotation。
2. 输入视角可完成一次 env-obj interaction。
3. 当前浏览器视角可触发 LivingWorld-style expansion。
4. 新视角 expansion 只更新 environment Gaussian。
5. 新视角 expansion 后先做 flow binding，再训练 hashgrid。
6. 每轮 interaction 都基于当前 `viewpoint_camera`。
7. 每轮 interaction 前 simulator 都重新 reset。
8. prompt / GPT / `scene-prompt` 可在多视角循环中持续更新。
9. 旧入口 `run_genesis.py` 行为保持不变。
10. `load_gen=True` 的保存路径和接口预留完整。
