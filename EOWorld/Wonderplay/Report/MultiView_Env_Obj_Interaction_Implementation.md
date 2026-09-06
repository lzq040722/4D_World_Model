# Multi-view Env-Obj Interaction Implementation Notes

## 1. 实现范围

本次实现以现有 WonderPlay V1 pipeline 为基准，新增 LivingWorld-style
多视角编排层。`run_genesis.py` 的默认离线行为保持原有路径；只有通过
`run_multiview_interaction.py` 启动并打开 `multiview.enabled` 时，才进入新
的交互循环。

运行顺序为：

```text
初始输入视角
  -> 用户 motion annotation
  -> WonderPlay env-obj interaction
  -> 用户移动浏览器视角
  -> LivingWorld-style expansion
  -> 当前视角重新 motion annotation
  -> WonderPlay env-obj interaction
  -> 下一次视角扩展
```

## 2. 修改文件

### `WonderPlay_new/run_multiview_interaction.py`

新增多视角入口和 LivingWorld 风格 socket 事件循环：

- 启动 Flask-SocketIO 服务和当前视角预览线程；
- 接收 `render-pose`、`gen`、`sam-click`、`ok-start`、`scene-prompt`、
  `set-sam-prompt`、`set-scale` 等已有前端事件；
- 保存当前浏览器视角和 motion annotation；
- 将事件和线程同步对象通过 `get_runtime_hooks()` 提供给控制器；
- 继续使用现有前端事件名称，前端无需改变协议。

入口同时挂载项目内 `WonderPlay_new/splat-main` 的
`index_stream.html` 和 `main_stream.js`：

- `/` 返回 `index_stream.html`；
- 静态资源由同一个 Flask 服务提供；
- Socket.IO 使用当前页面的 origin，不再固定连接 LivingWorld 的
  `17778` 端口；
- LivingWorld 的 `server-state` 文本协议保持不变，前端按钮状态可以
  正常随流程切换。

入口示例：

```bash
python WonderPlay_new/run_multiview_interaction.py \
  --config examples/configs/Golden_Gate_Bridge_I.yaml \
  --port 5000
```

### `WonderPlay_new/multiview_controller.py`

新增三个职责明确的兼容层：

- `PromptManager`：维护当前 scene prompt，并在 `use_gpt=True` 时调用项目内
  搬运的 `WonderPlay_new/util/chatGPT4.py` 中的 `TextpromptGen`；
- 按 LivingWorld 方式从项目内 `examples/examples.yaml` 读取默认
  `content_prompt`、`style_prompt` 和 `background`；
- `TextpromptGen` 在两种模式下都会初始化；只有 `use_gpt=True` 时调用
  `wonder_next_scene()`，两种模式都调用 `generate_prompt()`；
- `ExpansionManager`：复用 WonderPlay 的 render、ImageEdit、depth、
  `update_current_pc_by_kf()`、`convert_to_3dgs_traindata_latest()` 和
  `train_gaussian()`，完成当前视角的扩展；
- `InteractionManager`：调用现有 `run_interaction_pipeline()`，执行
  env2obj / obj2env 和交互视频渲染。

### `WonderPlay_new/splat-main`

从 LivingWorld 搬入完整前端目录，保留 `index_stream.html`、
`main_stream.js` 及其配套文件。仅修改 `main_stream.js` 的 Socket.IO
连接地址，使其连接当前 WonderPlay 页面所在端口；motion annotation、
视角发送、prompt、生成、停止等事件名称保持 LivingWorld 原协议。

### `WonderPlay_new/run_genesis.py`

增加可选兼容接口：

- `interaction_rendering(..., viewpoint_camera=None)` 支持当前浏览视角；
- `run_interaction_pipeline(..., viewpoint_camera=None)` 支持当前浏览视角；
- `motion_type=interaction` 且 `multiview.enabled=True` 时，将流程交给
  `MultiViewController`；
- 多视角模式下，初始 environment flow 延迟到用户完成输入视角标注后再绑定；
- 默认离线模式仍执行原有 interaction 分支。

### `WonderPlay_new/scene/gaussian_model.py`

新增：

```python
GaussianModel.append_environment_from_gaussian(source)
```

该方法只把 `source._xyz` 及其 Gaussian 属性追加到目标模型的
`_xyz_prev`、`_features_dc_prev`、`_scaling_prev`、`_rotation_prev`、
`_opacity_prev` 和对应 motion metadata 中。这样新视角新增点保持为
environment，原有 `_xyz` object 段保持不变。

### `examples/base-config.yaml`

增加配置接口：

```yaml
image_edit_checkpoint: "/root/autodl-tmp/huggingface/hub"

multiview:
  enabled: False
  expansion_iterations: 100
  inpainting_steps: 50
  stop: False

load_gen: False
```

LivingWorld 的 prompt 生成实现已经复制到
`WonderPlay_new/util/chatGPT4.py`，多视角运行时不再依赖外部
`LivingWorld` 路径。 `use_gpt=True` 时仍按 LivingWorld 原有的
`wonder_next_scene()` 和 `generate_prompt()` 顺序生成后续视角 prompt。

LivingWorld 的 `examples/examples.yaml` 也已复制到项目内
`examples/examples.yaml`。当场景配置没有显式提供 `content_prompt` 时，
代码按 `example_name` 查找该文件；如果两处都没有场景 prompt，则直接报错。
`text_prompt` 继续只用于交互运动/视频输出，不作为场景扩展 prompt。

ImageEdit 仍由 `FrameSyn.inpaint()` 使用的
`ImageEditInpaintPipeline` 完成；多视角入口没有切换回 Stable Diffusion
2 inpaint。

## 3. Gaussian 状态更新

每次视角扩展执行以下顺序：

```text
当前 Gaussian render
  -> LivingWorld outpaint mask
  -> FrameSyn.inpaint / ImageEdit
  -> Marigold depth
  -> update_current_pc_by_kf()
  -> convert_to_3dgs_traindata_latest()
  -> 临时 GaussianModel(previous_gaussian=old_gaussians)
  -> train_gaussian() 只优化新增当前点
  -> append_environment_from_gaussian()
  -> 新点进入 environment 段
```

扩展结束后控制器检查 object 点数、environment 点数和新增点数，防止新增
点被错误加入 object 段。

## 4. Motion 与 Interaction 状态

当前视角扩展完成后，控制器等待新的 motion annotation。收到
`ok-start` 后执行：

```text
SAM3 segmentation
  -> Cinemagraphy flow estimation
  -> attach_flow_to_current_pc_latest()
  -> sync_current_pc_scene_flow_to_final_environment_gaussians()
  -> train_interaction_motion_model()
  -> run_interaction_pipeline()
```

`run_interaction_pipeline()` 使用传入的 `viewpoint_camera` 进行 flow 投影、
环境位置积分和交互渲染；未传入时保留原有 train camera 0 的逻辑。

## 5. Simulator 生命周期

按照本次确认，所有视角复用同一个 Genesis `Simulator` 实例。每轮
interaction 调用同一个实例继续执行，因此 Genesis 物理状态会沿用上一轮
simulation 的结果。该行为是当前实现明确采用的生命周期策略，后续如果需要
独立视角物理状态，需要另行增加 Genesis scene 重建接口。

## 6. `load_gen` 接口

配置中已经预留：

```yaml
load_gen: False
```

多视角入口会识别该选项。当前保存/加载已有 Gaussian 和
`motion_model.pth` 的完整恢复流程尚未接入；设置为 `True` 时会在控制器入口
明确报错，避免误把未完整恢复的状态用于 interaction。

## 7. 验证结果

已完成：

- 新增和修改的 Python 文件通过 `py_compile`；
- Python AST 解析通过；
- OmegaConf base config 与场景 config 合并通过；
- `git diff --check` 通过。

完整入口导入验证被当前环境已有依赖阻断：

```text
ModuleNotFoundError: No module named 'pytorch3d'
```

该错误发生在导入已有 `WonderPlay_new/util/utils.py` 时，尚未进入新入口或
控制器逻辑。完整 GPU 运行还需要当前项目原有的 PyTorch3D、Genesis、扩散
模型权重以及 Cinemagraphy checkpoint。

## 8. 后续对照顺序

1. 安装并确认当前 WonderPlay 环境的 PyTorch3D 依赖。
2. 使用 `motion_type: interaction`、`direction: env2obj` 的配置启动新入口。
3. 在初始视角提交 motion annotation，确认第一轮 interaction 输出。
4. 移动浏览器视角并发送 `gen`，确认 expansion mask 和 environment 点数增长。
5. 在新视角重新提交 motion annotation，确认 flow binding、HashGrid 和
   interaction 输出均使用当前相机。
6. 连续执行第二次 expansion，确认 object 点数保持不变、environment 点数
   继续增长。
