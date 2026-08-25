"""Interactive LivingWorld-style expansion + WonderPlay interaction entry."""

import io
import threading
import time
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import cv2
import torch
from flask import Flask, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from omegaconf import OmegaConf
from PIL import Image

import run_genesis as genesis


_SPLAT_DIR = Path(__file__).resolve().parent / "splat-main"
app = Flask(
    __name__,
    static_folder=str(_SPLAT_DIR),
    static_url_path="",
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

_runtime_lock = threading.Lock()
_annotation_event = threading.Event()
_mask_review_event = threading.Event()
_expansion_event = threading.Event()
_stop_event = threading.Event()
_record_lock = threading.Lock()
_frame_lock = threading.Lock()
_client_lock = threading.Lock()
_client_ids = set()
_annotation_lock = threading.Lock()
_phase_lock = threading.Lock()
_pipeline_phase = "INITIALIZING"
_phase_message = "Pipeline is initializing."
_view_matrix = list(genesis.view_matrix_wonder)
_sam_enabled = False
_sam_prompt = "water"
_scale_factor = 1.0
_clicks = []
_confirmed_clicks = []
_annotation_frame_bytes = None
_pending_scene_prompt = None
_command_handler = None
_capture_active = False
_capture_writer = None
_capture_path = None
_capture_frame_count = 0


@app.route("/")
def index():
    return app.send_static_file("index_stream.html")


@app.route("/annotation-frame.png")
def annotation_frame():
    """Serve the current annotation image without requiring Socket.IO."""
    with _frame_lock:
        frame_bytes = _annotation_frame_bytes
    if frame_bytes is None:
        return Response(status=404)
    return Response(
        frame_bytes,
        mimetype="image/png",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _emit(event, payload):
    with _client_lock:
        client_ids = tuple(_client_ids)
    for client_id in client_ids:
        socketio.emit(event, payload, room=client_id)


def _set_pipeline_phase(phase, message):
    global _pipeline_phase, _phase_message
    with _phase_lock:
        _pipeline_phase = str(phase)
        _phase_message = str(message)
    print(f"[multiview] phase={_pipeline_phase}: {_phase_message}", flush=True)
    _emit(
        "pipeline-phase",
        {"phase": _pipeline_phase, "message": _phase_message},
    )
    _emit("server-state", _phase_message)


def _get_pipeline_phase():
    with _phase_lock:
        return _pipeline_phase, _phase_message


def _image_to_png_bytes(image):
    if torch.is_tensor(image):
        tensor = image.detach().cpu().clamp(0, 1)
        if tensor.ndim == 4:
            tensor = tensor[0]
        array = (
            tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
        )
        image = Image.fromarray(array, mode="RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _set_annotation_frame(image):
    global _annotation_frame_bytes
    frame_bytes = _image_to_png_bytes(image)
    with _frame_lock:
        _annotation_frame_bytes = frame_bytes
        _emit("annotation-frame", _annotation_frame_bytes)


def _reset_annotations():
    with _annotation_lock:
        _clicks.clear()
        _confirmed_clicks.clear()


def _review_motion_mask(image):
    global _annotation_frame_bytes
    with _frame_lock:
        input_frame_bytes = _annotation_frame_bytes
    _mask_review_event.clear()
    _set_pipeline_phase("PREPARING_MASK_REVIEW", "Preparing SAM3 mask preview...")
    _set_annotation_frame(image)
    _set_pipeline_phase(
        "MASK_REVIEW",
        "Check the SAM3 mask on the input image, then click Confirm.",
    )
    _mask_review_event.wait()
    if input_frame_bytes is not None:
        with _frame_lock:
            _annotation_frame_bytes = input_frame_bytes
            _emit("annotation-frame", _annotation_frame_bytes)


def _current_view_matrix():
    return list(_view_matrix)


def _clicks_to_hints(clicks):
    if not clicks:
        return np.empty((4, 0), dtype=int)
    points = list(clicks)
    if len(points) % 2 == 1:
        points = points[1:]
    if not points:
        return np.empty((4, 0), dtype=int)
    pairs = np.asarray(points, dtype=int).reshape(-1, 2, 2)
    return np.stack(
        [pairs[:, 0, 0], pairs[:, 0, 1], pairs[:, 1, 0], pairs[:, 1, 1]],
        axis=0,
    )


def _points_from_arrow_payload(data):
    if not isinstance(data, dict) or "arrows" not in data:
        return None
    points = []
    for arrow in data.get("arrows", []):
        if not isinstance(arrow, dict):
            raise ValueError("Each motion arrow must be an object")
        start, end = arrow.get("start"), arrow.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ValueError("Each motion arrow requires start and end points")
        for point in (start, end):
            x = max(0, min(511, int(round(float(point["x"])))))
            y = max(0, min(511, int(round(float(point["y"])))))
            points.append((x, y))
    return points


def _get_annotations():
    with _annotation_lock:
        confirmed = list(_confirmed_clicks)
    return _clicks_to_hints(confirmed)


def _consume_scene_prompt():
    global _pending_scene_prompt
    prompt = _pending_scene_prompt
    _pending_scene_prompt = None
    return prompt


def _set_command_handler(handler):
    global _command_handler
    _command_handler = handler


def _dispatch_command(command, payload=None):
    if _command_handler is None:
        emit("server-state", "Pipeline is still initializing.", room=request.sid)
        return
    try:
        _command_handler(command, payload)
    except Exception as exc:
        emit("server-state", f"{command} failed: {exc}", room=request.sid)


def get_runtime_hooks():
    """Return callbacks consumed by ``multiview_controller``."""
    return {
        "annotation_event": _annotation_event,
        "mask_review_event": _mask_review_event,
        "expansion_event": _expansion_event,
        "get_annotations": _get_annotations,
        "get_view_matrix": _current_view_matrix,
        "consume_scene_prompt": _consume_scene_prompt,
        "set_command_handler": _set_command_handler,
        "set_pipeline_phase": _set_pipeline_phase,
        "set_annotation_frame": _set_annotation_frame,
        "reset_annotations": _reset_annotations,
        "review_motion_mask": _review_motion_mask,
        "runtime_lock": _runtime_lock,
        "emit": _emit,
    }


@socketio.on("connect")
def _handle_connect():
    with _client_lock:
        _client_ids.add(request.sid)
    emit("scale-state", {"value": _scale_factor}, room=request.sid)
    if _annotation_frame_bytes is not None:
        emit("annotation-frame", _annotation_frame_bytes, room=request.sid)
    if isinstance(genesis.scene_dict, dict):
        emit("scene-prompt", genesis.scene_dict.get("scene_name", ""), room=request.sid)
    # A reconnect must restore the active phase so the relevant controls are
    # enabled again.  A generic "connected" message loses that information.
    phase, phase_message = _get_pipeline_phase()
    emit(
        "pipeline-phase",
        {"phase": phase, "message": phase_message},
        room=request.sid,
    )
    emit("server-state", phase_message, room=request.sid)


@socketio.on("disconnect")
def _handle_disconnect():
    with _client_lock:
        _client_ids.discard(request.sid)


@socketio.on("start")
def _handle_start(_data=None):
    _, phase_message = _get_pipeline_phase()
    emit("server-state", phase_message, room=request.sid)


@socketio.on("render-pose")
def handle_render_pose(data):
    global _view_matrix
    _view_matrix = list(data)
    genesis.view_matrix_wonder = list(data)


@socketio.on("gen")
def handle_gen(data):
    global _view_matrix
    phase, phase_message = _get_pipeline_phase()
    if phase != "WAITING_EXPANSION":
        return {"ok": False, "message": f"Generate rejected: {phase_message}"}
    _view_matrix = list(data)
    genesis.view_matrix = list(data)
    _set_pipeline_phase("PROCESSING", "Generating new scene...")
    _expansion_event.set()
    return {"ok": True, "message": "New viewpoint requested."}


@socketio.on("scene-prompt")
def handle_new_prompt(data):
    global _pending_scene_prompt
    if not isinstance(data, str):
        return
    _pending_scene_prompt = data
    genesis.scene_name = data
    if isinstance(genesis.scene_dict, dict):
        genesis.scene_dict["scene_name"] = data


@socketio.on("sam-toggle")
def on_sam_toggle(msg):
    global _sam_enabled
    _sam_enabled = bool(msg.get("enabled", False))
    emit("server-state", f"SAM {'ON' if _sam_enabled else 'OFF'}", room=request.sid)


@socketio.on("sam-click")
def on_sam_click(msg):
    if not _sam_enabled:
        return {"ok": False, "reason": "Motion annotation is disabled"}
    size = msg.get("size", [512, 512])
    width, height = int(size[0]), int(size[1])
    xy = msg.get("xy")
    if xy is None:
        uv = msg.get("uv")
        if uv is None:
            return {"ok": False, "reason": "Missing xy/uv"}
        x = round(float(uv[0]) * (width - 1))
        y = round(float(uv[1]) * (height - 1))
    else:
        x, y = int(xy[0]), int(xy[1])
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    with _annotation_lock:
        if not _clicks or abs(_clicks[-1][0] - x) > 4 or abs(_clicks[-1][1] - y) > 4:
            _clicks.append((x, y))
            if len(_clicks) > 100:
                del _clicks[:-100]
    return {"ok": True}


@socketio.on("sam-clear")
def on_sam_clear():
    _reset_annotations()
    emit("server-state", "Motion annotations cleared.", room=request.sid)
    return {"ok": True}


@socketio.on("ok-start")
def handle_ok_start(data=None):
    phase, phase_message = _get_pipeline_phase()
    if phase == "MASK_REVIEW":
        message = "SAM3 mask confirmed. Computing motion flow..."
        _set_pipeline_phase("PROCESSING", message)
        _mask_review_event.set()
        return {"ok": True, "arrow_count": 0, "message": message}

    if phase != "WAITING_ANNOTATION":
        message = f"Confirm rejected: pipeline state is '{phase_message}'."
        print(f"[multiview] {message}", flush=True)
        return {"ok": False, "message": message}

    try:
        payload_points = _points_from_arrow_payload(data)
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "message": f"Invalid motion arrows: {exc}"}
    with _annotation_lock:
        if payload_points is not None:
            _clicks[:] = payload_points
        _confirmed_clicks[:] = list(_clicks)
        arrow_count = len(_confirmed_clicks) // 2
    direction = str(
        genesis_runtime_config.get("interaction", {}).get("direction", "")
    ).lower()
    if direction == "env2obj" and arrow_count == 0:
        message = (
            "Confirm rejected: no motion arrows reached the server. "
            "Enable Motion annotation and draw at least one arrow."
        )
        print(f"[multiview] {message}", flush=True)
        return {"ok": False, "arrow_count": 0, "message": message}

    _set_pipeline_phase("PROCESSING", "Motion annotation received. Processing...")
    _annotation_event.set()
    message = f"OK clicked. Motion annotation confirmed: {arrow_count} arrow(s)."
    print(f"[multiview] {message}", flush=True)
    _emit("server-state", message)
    return {
        "ok": True,
        "arrow_count": arrow_count,
        "message": message,
    }


@socketio.on("set-sam-prompt")
def on_set_sam_prompt(msg):
    global _sam_prompt
    _sam_prompt = str(msg.get("value", _sam_prompt))
    if "environment_motion" in genesis_runtime_config:
        genesis_runtime_config["environment_motion"]["sam_prompt"] = _sam_prompt
    message = f"SAM3 mask prompt set to '{_sam_prompt}'."
    emit("server-state", message, room=request.sid)
    return {"ok": True, "value": _sam_prompt, "message": message}


@socketio.on("set-scale")
def on_set_scale(msg):
    global _scale_factor
    _scale_factor = max(0.0, float(msg.get("value", _scale_factor)))
    if "interaction" in genesis_runtime_config:
        genesis_runtime_config["interaction"]["environment_scale_factor"] = _scale_factor
    emit("scale-state", {"value": _scale_factor}, room=request.sid)


@socketio.on("undo")
def handle_undo():
    _dispatch_command("undo")


@socketio.on("save")
def handle_save():
    _dispatch_command("save")


@socketio.on("fill_hole")
def handle_fill_hole():
    _dispatch_command("fill_hole")


@socketio.on("delete")
def handle_delete(data):
    _dispatch_command("delete", list(data))


@socketio.on("capture")
def handle_capture():
    global _capture_active, _capture_writer, _capture_path, _capture_frame_count
    if genesis.kf_gen is None:
        emit("server-state", "Pipeline is still initializing.", room=request.sid)
        return
    with _record_lock:
        if _capture_writer is not None:
            _capture_writer.release()
        capture_dir = genesis.kf_gen.run_dir / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        _capture_path = capture_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        _capture_writer = None
        _capture_frame_count = 0
        _capture_active = True
    emit("server-state", "Recording preview frames.", room=request.sid)


@socketio.on("stop")
def handle_stop():
    global _capture_active, _capture_writer, _capture_path, _capture_frame_count
    with _record_lock:
        was_active = _capture_active
        _capture_active = False
        if _capture_writer is not None:
            _capture_writer.release()
        video_path = _capture_path
        frame_count = _capture_frame_count
        _capture_writer = None
        _capture_path = None
        _capture_frame_count = 0
    if not was_active:
        emit("server-state", "No preview recording is active.", room=request.sid)
        return
    if frame_count == 0:
        emit("server-state", "Recording stopped without frames.", room=request.sid)
        return
    emit("server-state", f"Recording saved to {video_path}", room=request.sid)


def render_current_scene():
    """Reuse the WonderPlay renderer for browser viewpoint previews."""
    global _capture_writer, _capture_frame_count
    while not _stop_event.is_set():
        try:
            phase, _ = _get_pipeline_phase()
            if phase != "WAITING_EXPANSION":
                time.sleep(0.05)
                continue
            if genesis.kf_gen is None or genesis.gaussians is None:
                time.sleep(0.1)
                continue
            with _runtime_lock, torch.no_grad():
                camera = genesis.kf_gen.get_camera_by_js_view_matrix(
                    _current_view_matrix(), xyz_scale=genesis.xyz_scale
                )
                tdgs_camera = genesis.convert_pt3d_cam_to_3dgs_cam(
                    camera, xyz_scale=genesis.xyz_scale
                )
                render_pkg = genesis.render(
                    tdgs_camera,
                    genesis.gaussians,
                    genesis.opt,
                    genesis.background,
                    timestep=0 if genesis.current_object_xyz is not None else None,
                    movement_sim=(
                        [genesis.current_object_xyz]
                        if genesis.current_object_xyz is not None
                        else None
                    ),
                )
                image = render_pkg["render"].detach().cpu().clamp(0, 1)
                image_np = (
                    image.permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(image_np).save(buffer, format="JPEG", quality=90)
                with _record_lock:
                    if _capture_active:
                        if _capture_writer is None:
                            height, width = image_np.shape[:2]
                            _capture_writer = cv2.VideoWriter(
                                _capture_path.as_posix(),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                20,
                                (width, height),
                            )
                            if not _capture_writer.isOpened():
                                raise RuntimeError("Failed to open preview MP4 writer")
                        _capture_writer.write(
                            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        )
                        _capture_frame_count += 1
                rendered_frame = buffer.getvalue()
                with _frame_lock:
                    phase, _ = _get_pipeline_phase()
                    if phase == "WAITING_EXPANSION":
                        _emit("frame", rendered_frame)
        except Exception as exc:
            _emit("server-state", f"Preview render skipped: {exc}")
        time.sleep(0.05)


def start_server(port):
    socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)


genesis_runtime_config = {}


def run(config, prefix=None, port=5000):
    global genesis_runtime_config, _capture_active, _capture_writer
    global _pipeline_phase, _phase_message, _scale_factor, _sam_prompt
    genesis_runtime_config = config
    config["multiview"]["enabled"] = True
    config["multiview"]["stop"] = False
    _stop_event.clear()
    _annotation_event.clear()
    _mask_review_event.clear()
    _expansion_event.clear()
    _reset_annotations()
    _pipeline_phase = "INITIALIZING"
    _phase_message = "Pipeline is initializing."
    _sam_prompt = str(config.get("environment_motion", {}).get("sam_prompt", "water"))
    _scale_factor = float(
        config.get("interaction", {}).get("environment_scale_factor", 10.0)
    )

    repo_root = Path(__file__).resolve().parent.parent
    configured_image_path = config.get("image_filepath", None)
    if configured_image_path:
        image_path = Path(configured_image_path)
    else:
        image_path = (
            Path(config.get("data_path", "examples/imgs"))
            / config["example_name"]
            / config.get("input_image", "image.png")
        )
    if not image_path.is_absolute():
        image_path = repo_root / image_path
    initial_image = Image.open(image_path).convert("RGB")
    side = min(initial_image.size)
    left = (initial_image.width - side) // 2
    top = (initial_image.height - side) // 2
    initial_image = initial_image.crop((left, top, left + side, top + side)).resize(
        (512, 512)
    )
    _set_annotation_frame(initial_image)

    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    render_thread = threading.Thread(target=render_current_scene, daemon=True)
    render_thread.start()

    try:
        genesis.run(config, dt_string=prefix)
    finally:
        _stop_event.set()
        with _record_lock:
            _capture_active = False
            if _capture_writer is not None:
                _capture_writer.release()
                _capture_writer = None


def main():
    parser = ArgumentParser(description="WonderPlay multi-view interaction")
    parser.add_argument("--config", default="examples/configs/venice_I.yaml")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    base_config_path = repo_root / "examples" / "base-config.yaml"
    if base_config_path.exists():
        config = OmegaConf.merge(
            OmegaConf.load(str(base_config_path)),
            OmegaConf.load(str(config_path)),
        )
    else:
        config = OmegaConf.load(str(config_path))

    OmegaConf.set_struct(config, False)
    if "runs_dir" not in config:
        config["runs_dir"] = config.get("work_dir", "3d_result/wonderplay")
    if "num_scenes" not in config:
        config["num_scenes"] = 1
    if "rotation_path" not in config:
        config["rotation_path"] = [0]
    if "boundary_rules" not in config:
        config["boundary_rules"] = {}
    if "use_gpt" not in config:
        config["use_gpt"] = False
    if "load_gen" not in config:
        config["load_gen"] = False
    if "multiview" not in config:
        config["multiview"] = {}
    if "enabled" not in config["multiview"]:
        config["multiview"]["enabled"] = True
    if "stop" not in config["multiview"]:
        config["multiview"]["stop"] = False
    if "expansion_iterations" not in config["multiview"]:
        config["multiview"]["expansion_iterations"] = 100
    if "environment_motion" not in config:
        config["environment_motion"] = {}
    if "sam_prompt" not in config["environment_motion"]:
        config["environment_motion"]["sam_prompt"] = "water"
    if "force_function" not in config and "force_function_name" in config:
        config["force_function"] = (
            f"simulator.genesis_functions.{config['force_function_name']}"
        )
    OmegaConf.set_struct(config, True)
    run(config, prefix=args.prefix, port=args.port)


if __name__ == "__main__":
    main()
