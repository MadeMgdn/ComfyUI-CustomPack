import base64
import io
import json
import math
import threading
import time
from typing import Any

import numpy as np
import torch
from aiohttp import web
from PIL import Image
from server import PromptServer


categoryFolder = "Custom pack/Web"

_CANVAS_STATES: dict[str, dict[str, Any]] = {}
_CANVAS_RUNTIME: dict[str, dict[str, Any]] = {}
_CANVAS_REVISIONS: dict[str, int] = {}
_CANVAS_LAST_NODE_ID: str = ""
_STATE_LOCK = threading.Lock()
_ROUTES_REGISTERED = False


def _clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
    return default


def _tensor_slice_to_pil_rgba(image_slice: torch.Tensor) -> Image.Image:
    image_np = image_slice.detach().cpu().numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255.0).astype(np.uint8)

    if image_np.ndim == 2:
        image_np = np.stack([image_np, image_np, image_np], axis=-1)

    channels = image_np.shape[-1]
    if channels == 4:
        return Image.fromarray(image_np, mode="RGBA")
    if channels == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    return Image.fromarray(image_np, mode="RGB").convert("RGBA")


def _pil_rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0)


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _data_url_to_pil(data_url: str) -> Image.Image | None:
    if not isinstance(data_url, str):
        return None
    if not data_url.startswith("data:image"):
        return None
    split_idx = data_url.find(",")
    if split_idx == -1:
        return None
    b64_data = data_url[split_idx + 1 :]
    try:
        raw = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None


def _cover_resize(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    if src_w == target_w and src_h == target_h:
        return image.copy()

    src_ratio = src_w / max(src_h, 1)
    target_ratio = target_w / max(target_h, 1)

    if src_ratio > target_ratio:
        scale = target_h / max(src_h, 1)
    else:
        scale = target_w / max(src_w, 1)

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    right = left + target_w
    bottom = top + target_h
    return resized.crop((left, top, right, bottom))


def _normalize_blend_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return "normal"
    m = mode.strip().lower()
    if m in ("normal", "source-over"):
        return "normal"
    if m == "multiply":
        return "multiply"
    if m == "screen":
        return "screen"
    if m == "overlay":
        return "overlay"
    return "normal"


def _blend_mode_rgb(dst_rgb: np.ndarray, src_rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "multiply":
        return dst_rgb * src_rgb
    if mode == "screen":
        return 1.0 - (1.0 - dst_rgb) * (1.0 - src_rgb)
    if mode == "overlay":
        low = 2.0 * dst_rgb * src_rgb
        high = 1.0 - 2.0 * (1.0 - dst_rgb) * (1.0 - src_rgb)
        return np.where(dst_rgb <= 0.5, low, high)
    return src_rgb


def _blend_rgba(dst: np.ndarray, src: np.ndarray, mode: str) -> np.ndarray:
    dst_rgb = dst[..., :3]
    dst_a = dst[..., 3:4]
    src_rgb = src[..., :3]
    src_a = src[..., 3:4]

    backdrop = np.where(dst_a > 1e-6, dst_rgb, src_rgb)
    blend_rgb = _blend_mode_rgb(backdrop, src_rgb, mode)

    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb_num = blend_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
    out_rgb = np.where(out_a > 1e-6, out_rgb_num / out_a, 0.0)

    return np.concatenate([out_rgb, out_a], axis=-1)


def _apply_levels(rgb: np.ndarray, levels: dict[str, Any]) -> np.ndarray:
    in_black = _clamp(_safe_float(levels.get("in_black", 0.0), 0.0), 0.0, 1.0)
    in_white = _clamp(_safe_float(levels.get("in_white", 1.0), 1.0), 0.0, 1.0)
    in_gamma = _clamp(_safe_float(levels.get("in_gamma", 1.0), 1.0), 0.01, 9.99)
    out_black = _clamp(_safe_float(levels.get("out_black", 0.0), 0.0), 0.0, 1.0)
    out_white = _clamp(_safe_float(levels.get("out_white", 1.0), 1.0), 0.0, 1.0)

    if in_white <= in_black:
        in_white = min(1.0, in_black + 1e-4)
    if out_white < out_black:
        out_white = out_black

    normalized = np.clip((rgb - in_black) / max(in_white - in_black, 1e-6), 0.0, 1.0)
    normalized = np.power(normalized, 1.0 / in_gamma)
    return np.clip(out_black + normalized * (out_white - out_black), 0.0, 1.0)


def _apply_hue(rgb: np.ndarray, hue_cfg: dict[str, Any]) -> np.ndarray:
    hue_shift_deg = _safe_float(hue_cfg.get("hue", 0.0), 0.0)
    sat_mult = _clamp(_safe_float(hue_cfg.get("saturation", 1.0), 1.0), 0.0, 3.0)
    light_mult = _clamp(_safe_float(hue_cfg.get("lightness", 1.0), 1.0), 0.0, 3.0)

    if abs(hue_shift_deg) < 1e-6 and abs(sat_mult - 1.0) < 1e-6 and abs(light_mult - 1.0) < 1e-6:
        return rgb

    rgb_img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    hsv_np = np.array(rgb_img.convert("HSV")).astype(np.float32)

    hue_shift = (hue_shift_deg / 360.0) * 255.0
    hsv_np[..., 0] = (hsv_np[..., 0] + hue_shift) % 255.0
    hsv_np[..., 1] = np.clip(hsv_np[..., 1] * sat_mult, 0.0, 255.0)
    hsv_np[..., 2] = np.clip(hsv_np[..., 2] * light_mult, 0.0, 255.0)

    out_rgb = np.array(Image.fromarray(hsv_np.astype(np.uint8), mode="HSV").convert("RGB")).astype(np.float32) / 255.0
    return np.clip(out_rgb, 0.0, 1.0)


def _apply_levels_hue(layer_image: Image.Image, levels: dict[str, Any], hue_cfg: dict[str, Any]) -> Image.Image:
    rgba_np = np.array(layer_image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = rgba_np[..., :3]
    alpha = rgba_np[..., 3:4]

    rgb = _apply_levels(rgb, levels)
    rgb = _apply_hue(rgb, hue_cfg)

    out = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8), mode="RGBA")


def _apply_opacity(layer_image: Image.Image, opacity: float) -> Image.Image:
    opacity = _clamp(opacity, 0.0, 1.0)
    if opacity >= 0.999:
        return layer_image
    rgba_np = np.array(layer_image.convert("RGBA")).astype(np.float32)
    rgba_np[..., 3] = np.clip(rgba_np[..., 3] * opacity, 0.0, 255.0)
    return Image.fromarray(rgba_np.astype(np.uint8), mode="RGBA")


def _affine_coeffs(width: int, height: int, transform: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    tx = _safe_float(transform.get("x", 0.0), 0.0)
    ty = _safe_float(transform.get("y", 0.0), 0.0)
    scale_x = _safe_float(transform.get("scale_x", 1.0), 1.0)
    scale_y = _safe_float(transform.get("scale_y", 1.0), 1.0)
    rotation_deg = _safe_float(transform.get("rotation", 0.0), 0.0)
    flip_x = _safe_bool(transform.get("flip_x", False), False)
    flip_y = _safe_bool(transform.get("flip_y", False), False)

    sx = max(abs(scale_x), 1e-4)
    sy = max(abs(scale_y), 1e-4)
    if flip_x:
        sx *= -1.0
    if flip_y:
        sy *= -1.0

    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    cx = width / 2.0
    cy = height / 2.0

    translate_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    scale_m = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rotate_m = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    translate_back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    translate_m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)

    forward = translate_m @ translate_back @ rotate_m @ scale_m @ translate_to_origin
    try:
        inverse = np.linalg.inv(forward)
    except np.linalg.LinAlgError:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    return (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2]),
    )


def _apply_transform(layer_image: Image.Image, transform: dict[str, Any], width: int, height: int) -> Image.Image:
    coeffs = _affine_coeffs(width, height, transform)
    return layer_image.transform(
        (width, height),
        Image.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )


def _pick_batch_item(image_tensor: torch.Tensor, batch_index: int) -> torch.Tensor:
    batch_size = image_tensor.shape[0]
    if batch_size <= 0:
        raise ValueError("Input IMAGE tensor has empty batch")
    idx = min(batch_index, batch_size - 1)
    return image_tensor[idx]


def _get_state_layer(state_layers: list[dict[str, Any]], index: int, layer_id: str) -> dict[str, Any]:
    for item in state_layers:
        if isinstance(item, dict) and str(item.get("id", "")) == layer_id:
            return item
    if 0 <= index < len(state_layers) and isinstance(state_layers[index], dict):
        return state_layers[index]
    return {}


def _compose_single(base_layers: list[Image.Image], state: dict[str, Any] | None) -> Image.Image:
    width, height = base_layers[0].size

    if not state or not isinstance(state, dict):
        state_layers = []
    else:
        state_layers = state.get("layers", [])
        if not isinstance(state_layers, list):
            state_layers = []

    result = np.zeros((height, width, 4), dtype=np.float32)

    for index in range(len(base_layers) - 1, -1, -1):
        base_layer = base_layers[index]
        layer_id = f"image_{index + 1}"
        layer_state = _get_state_layer(state_layers, index, layer_id)

        bitmap = _data_url_to_pil(layer_state.get("bitmap", "")) if isinstance(layer_state, dict) else None
        working = bitmap if bitmap is not None else base_layer
        if working.size != (width, height):
            working = _cover_resize(working.convert("RGBA"), width, height)

        if isinstance(layer_state, dict):
            levels_cfg = layer_state.get("levels", {})
            hue_cfg = layer_state.get("hue", {})
        else:
            levels_cfg = {}
            hue_cfg = {}

        if isinstance(levels_cfg, dict) or isinstance(hue_cfg, dict):
            working = _apply_levels_hue(
                working.convert("RGBA"),
                levels_cfg if isinstance(levels_cfg, dict) else {},
                hue_cfg if isinstance(hue_cfg, dict) else {},
            )

        transform_cfg = layer_state.get("transform", {}) if isinstance(layer_state, dict) else {}
        if isinstance(transform_cfg, dict):
            working = _apply_transform(working, transform_cfg, width, height)

        visible = _safe_bool(layer_state.get("visible", True), True) if isinstance(layer_state, dict) else True
        if not visible:
            continue

        opacity = _safe_float(layer_state.get("opacity", 1.0), 1.0) if isinstance(layer_state, dict) else 1.0
        opacity = _clamp(opacity, 0.0, 1.0)
        if opacity <= 0.0:
            continue

        working = _apply_opacity(working, opacity)
        src = np.array(working.convert("RGBA")).astype(np.float32) / 255.0
        mode = _normalize_blend_mode(layer_state.get("blend_mode", "normal")) if isinstance(layer_state, dict) else "normal"
        result = _blend_rgba(result, src, mode)

    return Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), mode="RGBA")


def _register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    if web is None or PromptServer is None:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    routes = server_instance.routes

    @routes.post("/cp_canvas_editor/state")
    async def cp_canvas_editor_set_state(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON payload"}, status=400)

        node_id = str(payload.get("node_id", "")).strip()
        state = payload.get("state", None)

        if not node_id:
            return web.json_response({"ok": False, "error": "node_id is required"}, status=400)
        if not isinstance(state, dict):
            return web.json_response({"ok": False, "error": "state must be object"}, status=400)

        with _STATE_LOCK:
            global _CANVAS_LAST_NODE_ID
            _CANVAS_STATES[node_id] = state
            _CANVAS_REVISIONS[node_id] = _CANVAS_REVISIONS.get(node_id, 0) + 1
            _CANVAS_LAST_NODE_ID = node_id
            revision = _CANVAS_REVISIONS[node_id]

        return web.json_response({"ok": True, "revision": revision})

    @routes.get("/cp_canvas_editor/state/{node_id}")
    async def cp_canvas_editor_get_state(request):
        node_id = str(request.match_info.get("node_id", "")).strip()
        if not node_id:
            return web.json_response({"ok": False, "error": "node_id is required"}, status=400)

        with _STATE_LOCK:
            resolved_node_id = node_id
            state = _CANVAS_STATES.get(resolved_node_id)
            runtime = _CANVAS_RUNTIME.get(resolved_node_id)
            revision = _CANVAS_REVISIONS.get(resolved_node_id, 0)

            if runtime is None:
                if _CANVAS_LAST_NODE_ID and _CANVAS_LAST_NODE_ID in _CANVAS_RUNTIME:
                    resolved_node_id = _CANVAS_LAST_NODE_ID
                    runtime = _CANVAS_RUNTIME.get(resolved_node_id)
                    if state is None:
                        state = _CANVAS_STATES.get(resolved_node_id)
                    revision = _CANVAS_REVISIONS.get(resolved_node_id, revision)
                elif len(_CANVAS_RUNTIME) == 1:
                    resolved_node_id = next(iter(_CANVAS_RUNTIME.keys()))
                    runtime = _CANVAS_RUNTIME.get(resolved_node_id)
                    if state is None:
                        state = _CANVAS_STATES.get(resolved_node_id)
                    revision = _CANVAS_REVISIONS.get(resolved_node_id, revision)

        return web.json_response(
            {
                "ok": True,
                "node_id": node_id,
                "resolved_node_id": resolved_node_id,
                "revision": revision,
                "state": state,
                "runtime": runtime,
            }
        )

    _ROUTES_REGISTERED = True


_register_routes()


class CP_CanvasLayerEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compose_layers"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, image_1, image_2, image_3=None, image_4=None, unique_id=None):
        node_id = str(unique_id) if unique_id is not None else ""
        with _STATE_LOCK:
            if node_id:
                revision = _CANVAS_REVISIONS.get(node_id, 0)
            elif _CANVAS_LAST_NODE_ID:
                revision = _CANVAS_REVISIONS.get(_CANVAS_LAST_NODE_ID, 0)
            elif len(_CANVAS_REVISIONS) == 1:
                revision = next(iter(_CANVAS_REVISIONS.values()))
            else:
                revision = 0
        return revision

    def _prepare_base_layers(
        self,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        image_3: torch.Tensor | None,
        image_4: torch.Tensor | None,
        batch_index: int,
    ) -> list[Image.Image]:
        first = _tensor_slice_to_pil_rgba(_pick_batch_item(image_1, batch_index))
        width, height = first.size

        second = _cover_resize(_tensor_slice_to_pil_rgba(_pick_batch_item(image_2, batch_index)), width, height)
        layers = [first.convert("RGBA"), second.convert("RGBA")]

        if image_3 is not None:
            third = _cover_resize(_tensor_slice_to_pil_rgba(_pick_batch_item(image_3, batch_index)), width, height)
            layers.append(third.convert("RGBA"))

        if image_4 is not None:
            fourth = _cover_resize(_tensor_slice_to_pil_rgba(_pick_batch_item(image_4, batch_index)), width, height)
            layers.append(fourth.convert("RGBA"))

        return layers

    def compose_layers(self, image_1, image_2, image_3=None, image_4=None, unique_id=None):
        global _CANVAS_LAST_NODE_ID
        _register_routes()
        node_id = str(unique_id) if unique_id is not None else ""
        batch_size = int(image_1.shape[0])
        output_tensors = []
        runtime_payload = None

        with _STATE_LOCK:
            if node_id:
                saved_state = _CANVAS_STATES.get(node_id)
            elif _CANVAS_LAST_NODE_ID:
                saved_state = _CANVAS_STATES.get(_CANVAS_LAST_NODE_ID)
            elif len(_CANVAS_STATES) == 1:
                saved_state = next(iter(_CANVAS_STATES.values()))
            else:
                saved_state = None

        for batch_index in range(batch_size):
            base_layers = self._prepare_base_layers(image_1, image_2, image_3, image_4, batch_index)
            composed = _compose_single(base_layers, saved_state)
            output_tensors.append(_pil_rgb_to_tensor(composed.convert("RGB")))

            if batch_index == 0:
                width, height = base_layers[0].size
                runtime_layers = []
                for idx, layer in enumerate(base_layers):
                    runtime_layers.append(
                        {
                            "id": f"image_{idx + 1}",
                            "name": f"Image {idx + 1}",
                            "bitmap": _pil_to_data_url(layer.convert("RGBA")),
                        }
                    )
                runtime_payload = {
                    "width": width,
                    "height": height,
                    "layers": runtime_layers,
                    "updated_at": time.time(),
                }

        if runtime_payload is not None:
            with _STATE_LOCK:
                target_node_id = node_id or _CANVAS_LAST_NODE_ID or "__last__"
                _CANVAS_RUNTIME[target_node_id] = runtime_payload
                if node_id:
                    _CANVAS_LAST_NODE_ID = node_id

        result = torch.cat(output_tensors, dim=0)
        return (result,)


NODE_CLASS_MAPPINGS_WEB_CANVASEDITOR = {
    "CP CanvasLayerEditor": CP_CanvasLayerEditor,
}

NODE_DISPLAY_NAME_MAPPINGS_WEB_CANVASEDITOR = {
    "CP CanvasLayerEditor": "Canvas layer editor",
}
