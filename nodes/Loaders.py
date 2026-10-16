import os

import comfy.sd
import comfy.utils
import folder_paths
import torch
from aiohttp import web
from server import PromptServer


categoryFolder = "Custom pack/Loaders"

_ROUTES_REGISTERED = False
_MODEL_FOLDERS = {
    "lora": ("loras",),
    "checkpoint": ("checkpoints",),
    "diffusion_model": ("diffusion_models", "unet"),
}
_PREVIEW_SUFFIXES = (
    ".preview.png",
    ".preview.jpg",
    ".preview.jpeg",
    ".preview.webp",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)
_DESCRIPTION_SUFFIXES = (
    ".description.txt",
)


def _get_filename_list(model_type):
    result = []
    seen = set()

    for folder_name in _MODEL_FOLDERS.get(model_type, ()):
        try:
            names = folder_paths.get_filename_list(folder_name)
        except Exception:
            continue

        for name in names:
            if name not in seen:
                result.append(name)
                seen.add(name)

    return result


def _get_model_path(model_type, model_name):
    if not model_name:
        return None

    for folder_name in _MODEL_FOLDERS.get(model_type, ()):
        try:
            return folder_paths.get_full_path_or_raise(folder_name, model_name)
        except Exception:
            try:
                path = folder_paths.get_full_path(folder_name, model_name)
            except Exception:
                path = None

        if path:
            return path

    return None


def _find_sidecar_path(model_path, suffixes):
    if not model_path:
        return None

    base_path = os.path.splitext(model_path)[0]
    for suffix in suffixes:
        sidecar_path = base_path + suffix
        if os.path.isfile(sidecar_path):
            return sidecar_path
    return None


def _find_preview_path(model_type, model_name):
    return _find_sidecar_path(_get_model_path(model_type, model_name), _PREVIEW_SUFFIXES)


def _find_description_path(model_type, model_name):
    return _find_sidecar_path(_get_model_path(model_type, model_name), _DESCRIPTION_SUFFIXES)


def _read_description(path):
    if not path:
        return ""

    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with open(path, "r", encoding=encoding) as description_file:
                return description_file.read()
        except UnicodeDecodeError:
            continue

    with open(path, "r", encoding="utf-8", errors="replace") as description_file:
        return description_file.read()


def _sidecar_index(model_type):
    result = {}
    for model_name in _get_filename_list(model_type):
        model_path = _get_model_path(model_type, model_name)
        if not model_path:
            continue

        preview_path = _find_sidecar_path(model_path, _PREVIEW_SUFFIXES)
        description_path = _find_sidecar_path(model_path, _DESCRIPTION_SUFFIXES)
        if preview_path or description_path:
            result[model_name] = {
                "preview": preview_path is not None,
                "description": description_path is not None,
            }
    return result


def _json_sidecars_response(model_type):
    if model_type not in _MODEL_FOLDERS:
        return web.Response(status=404)
    return web.json_response(_sidecar_index(model_type))


def _preview_response(model_type, model_name):
    if model_type not in _MODEL_FOLDERS:
        return web.Response(status=404)

    preview_path = _find_preview_path(model_type, model_name)
    if not preview_path:
        return web.Response(status=404)

    filename = os.path.basename(preview_path)
    return web.FileResponse(
        preview_path,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


def _description_response(model_type, model_name):
    if model_type not in _MODEL_FOLDERS:
        return web.Response(status=404)

    description_path = _find_description_path(model_type, model_name)
    if not description_path:
        return web.Response(text="", content_type="text/plain", charset="utf-8")

    return web.Response(
        text=_read_description(description_path),
        content_type="text/plain",
        charset="utf-8",
    )


def _register_routes():
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED or web is None or PromptServer is None:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    routes = server_instance.routes

    @routes.get("/cp_custom_model_loader/{model_type}/sidecars")
    async def get_model_sidecars(request):
        return _json_sidecars_response(request.match_info["model_type"])

    @routes.get("/cp_custom_model_loader/{model_type}/preview/{model_name:.+}")
    async def get_model_preview(request):
        return _preview_response(request.match_info["model_type"], request.match_info["model_name"])

    @routes.get("/cp_custom_model_loader/{model_type}/description/{model_name:.+}")
    async def get_model_description(request):
        return _description_response(request.match_info["model_type"], request.match_info["model_name"])

    @routes.get("/cp_custom_lora_loader/sidecars")
    async def get_lora_sidecars(_request):
        return _json_sidecars_response("lora")

    @routes.get("/cp_custom_lora_loader/preview/{lora_name:.+}")
    async def get_lora_preview(request):
        return _preview_response("lora", request.match_info["lora_name"])

    @routes.get("/cp_custom_lora_loader/description/{lora_name:.+}")
    async def get_lora_description(request):
        return _description_response("lora", request.match_info["lora_name"])

    _ROUTES_REGISTERED = True


_register_routes()


class CP_CustomLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (_get_filename_list("lora"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = categoryFolder

    def load_lora_model_only(self, model, lora_name, strength_model):
        _register_routes()

        if strength_model == 0:
            return (model,)

        lora_path = _get_model_path("lora", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, _clip_lora = comfy.sd.load_lora_for_models(
            model,
            None,
            lora,
            strength_model,
            0,
        )
        return (model_lora,)


class CP_CustomLoadCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    _get_filename_list("checkpoint"),
                    {"tooltip": "The name of the checkpoint model to load."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = categoryFolder

    def load_checkpoint(self, ckpt_name):
        _register_routes()

        ckpt_path = _get_model_path("checkpoint", ckpt_name)
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class CP_CustomLoadDiffusionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_get_filename_list("diffusion_model"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"advanced": True},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = categoryFolder

    def load_unet(self, unet_name, weight_dtype):
        _register_routes()

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = _get_model_path("diffusion_model", unet_name)
        if not unet_path:
            raise FileNotFoundError(f"Diffusion model not found: {unet_name}")

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


NODE_CLASS_MAPPINGS_LOADERS = {
    "CP_CustomLoraLoader": CP_CustomLoraLoader,
    "CP_CustomLoadCheckpoint": CP_CustomLoadCheckpoint,
    "CP_CustomLoadDiffusionModel": CP_CustomLoadDiffusionModel,
}

NODE_DISPLAY_NAME_MAPPINGS_LOADERS = {
    "CP_CustomLoraLoader": "ComfyUI Custom Lora Loader",
    "CP_CustomLoadCheckpoint": "ComfyUI Custom Load Checkpoint",
    "CP_CustomLoadDiffusionModel": "ComfyUI Custom Load Diffusion Model",
}
