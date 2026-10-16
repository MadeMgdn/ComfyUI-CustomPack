import logging
import os

import folder_paths
from aiohttp import web
from nodes import LoraLoader, PreviewImage
from server import PromptServer


categoryFolder = "Custom pack/Imported/Rgthree"


LOG = logging.getLogger(__name__)
POWER_LORA_LOADER_NODE_NAME = "Power Lora Loader (rgthree)"
_POWER_LORA_ROUTES_REGISTERED = False


class AnyType(str):
    def __ne__(self, _value):
        return False


class FlexibleOptionalInputType(dict):
    def __init__(self, input_type, data=None):
        super().__init__()
        self.input_type = input_type
        self.data = data or {}
        self.update(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.input_type,)

    def __contains__(self, _key):
        return True


ANY_TYPE = AnyType("*")


def _get_lora_filenames():
    try:
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


def _find_lora_filename(file_path):
    if not file_path:
        return None

    lora_paths = _get_lora_filenames()
    if file_path in lora_paths:
        return file_path

    requested_no_ext = os.path.splitext(file_path)[0]
    paths_no_ext = [os.path.splitext(path)[0] for path in lora_paths]
    if requested_no_ext in paths_no_ext:
        return lora_paths[paths_no_ext.index(requested_no_ext)]

    requested_name = os.path.basename(file_path)
    requested_name_no_ext = os.path.splitext(requested_name)[0]
    names = [os.path.basename(path) for path in lora_paths]
    names_no_ext = [os.path.splitext(name)[0] for name in names]

    if requested_name in names:
        return lora_paths[names.index(requested_name)]
    if requested_name_no_ext in names_no_ext:
        return lora_paths[names_no_ext.index(requested_name_no_ext)]

    matches = [path for path in lora_paths if file_path in path]
    if len(matches) == 1:
        LOG.warning(
            "[%s] Fuzzy-matched LoRA '%s' to '%s'.",
            POWER_LORA_LOADER_NODE_NAME,
            file_path,
            matches[0],
        )
        return matches[0]

    LOG.warning(
        "[%s] LoRA '%s' was not found; skipping it.",
        POWER_LORA_LOADER_NODE_NAME,
        file_path,
    )
    return None


def _register_power_lora_routes():
    global _POWER_LORA_ROUTES_REGISTERED

    if _POWER_LORA_ROUTES_REGISTERED or web is None or PromptServer is None:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.get("/cp_rgthree_power_lora_loader/loras")
    async def get_power_lora_loader_loras(_request):
        return web.json_response(_get_lora_filenames())

    _POWER_LORA_ROUTES_REGISTERED = True


_register_power_lora_routes()


class CP_RgthreeFastBypasser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(input_type=ANY_TYPE),
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("OPT_CONNECTION",)
    FUNCTION = "noop"
    CATEGORY = categoryFolder

    def noop(self, **_kwargs):
        return (None,)


class CP_RgthreeFastGroupsBypasser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("OPT_CONNECTION",)
    FUNCTION = "noop"
    CATEGORY = categoryFolder

    def noop(self):
        return (None,)


class CP_RgthreeFastGroupsMuter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("OPT_CONNECTION",)
    FUNCTION = "noop"
    CATEGORY = categoryFolder

    def noop(self):
        return (None,)


class CP_RgthreeImageComparer(PreviewImage):
    """Compares two images in the ComfyUI frontend."""

    CATEGORY = categoryFolder
    FUNCTION = "compare_images"
    DESCRIPTION = "Compares two images with a hover slider, or click from properties."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def compare_images(
        self,
        image_a=None,
        image_b=None,
        filename_prefix="rgthree.compare.",
        prompt=None,
        extra_pnginfo=None,
    ):
        result = {"ui": {"a_images": [], "b_images": []}}

        if image_a is not None and len(image_a) > 0:
            result["ui"]["a_images"] = self.save_images(
                image_a,
                filename_prefix,
                prompt,
                extra_pnginfo,
            )["ui"]["images"]

        if image_b is not None and len(image_b) > 0:
            result["ui"]["b_images"] = self.save_images(
                image_b,
                filename_prefix,
                prompt,
                extra_pnginfo,
            )["ui"]["images"]

        result["result"] = (image_a if image_a is not None else image_b,)
        return result


class CP_RgthreePowerLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        _register_power_lora_routes()
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(
                input_type=ANY_TYPE,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Loads any number of LoRAs into an optional model and CLIP. LoRAs can be enabled, "
        "disabled, reordered, and assigned either one shared strength or separate model and "
        "CLIP strengths."
    )

    def load_loras(self, model=None, clip=None, **kwargs):
        for key, value in kwargs.items():
            if not key.upper().startswith("LORA_") or not isinstance(value, dict):
                continue
            if not {"on", "lora", "strength"}.issubset(value):
                continue

            try:
                strength_model = float(value["strength"])
                strength_clip_value = value.get("strengthTwo")
                strength_clip = (
                    float(strength_clip_value)
                    if strength_clip_value is not None
                    else strength_model
                )
            except (TypeError, ValueError):
                LOG.warning(
                    "[%s] Invalid strengths in '%s'; skipping it.",
                    POWER_LORA_LOADER_NODE_NAME,
                    key,
                )
                continue

            if clip is None:
                strength_clip = 0.0

            if not value["on"] or (strength_model == 0 and strength_clip == 0):
                continue

            lora_name = _find_lora_filename(value["lora"])
            if model is not None and lora_name is not None:
                model, clip = LoraLoader().load_lora(
                    model,
                    clip,
                    lora_name,
                    strength_model,
                    strength_clip,
                )

        return (model, clip)


NODE_CLASS_MAPPINGS_IMPORTED_RGTHREE_RGTHREE = {
    "Fast Bypasser (rgthree)": CP_RgthreeFastBypasser,
    "Fast Groups Bypasser (rgthree)": CP_RgthreeFastGroupsBypasser,
    "Fast Groups Muter (rgthree)": CP_RgthreeFastGroupsMuter,
    "Image Comparer (rgthree)": CP_RgthreeImageComparer,
    POWER_LORA_LOADER_NODE_NAME: CP_RgthreePowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_RGTHREE_RGTHREE = {
    "Fast Bypasser (rgthree)": "Fast Bypasser (rgthree)",
    "Fast Groups Bypasser (rgthree)": "Fast Groups Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)": "Fast Groups Muter (rgthree)",
    "Image Comparer (rgthree)": "Image Comparer (rgthree)",
    POWER_LORA_LOADER_NODE_NAME: POWER_LORA_LOADER_NODE_NAME,
}
