import gc
import logging

import comfy.model_management as model_management
import torch


categoryFolder = "Custom pack/Imported/EasyUse"

LOG = logging.getLogger(__name__)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def clean_vram_used_force():
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (AttributeError, OSError, RuntimeError) as exc:
        LOG.debug("Skipping CUDA synchronize during Clean VRAM Used: %s", exc)

    model_management.unload_all_models()
    model_management.soft_empty_cache()


class CP_EasyUseCleanVRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (ANY_TYPE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("output",)
    FUNCTION = "clean"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    def clean(self, anything, **kwargs):
        clean_vram_used_force()
        return (anything,)


NODE_CLASS_MAPPINGS_IMPORTED_EASYUSE_CLEANVRAM = {
    "easy cleanGpuUsed": CP_EasyUseCleanVRAM,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_EASYUSE_CLEANVRAM = {
    "easy cleanGpuUsed": "Clean VRAM Used",
}
