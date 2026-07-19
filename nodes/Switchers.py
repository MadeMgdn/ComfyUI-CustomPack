categoryFolder = "Custom pack/Switchers/Byflag"


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


class _SwitchByFlagBase:
    TYPE = None
    RETURN_NAME = "value"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": (cls.TYPE, {"forceInput": True}),
                "on_false": (cls.TYPE, {"forceInput": True}),
                "flag": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "switch"
    CATEGORY = categoryFolder

    def switch(self, on_true, on_false, flag):
        return (on_true if flag else on_false,)


class CP_SwitchStringByFlag(_SwitchByFlagBase):
    TYPE = "STRING"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


class CP_SwitchIntByFlag(_SwitchByFlagBase):
    TYPE = "INT"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)


class CP_SwitchFloatByFlag(_SwitchByFlagBase):
    TYPE = "FLOAT"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)


class CP_SwitchBooleanByFlag(_SwitchByFlagBase):
    TYPE = "BOOLEAN"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)


class CP_SwitchModelByFlag(_SwitchByFlagBase):
    TYPE = "MODEL"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)


class CP_SwitchClipByFlag(_SwitchByFlagBase):
    TYPE = "CLIP"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)


class CP_SwitchVaeByFlag(_SwitchByFlagBase):
    TYPE = "VAE"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)


class CP_SwitchConditioningByFlag(_SwitchByFlagBase):
    TYPE = "CONDITIONING"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)


class CP_SwitchLatentByFlag(_SwitchByFlagBase):
    TYPE = "LATENT"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)


class CP_SwitchImageByFlag(_SwitchByFlagBase):
    TYPE = "IMAGE"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)


class CP_SwitchMaskByFlag(_SwitchByFlagBase):
    TYPE = "MASK"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)


class CP_SwitchNoiseByFlag(_SwitchByFlagBase):
    TYPE = "NOISE"
    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)


class CP_SwitchSamplerByFlag(_SwitchByFlagBase):
    TYPE = "SAMPLER"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)


class CP_SwitchSigmasByFlag(_SwitchByFlagBase):
    TYPE = "SIGMAS"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)


class CP_SwitchGuiderByFlag(_SwitchByFlagBase):
    TYPE = "GUIDER"
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)


class CP_SwitchAnythingByFlag(_SwitchByFlagBase):
    TYPE = ANY_TYPE
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)


NODE_CLASS_MAPPINGS_SWITCHERS = {
    "CP Switch String By Flag": CP_SwitchStringByFlag,
    "CP Switch Int By Flag": CP_SwitchIntByFlag,
    "CP Switch Float By Flag": CP_SwitchFloatByFlag,
    "CP Switch Boolean By Flag": CP_SwitchBooleanByFlag,
    "CP Switch Model By Flag": CP_SwitchModelByFlag,
    "CP Switch Clip By Flag": CP_SwitchClipByFlag,
    "CP Switch Vae By Flag": CP_SwitchVaeByFlag,
    "CP Switch Conditioning By Flag": CP_SwitchConditioningByFlag,
    "CP Switch Latent By Flag": CP_SwitchLatentByFlag,
    "CP Switch Image By Flag": CP_SwitchImageByFlag,
    "CP Switch Mask By Flag": CP_SwitchMaskByFlag,
    "CP Switch Noise By Flag": CP_SwitchNoiseByFlag,
    "CP Switch Sampler By Flag": CP_SwitchSamplerByFlag,
    "CP Switch Sigmas By Flag": CP_SwitchSigmasByFlag,
    "CP Switch Guider By Flag": CP_SwitchGuiderByFlag,
    "CP Switch Anything By Flag": CP_SwitchAnythingByFlag,
}

NODE_DISPLAY_NAME_MAPPINGS_SWITCHERS = {
    "CP Switch String By Flag": "CP Switch String By Flag",
    "CP Switch Int By Flag": "CP Switch Int By Flag",
    "CP Switch Float By Flag": "CP Switch Float By Flag",
    "CP Switch Boolean By Flag": "CP Switch Boolean By Flag",
    "CP Switch Model By Flag": "CP Switch Model By Flag",
    "CP Switch Clip By Flag": "CP Switch Clip By Flag",
    "CP Switch Vae By Flag": "CP Switch Vae By Flag",
    "CP Switch Conditioning By Flag": "CP Switch Conditioning By Flag",
    "CP Switch Latent By Flag": "CP Switch Latent By Flag",
    "CP Switch Image By Flag": "CP Switch Image By Flag",
    "CP Switch Mask By Flag": "CP Switch Mask By Flag",
    "CP Switch Noise By Flag": "CP Switch Noise By Flag",
    "CP Switch Sampler By Flag": "CP Switch Sampler By Flag",
    "CP Switch Sigmas By Flag": "CP Switch Sigmas By Flag",
    "CP Switch Guider By Flag": "CP Switch Guider By Flag",
    "CP Switch Anything By Flag": "CP Switch Anything By Flag",
}
