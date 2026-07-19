import torch


categoryFolder = "Custom pack/Switchers/First"


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def _has_content(value):
    if value is None:
        return False
    if hasattr(value, "nelement"):
        try:
            return value.nelement() > 0
        except TypeError:
            return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def _is_not_empty_string(value):
    return value is not None and value != ""


def _is_positive_number(value):
    return value is not None and value > 0


def _is_true(value):
    return value is True


def _is_not_none(value):
    return value is not None


class _SwitchFirstBase:
    TYPE = None
    CHECK = staticmethod(_has_content)
    FALLBACK = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "value_1": (cls.TYPE, {"default": None, "forceInput": True}),
                "value_2": (cls.TYPE, {"default": None, "forceInput": True}),
                "value_3": (cls.TYPE, {"default": None, "forceInput": True}),
                "value_4": (cls.TYPE, {"default": None, "forceInput": True}),
            },
        }

    FUNCTION = "switch"
    CATEGORY = categoryFolder

    def switch(self, value_1=None, value_2=None, value_3=None, value_4=None):
        for value in (value_1, value_2, value_3, value_4):
            if self.CHECK(value):
                return (value,)
        return (self.get_fallback(),)

    def get_fallback(self):
        return self.FALLBACK


class CP_SwitchFirstString(_SwitchFirstBase):
    TYPE = "STRING"
    CHECK = staticmethod(_is_not_empty_string)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)


class CP_SwitchFirstInt(_SwitchFirstBase):
    TYPE = "INT"
    CHECK = staticmethod(_is_positive_number)
    FALLBACK = 0
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)


class CP_SwitchFirstFloat(_SwitchFirstBase):
    TYPE = "FLOAT"
    CHECK = staticmethod(_is_positive_number)
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)


class CP_SwitchFirstBoolean(_SwitchFirstBase):
    TYPE = "BOOLEAN"
    CHECK = staticmethod(_is_true)
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)


class CP_SwitchFirstModel(_SwitchFirstBase):
    TYPE = "MODEL"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)


class CP_SwitchFirstClip(_SwitchFirstBase):
    TYPE = "CLIP"
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)


class CP_SwitchFirstVae(_SwitchFirstBase):
    TYPE = "VAE"
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)


class CP_SwitchFirstConditioning(_SwitchFirstBase):
    TYPE = "CONDITIONING"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)


class CP_SwitchFirstLatent(_SwitchFirstBase):
    TYPE = "LATENT"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    def get_fallback(self):
        return {"samples": torch.zeros([1, 4, 8, 8])}


class CP_SwitchFirstImage(_SwitchFirstBase):
    TYPE = "IMAGE"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    def get_fallback(self):
        return torch.zeros(1, 64, 64, 3)


class CP_SwitchFirstMask(_SwitchFirstBase):
    TYPE = "MASK"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)


class CP_SwitchFirstNoise(_SwitchFirstBase):
    TYPE = "NOISE"
    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)


class CP_SwitchFirstSampler(_SwitchFirstBase):
    TYPE = "SAMPLER"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)


class CP_SwitchFirstSigmas(_SwitchFirstBase):
    TYPE = "SIGMAS"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)


class CP_SwitchFirstGuider(_SwitchFirstBase):
    TYPE = "GUIDER"
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)


class CP_SwitchFirstAnything(_SwitchFirstBase):
    TYPE = ANY_TYPE
    CHECK = staticmethod(_is_not_none)
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)


NODE_CLASS_MAPPINGS_SWITCHFIRST = {
    "CP Switch First String": CP_SwitchFirstString,
    "CP Switch First Int": CP_SwitchFirstInt,
    "CP Switch First Float": CP_SwitchFirstFloat,
    "CP Switch First Boolean": CP_SwitchFirstBoolean,
    "CP Switch First Model": CP_SwitchFirstModel,
    "CP Switch First Clip": CP_SwitchFirstClip,
    "CP Switch First Vae": CP_SwitchFirstVae,
    "CP Switch First Conditioning": CP_SwitchFirstConditioning,
    "CP Switch First Latent": CP_SwitchFirstLatent,
    "CP Switch First Image": CP_SwitchFirstImage,
    "CP Switch First Mask": CP_SwitchFirstMask,
    "CP Switch First Noise": CP_SwitchFirstNoise,
    "CP Switch First Sampler": CP_SwitchFirstSampler,
    "CP Switch First Sigmas": CP_SwitchFirstSigmas,
    "CP Switch First Guider": CP_SwitchFirstGuider,
    "CP Switch First Anything": CP_SwitchFirstAnything,
}

NODE_DISPLAY_NAME_MAPPINGS_SWITCHFIRST = {
    "CP Switch First String": "CP Switch First String",
    "CP Switch First Int": "CP Switch First Int",
    "CP Switch First Float": "CP Switch First Float",
    "CP Switch First Boolean": "CP Switch First Boolean",
    "CP Switch First Model": "CP Switch First Model",
    "CP Switch First Clip": "CP Switch First Clip",
    "CP Switch First Vae": "CP Switch First Vae",
    "CP Switch First Conditioning": "CP Switch First Conditioning",
    "CP Switch First Latent": "CP Switch First Latent",
    "CP Switch First Image": "CP Switch First Image",
    "CP Switch First Mask": "CP Switch First Mask",
    "CP Switch First Noise": "CP Switch First Noise",
    "CP Switch First Sampler": "CP Switch First Sampler",
    "CP Switch First Sigmas": "CP Switch First Sigmas",
    "CP Switch First Guider": "CP Switch First Guider",
    "CP Switch First Anything": "CP Switch First Anything",
}
