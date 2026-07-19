categoryFolder = "Custom pack/Converter/Object"


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


class CP_AnythingToModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToVae:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_AnythingToGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, anything):
        return (anything,)


class CP_ModelToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, model):
        return (model,)


class CP_ClipToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, clip):
        return (clip,)


class CP_VaeToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, vae):
        return (vae,)


class CP_ConditioningToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, conditioning):
        return (conditioning,)


class CP_LatentToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, latent):
        return (latent,)


class CP_ImageToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, image):
        return (image,)


class CP_MaskToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, mask):
        return (mask,)


class CP_NoiseToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, noise):
        return (noise,)


class CP_SamplerToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler": ("SAMPLER", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, sampler):
        return (sampler,)


class CP_SigmasToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, sigmas):
        return (sigmas,)


class CP_GuiderToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"guider": ("GUIDER", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, guider):
        return (guider,)


NODE_CLASS_MAPPINGS_CONVERTEROBJECT = {
    "CP Anything To Model": CP_AnythingToModel,
    "CP Anything To Clip": CP_AnythingToClip,
    "CP Anything To Vae": CP_AnythingToVae,
    "CP Anything To Conditioning": CP_AnythingToConditioning,
    "CP Anything To Latent": CP_AnythingToLatent,
    "CP Anything To Image": CP_AnythingToImage,
    "CP Anything To Mask": CP_AnythingToMask,
    "CP Anything To Noise": CP_AnythingToNoise,
    "CP Anything To Sampler": CP_AnythingToSampler,
    "CP Anything To Sigmas": CP_AnythingToSigmas,
    "CP Anything To Guider": CP_AnythingToGuider,
    "CP Model To Anything": CP_ModelToAnything,
    "CP Clip To Anything": CP_ClipToAnything,
    "CP Vae To Anything": CP_VaeToAnything,
    "CP Conditioning To Anything": CP_ConditioningToAnything,
    "CP Latent To Anything": CP_LatentToAnything,
    "CP Image To Anything": CP_ImageToAnything,
    "CP Mask To Anything": CP_MaskToAnything,
    "CP Noise To Anything": CP_NoiseToAnything,
    "CP Sampler To Anything": CP_SamplerToAnything,
    "CP Sigmas To Anything": CP_SigmasToAnything,
    "CP Guider To Anything": CP_GuiderToAnything,
}

NODE_DISPLAY_NAME_MAPPINGS_CONVERTEROBJECT = {
    "CP Anything To Model": "CP Anything To Model",
    "CP Anything To Clip": "CP Anything To Clip",
    "CP Anything To Vae": "CP Anything To Vae",
    "CP Anything To Conditioning": "CP Anything To Conditioning",
    "CP Anything To Latent": "CP Anything To Latent",
    "CP Anything To Image": "CP Anything To Image",
    "CP Anything To Mask": "CP Anything To Mask",
    "CP Anything To Noise": "CP Anything To Noise",
    "CP Anything To Sampler": "CP Anything To Sampler",
    "CP Anything To Sigmas": "CP Anything To Sigmas",
    "CP Anything To Guider": "CP Anything To Guider",
    "CP Model To Anything": "CP Model To Anything",
    "CP Clip To Anything": "CP Clip To Anything",
    "CP Vae To Anything": "CP Vae To Anything",
    "CP Conditioning To Anything": "CP Conditioning To Anything",
    "CP Latent To Anything": "CP Latent To Anything",
    "CP Image To Anything": "CP Image To Anything",
    "CP Mask To Anything": "CP Mask To Anything",
    "CP Noise To Anything": "CP Noise To Anything",
    "CP Sampler To Anything": "CP Sampler To Anything",
    "CP Sigmas To Anything": "CP Sigmas To Anything",
    "CP Guider To Anything": "CP Guider To Anything",
}
