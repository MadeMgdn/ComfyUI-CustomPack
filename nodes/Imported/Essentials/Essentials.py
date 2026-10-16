import torch
import torch.nn.functional as F
import comfy.utils
from nodes import MAX_RESOLUTION


categoryFolder = "Custom pack/Imported/Essentials"


class CP_EssentialsImageFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "length": ("INT", {"default": -1, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = categoryFolder

    def execute(self, image, start, length):
        if length < 0:
            length = image.shape[0]

        start = min(start, image.shape[0] - 1)
        length = min(image.shape[0] - start, length)
        return (image[start:start + length],)


class CP_EssentialsImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1},
                ),
                "interpolation": (
                    ["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],
                ),
                "method": (
                    ["stretch", "keep proportion", "fill / crop", "pad"],
                ),
                "condition": (
                    [
                        "always",
                        "downscale if bigger",
                        "upscale if smaller",
                        "if bigger area",
                        "if smaller area",
                    ],
                ),
                "multiple_of": (
                    "INT",
                    {"default": 0, "min": 0, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "execute"
    CATEGORY = categoryFolder

    def execute(
        self,
        image,
        width,
        height,
        method="stretch",
        interpolation="nearest",
        condition="always",
        multiple_of=0,
        keep_proportion=False,
    ):
        _, original_height, original_width, _ = image.shape
        crop_x = crop_y = crop_x2 = crop_y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width -= width % multiple_of
            height -= height % multiple_of

        if method in ("keep proportion", "pad"):
            if width == 0 and original_height < height:
                width = MAX_RESOLUTION
            elif width == 0 and original_height >= height:
                width = original_width

            if height == 0 and original_width < width:
                height = MAX_RESOLUTION
            elif height == 0 and original_width >= width:
                height = original_height

            ratio = min(width / original_width, height / original_height)
            new_width = round(original_width * ratio)
            new_height = round(original_height * ratio)

            if method == "pad":
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith("fill"):
            width = width if width > 0 else original_width
            height = height if height > 0 else original_height

            ratio = max(width / original_width, height / original_height)
            new_width = round(original_width * ratio)
            new_height = round(original_height * ratio)
            crop_x = (new_width - width) // 2
            crop_y = (new_height - height) // 2
            crop_x2 = crop_x + width
            crop_y2 = crop_y + height

            if crop_x2 > new_width:
                crop_x -= crop_x2 - new_width
            if crop_x < 0:
                crop_x = 0
            if crop_y2 > new_height:
                crop_y -= crop_y2 - new_height
            if crop_y < 0:
                crop_y = 0

            width = new_width
            height = new_height
        else:
            width = width if width > 0 else original_width
            height = height if height > 0 else original_height

        should_resize = (
            "always" in condition
            or (
                condition == "downscale if bigger"
                and (original_height > height or original_width > width)
            )
            or (
                condition == "upscale if smaller"
                and (original_height < height or original_width < width)
            )
            or (
                "bigger area" in condition
                and original_height * original_width > height * width
            )
            or (
                "smaller area" in condition
                and original_height * original_width < height * width
            )
        )

        if should_resize:
            output = image.permute(0, 3, 1, 2)

            if interpolation == "lanczos":
                output = comfy.utils.lanczos(output, width, height)
            else:
                output = F.interpolate(output, size=(height, width), mode=interpolation)

            if method == "pad" and any(
                value > 0 for value in (pad_left, pad_right, pad_top, pad_bottom)
            ):
                output = F.pad(
                    output,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    value=0,
                )

            output = output.permute(0, 2, 3, 1)

            if method.startswith("fill") and any(
                value > 0 for value in (crop_x, crop_y, crop_x2, crop_y2)
            ):
                output = output[:, crop_y:crop_y2, crop_x:crop_x2, :]
        else:
            output = image

        if multiple_of > 1 and (
            output.shape[2] % multiple_of != 0
            or output.shape[1] % multiple_of != 0
        ):
            output_width = output.shape[2]
            output_height = output.shape[1]
            crop_x = (output_width % multiple_of) // 2
            crop_y = (output_height % multiple_of) // 2
            crop_x2 = output_width - ((output_width % multiple_of) - crop_x)
            crop_y2 = output_height - ((output_height % multiple_of) - crop_y)
            output = output[:, crop_y:crop_y2, crop_x:crop_x2, :]

        output = torch.clamp(output, 0, 1)
        return (output, output.shape[2], output.shape[1])


NODE_CLASS_MAPPINGS_IMPORTED_ESSENTIALS_ESSENTIALS = {
    "ImageFromBatch+": CP_EssentialsImageFromBatch,
    "ImageResize+": CP_EssentialsImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_ESSENTIALS_ESSENTIALS = {
    "ImageFromBatch+": "Image From Batch",
    "ImageResize+": "Image Resize",
}
