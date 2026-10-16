import logging
import math

import nvvfx
import torch
import torch.nn.functional as F
from PIL import ImageColor
from comfy import model_management
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION
from server import PromptServer


categoryFolder = "Custom pack/Imported/KJNodes"


def _string_to_color(color_string):
    color_list = [0, 0, 0]
    color_string = str(color_string).strip()

    if "," in color_string:
        try:
            values = [float(channel.strip()) for channel in color_string.split(",")]
            color_list = [int(value * 255) for value in values] if all(0 <= value <= 1 for value in values) else [int(value) for value in values]
        except ValueError:
            logging.warning("Invalid color format: %s. Using default black.", color_string)
    elif color_string.startswith("#") or (
        color_string.lstrip("#").isalnum()
        and not color_string.lstrip("#").replace(".", "", 1).isdigit()
    ):
        color_value = color_string.lstrip("#")
        if len(color_value) in (6, 8) and all(char in "0123456789abcdefABCDEF" for char in color_value):
            color_list = [int(color_value[index:index + 2], 16) for index in range(0, len(color_value), 2)]
        else:
            try:
                color_list = list(ImageColor.getrgb(color_string))
            except ValueError:
                logging.warning("Invalid color name or hex value: %s. Using default black.", color_string)
    else:
        try:
            value = float(color_string)
            value = int(value * 255) if 0 <= value <= 1 else int(value)
            color_list = [value, value, value]
        except ValueError:
            logging.warning("Invalid color format: %s. Using default black.", color_string)

    return [max(0, min(255, value)) for value in color_list]


def _gaussian_blur_nchw(image, sigma_px):
    if sigma_px <= 0:
        return image

    radius = max(1, int(3.0 * float(sigma_px)))
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel = torch.exp(-(coordinates * coordinates) / (2.0 * float(sigma_px) ** 2))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    image = F.conv2d(image, kernel_x, padding=(0, radius), groups=channels)
    return F.conv2d(image, kernel_y, padding=(radius, 0), groups=channels)


def _pad_image(image, left, right, top, bottom, color, pad_mode, mask=None):
    batch, height, width, channels = image.shape

    if mask is not None and mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest-exact").squeeze(1)

    color_values = _string_to_color(color)
    if len(color_values) == 1:
        color_values *= 3
    if channels > len(color_values):
        color_values.extend([255] * (channels - len(color_values)))
    background = torch.tensor(
        color_values[:channels],
        dtype=image.dtype,
        device=image.device,
    ) / 255.0

    padded_width = width + left + right
    padded_height = height + top + bottom

    if pad_mode == "pillarbox_blur":
        out_image = torch.zeros(
            (batch, padded_height, padded_width, channels),
            dtype=image.dtype,
            device=image.device,
        )
        for batch_index in range(batch):
            scale_fill = max(padded_width / float(width), padded_height / float(height))
            background_width = max(1, int(round(width * scale_fill)))
            background_height = max(1, int(round(height * scale_fill)))
            source = image[batch_index].movedim(-1, 0).unsqueeze(0)
            blurred = common_upscale(
                source,
                background_width,
                background_height,
                "bilinear",
                crop="disabled",
            )
            y0 = max(0, (background_height - padded_height) // 2)
            x0 = max(0, (background_width - padded_width) // 2)
            blurred = blurred[:, :, y0:y0 + padded_height, x0:x0 + padded_width]

            if blurred.shape[2:] != (padded_height, padded_width):
                missing_height = padded_height - blurred.shape[2]
                missing_width = padded_width - blurred.shape[3]
                fix_top = max(0, missing_height // 2)
                fix_bottom = max(0, missing_height - fix_top)
                fix_left = max(0, missing_width // 2)
                fix_right = max(0, missing_width - fix_left)
                blurred = F.pad(
                    blurred,
                    (fix_left, fix_right, fix_top, fix_bottom),
                    mode="replicate",
                )

            sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
            blurred = _gaussian_blur_nchw(blurred, sigma)
            if channels >= 3:
                red, green, blue = blurred[:, 0:1], blurred[:, 1:2], blurred[:, 2:3]
                luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                gray = torch.cat([luma, luma, luma], dim=1)
                rgb = torch.cat([red, green, blue], dim=1)
                blurred[:, 0:3] = rgb * 0.8 + gray * 0.2
            out_image[batch_index] = torch.clamp(blurred * 0.35, 0.0, 1.0).squeeze(0).movedim(0, -1)

        out_image[:, top:top + height, left:left + width, :] = image
        out_mask = torch.ones(
            (batch, padded_height, padded_width),
            dtype=image.dtype,
            device=image.device,
        )
        if mask is not None:
            out_mask[:, top:top + height, left:left + width] = mask
        else:
            out_mask[:, top:top + height, left:left + width] = 0.0
        return out_image, out_mask

    out_image = torch.zeros(
        (batch, padded_height, padded_width, channels),
        dtype=image.dtype,
        device=image.device,
    )
    for batch_index in range(batch):
        if pad_mode == "edge":
            out_image[batch_index, :top, :, :] = image[batch_index, 0, :, :].mean(dim=0)
            out_image[batch_index, top + height:, :, :] = image[batch_index, height - 1, :, :].mean(dim=0)
            out_image[batch_index, :, :left, :] = image[batch_index, :, 0, :].mean(dim=0)
            out_image[batch_index, :, left + width:, :] = image[batch_index, :, width - 1, :].mean(dim=0)
        elif pad_mode == "edge_pixel":
            if top:
                out_image[batch_index, :top, left:left + width, :] = image[batch_index, 0:1, :, :]
            if bottom:
                out_image[batch_index, top + height:, left:left + width, :] = image[batch_index, -1:, :, :]
            if left:
                out_image[batch_index, top:top + height, :left, :] = image[batch_index, :, 0:1, :]
            if right:
                out_image[batch_index, top:top + height, left + width:, :] = image[batch_index, :, -1:, :]
            out_image[batch_index, :top, :left, :] = image[batch_index, 0, 0, :]
            out_image[batch_index, :top, left + width:, :] = image[batch_index, 0, -1, :]
            out_image[batch_index, top + height:, :left, :] = image[batch_index, -1, 0, :]
            out_image[batch_index, top + height:, left + width:, :] = image[batch_index, -1, -1, :]
        else:
            out_image[batch_index] = background

        out_image[batch_index, top:top + height, left:left + width, :] = image[batch_index]

    if mask is not None:
        out_mask = F.pad(mask, (left, right, top, bottom), mode="replicate")
    else:
        out_mask = torch.ones(
            (batch, padded_height, padded_width),
            dtype=image.dtype,
            device=image.device,
        )
        out_mask[:, top:top + height, left:left + width] = 0.0

    return out_image, out_mask


class CP_KJNodesImageResizeKJ:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": ("BOOLEAN", {"default": False}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "get_image_size": ("IMAGE",),
                "crop": (
                    ["disabled", "center", 0],
                    {
                        "tooltip": (
                            "0 will do the default center crop, this is a workaround for the "
                            "widget order changing with the new frontend, as in old workflows "
                            "the value of this widget becomes 0 automatically"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "resize"
    CATEGORY = categoryFolder
    DEPRECATED = True
    DESCRIPTION = """
DEPRECATED!

Due to ComfyUI frontend changes, this node should no longer be used, please check the
v2 of the node. This node is only kept to not completely break older workflows.
"""

    def resize(
        self,
        image,
        width,
        height,
        keep_proportion,
        upscale_method,
        divisible_by,
        width_input=None,
        height_input=None,
        get_image_size=None,
        crop="disabled",
    ):
        _, source_height, source_width, _ = image.shape

        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape

        if keep_proportion and get_image_size is None:
            if width == 0 and height != 0:
                ratio = height / source_height
                width = round(source_width * ratio)
            elif height == 0 and width != 0:
                ratio = width / source_width
                height = round(source_height * ratio)
            elif width != 0 and height != 0:
                ratio = min(width / source_width, height / source_height)
                width = round(source_width * ratio)
                height = round(source_height * ratio)
        else:
            if width == 0:
                width = source_width
            if height == 0:
                height = source_height

        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        if crop == 0:
            crop = "center"

        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)

        return (image, image.shape[2], image.shape[1])


class CP_KJNodesImageResizeKJv2:
    upscale_methods = [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
        "nvidia_rtx_vsr",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (
                    [
                        "stretch",
                        "resize",
                        "pad",
                        "pad_edge",
                        "pad_edge_pixel",
                        "crop",
                        "pillarbox_blur",
                        "total_pixels",
                    ],
                    {"default": False},
                ),
                "pad_color": (
                    "STRING",
                    {"default": "0, 0, 0", "tooltip": "Color to use for padding."},
                ),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (["cpu", "gpu"],),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = categoryFolder
    DESCRIPTION = """
Resizes the image to the specified width and height.

Supports stretching, proportional resizing, cropping, padding, blurred pillarbox
padding, total-pixel sizing, masks, CPU/GPU processing, and optional NVIDIA RTX VSR.
"""

    @staticmethod
    def _send_progress(message, unique_id):
        if unique_id and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(message, unique_id)
                return
            except Exception:
                pass
        logging.info("[ImageResizeKJv2] %s", message)

    def resize(
        self,
        image,
        width,
        height,
        keep_proportion,
        upscale_method,
        divisible_by,
        pad_color,
        crop_position,
        unique_id,
        device="cpu",
        mask=None,
        per_batch=64,
    ):
        batch, source_height, source_width, channels = image.shape

        if not isinstance(keep_proportion, str):
            keep_proportion = "stretch"

        if mask is not None and mask.shape[-2:] == (64, 64) and (source_height != 64 or source_width != 64):
            mask = None

        if mask is not None and mask.shape[-2:] != (source_height, source_width):
            mask = common_upscale(
                mask.unsqueeze(1),
                source_width,
                source_height,
                "bilinear",
                crop="disabled",
            ).squeeze(1)

        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on the GPU")
            work_device = model_management.get_torch_device()
        else:
            work_device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"
        pad_left = pad_right = pad_top = pad_bottom = 0

        if (
            keep_proportion in ("resize", "total_pixels")
            or keep_proportion.startswith("pad")
            or pillarbox_blur
        ):
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = source_width / source_height
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))
            elif width == 0 and height == 0:
                new_width = source_width
                new_height = source_height
            elif width == 0:
                ratio = height / source_height
                new_width = round(source_width * ratio)
                new_height = height
            elif height == 0:
                ratio = width / source_width
                new_width = width
                new_height = round(source_height * ratio)
            else:
                ratio = min(width / source_width, height / source_height)
                new_width = round(source_width * ratio)
                new_height = round(source_height * ratio)

            if keep_proportion.startswith("pad") or pillarbox_blur:
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                elif crop_position == "left":
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        else:
            if width == 0:
                width = source_width
            if height == 0:
                height = source_height

        if divisible_by > 1:
            width -= width % divisible_by
            height -= height % divisible_by

        if width < 1 or height < 1:
            raise ValueError(
                "The calculated resize dimensions must be at least 1x1. "
                "Increase width/height or reduce divisible_by."
            )

        if per_batch != 0 and batch > per_batch:
            try:
                output_bytes = batch * height * width * channels * image.element_size()
                self._send_progress(
                    f"Resize v2: estimated output ~{output_bytes / (1024 * 1024):.2f} MB; "
                    f"batching {per_batch}/{batch}",
                    unique_id,
                )
            except Exception:
                pass

        nvvfx_context = None
        nvvfx_super_resolution = None
        nvvfx_device = None
        nvvfx_output_width = None
        nvvfx_output_height = None
        if upscale_method == "nvidia_rtx_vsr":
            nvvfx_device = model_management.get_torch_device()
            if nvvfx_device.type != "cuda":
                raise RuntimeError("NVIDIA RTX VSR requires ComfyUI to use a CUDA device.")

            nvvfx_device_index = (
                nvvfx_device.index
                if nvvfx_device.index is not None
                else torch.cuda.current_device()
            )
            nvvfx_output_width = max(8, round(width / 8) * 8)
            nvvfx_output_height = max(8, round(height / 8) * 8)

            if not (64 <= nvvfx_output_width <= 8192 and 64 <= nvvfx_output_height <= 8192):
                raise ValueError(
                    "NVIDIA RTX VSR output dimensions must be between 64 and 8192 pixels; "
                    f"received {nvvfx_output_width}x{nvvfx_output_height}."
                )

            quality_levels = getattr(nvvfx, "QualityLevel", None)
            if quality_levels is None:
                quality_levels = nvvfx.effects.QualityLevel

            with torch.cuda.device(nvvfx_device):
                nvvfx_context = nvvfx.VideoSuperRes(
                    quality_levels.ULTRA,
                    device=nvvfx_device_index,
                )
                nvvfx_super_resolution = nvvfx_context.__enter__()
                nvvfx_super_resolution.output_width = nvvfx_output_width
                nvvfx_super_resolution.output_height = nvvfx_output_height
                nvvfx_super_resolution.load()

        def process_subbatch(in_image, in_mask, left, right, top, bottom):
            out_image = in_image if in_image.device == work_device else in_image.to(work_device)
            out_mask = None if in_mask is None else (
                in_mask if in_mask.device == work_device else in_mask.to(work_device)
            )

            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                if old_aspect > new_aspect:
                    crop_width = round(old_height * new_aspect)
                    crop_height = old_height
                else:
                    crop_width = old_width
                    crop_height = round(old_width / new_aspect)

                if crop_position == "center":
                    x = (old_width - crop_width) // 2
                    y = (old_height - crop_height) // 2
                elif crop_position == "top":
                    x = (old_width - crop_width) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_width) // 2
                    y = old_height - crop_height
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_height) // 2
                else:
                    x = old_width - crop_width
                    y = (old_height - crop_height) // 2

                out_image = out_image.narrow(-2, x, crop_width).narrow(-3, y, crop_height)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x, crop_width).narrow(-2, y, crop_height)

            if upscale_method == "nvidia_rtx_vsr":
                if out_image.shape[-1] < 3:
                    raise ValueError(
                        "NVIDIA RTX VSR requires an RGB image with at least three channels."
                    )

                input_height = out_image.shape[-3]
                input_width = out_image.shape[-2]
                if nvvfx_output_width <= input_width or nvvfx_output_height <= input_height:
                    raise ValueError(
                        "NVIDIA RTX VSR in ULTRA mode only supports upscaling. "
                        f"The requested size is {nvvfx_output_width}x{nvvfx_output_height}, "
                        f"but the input is {input_width}x{input_height}."
                    )
                if (
                    nvvfx_output_width > input_width * 4
                    or nvvfx_output_height > input_height * 4
                ):
                    raise ValueError(
                        "NVIDIA RTX VSR supports at most 4x enlargement per dimension. "
                        f"The requested resize is {input_width}x{input_height} to "
                        f"{nvvfx_output_width}x{nvvfx_output_height}."
                    )

                extra_channels = out_image[..., 3:] if out_image.shape[-1] > 3 else None
                frames_chw = (
                    out_image[..., :3]
                    .movedim(-1, 1)
                    .to(device=nvvfx_device, dtype=torch.float32)
                    .contiguous()
                )
                upscaled_frames = []
                with torch.cuda.device(nvvfx_device):
                    for frame in frames_chw:
                        try:
                            dlpack_output = nvvfx_super_resolution.run(frame).image
                        except nvvfx.NvVFXError as error:
                            raise RuntimeError(
                                "NVIDIA RTX VSR rejected an RGB float32 CUDA frame with "
                                f"shape 3x{input_height}x{input_width} and output size "
                                f"{nvvfx_output_width}x{nvvfx_output_height}: {error}"
                            ) from error
                        upscaled_frames.append(torch.from_dlpack(dlpack_output).clone())

                out_image = torch.stack(upscaled_frames, dim=0).movedim(1, -1).cpu()
                if extra_channels is not None:
                    extra_channels = common_upscale(
                        extra_channels.movedim(-1, 1),
                        out_image.shape[2],
                        out_image.shape[1],
                        "bilinear",
                        crop="disabled",
                    ).movedim(1, -1).cpu()
                    out_image = torch.cat((out_image, extra_channels), dim=-1)
                if out_mask is not None:
                    out_mask = common_upscale(
                        out_mask.unsqueeze(1),
                        out_image.shape[2],
                        out_image.shape[1],
                        "bilinear",
                        crop="disabled",
                    ).squeeze(1)
            else:
                out_image = common_upscale(
                    out_image.movedim(-1, 1),
                    width,
                    height,
                    upscale_method,
                    crop="disabled",
                ).movedim(1, -1)
                if out_mask is not None:
                    if upscale_method == "lanczos":
                        out_mask = common_upscale(
                            out_mask.unsqueeze(1).repeat(1, 3, 1, 1),
                            width,
                            height,
                            upscale_method,
                            crop="disabled",
                        ).movedim(1, -1)[:, :, :, 0]
                    else:
                        out_mask = common_upscale(
                            out_mask.unsqueeze(1),
                            width,
                            height,
                            upscale_method,
                            crop="disabled",
                        ).squeeze(1)

            if (
                (keep_proportion.startswith("pad") or pillarbox_blur)
                and (left > 0 or right > 0 or top > 0 or bottom > 0)
            ):
                padded_width = out_image.shape[2] + left + right
                padded_height = out_image.shape[1] + top + bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder:
                        right += divisible_by - width_remainder
                    if height_remainder:
                        bottom += divisible_by - height_remainder

                pad_mode = (
                    "pillarbox_blur"
                    if pillarbox_blur
                    else "edge"
                    if keep_proportion == "pad_edge"
                    else "edge_pixel"
                    if keep_proportion == "pad_edge_pixel"
                    else "color"
                )
                out_image, out_mask = _pad_image(
                    out_image,
                    left,
                    right,
                    top,
                    bottom,
                    pad_color,
                    pad_mode,
                    mask=out_mask,
                )

            return out_image, out_mask

        try:
            if per_batch == 0 or batch <= per_batch:
                out_image, out_mask = process_subbatch(
                    image,
                    mask,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                )
            else:
                image_chunks = []
                mask_chunks = [] if mask is not None else None
                total_batches = (batch + per_batch - 1) // per_batch
                for batch_index, start_index in enumerate(range(0, batch, per_batch), start=1):
                    end_index = min(start_index + per_batch, batch)
                    sub_image = image[start_index:end_index]
                    sub_mask = mask[start_index:end_index] if mask is not None else None
                    sub_out_image, sub_out_mask = process_subbatch(
                        sub_image,
                        sub_mask,
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                    )
                    image_chunks.append(sub_out_image.cpu())
                    if mask_chunks is not None:
                        mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                    self._send_progress(
                        f"Resize v2: batch {batch_index}/{total_batches}; images {end_index}/{batch}",
                        unique_id,
                    )

                out_image = torch.cat(image_chunks, dim=0)
                if mask_chunks is not None and any(item is not None for item in mask_chunks):
                    out_mask = torch.cat([item for item in mask_chunks if item is not None], dim=0)
                else:
                    out_mask = None
        finally:
            if nvvfx_context is not None:
                with torch.cuda.device(nvvfx_device):
                    nvvfx_context.__exit__(None, None, None)

        if unique_id and PromptServer is not None:
            try:
                memory_size_mb = out_image.numel() * out_image.element_size() / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"Output: {out_image.shape[0]} x {out_image.shape[2]} x "
                    f"{out_image.shape[1]} | {memory_size_mb:.2f}MB",
                    unique_id,
                )
            except Exception:
                pass

        if out_mask is None:
            out_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        return (
            out_image.cpu(),
            out_image.shape[2],
            out_image.shape[1],
            out_mask.cpu(),
        )


NODE_CLASS_MAPPINGS_IMPORTED_KJNODES_IMAGERESIZEKJ = {
    "ImageResizeKJ": CP_KJNodesImageResizeKJ,
    "ImageResizeKJv2": CP_KJNodesImageResizeKJv2,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_KJNODES_IMAGERESIZEKJ = {
    "ImageResizeKJ": "Resize Image (deprecated)",
    "ImageResizeKJv2": "Resize Image v2",
}
