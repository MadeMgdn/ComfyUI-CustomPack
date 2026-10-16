import ast
import logging
import math
import operator
import os
import struct
import time
from io import BytesIO
from threading import Thread

import folder_paths
import nvvfx
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor
from comfy import model_management
from comfy.patcher_extension import WrappersMP
from comfy.sd import VAE
from comfy.utils import (
    ProgressBar,
    common_upscale,
    load_torch_file,
    state_dict_prefix_replace,
)
from nodes import MAX_RESOLUTION
from server import BinaryEventTypes, PromptServer


categoryFolder = "Custom pack/Imported/KJNodes"


_KJ_PREVIEW_DEVICE = model_management.get_torch_device()


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


class CP_KJNodesVAELoaderKJ:
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @classmethod
    def vae_list(cls):
        vaes = list(folder_paths.get_filename_list("vae"))
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        approximate_pairs = {
            "taesd": {"encoder": False, "decoder": False},
            "taesdxl": {"encoder": False, "decoder": False},
            "taesd3": {"encoder": False, "decoder": False},
            "taef1": {"encoder": False, "decoder": False},
        }

        for filename in approx_vaes:
            matched_image_tae = False
            for tae_name, parts in approximate_pairs.items():
                if filename.startswith(f"{tae_name}_encoder."):
                    parts["encoder"] = True
                    matched_image_tae = True
                    break
                if filename.startswith(f"{tae_name}_decoder."):
                    parts["decoder"] = True
                    matched_image_tae = True
                    break

            if not matched_image_tae and any(
                filename.startswith(tae_name) for tae_name in cls.video_taes
            ):
                vaes.append(filename)

        for tae_name, parts in approximate_pairs.items():
            if parts["encoder"] and parts["decoder"]:
                vaes.append(tae_name)

        vaes.append("pixel_space")
        return list(dict.fromkeys(vaes))

    @staticmethod
    def load_taesd(name):
        state_dict = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        encoder = next(
            filename
            for filename in approx_vaes
            if filename.startswith(f"{name}_encoder.")
        )
        decoder = next(
            filename
            for filename in approx_vaes
            if filename.startswith(f"{name}_decoder.")
        )

        encoder_state_dict = load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", encoder)
        )
        for key, value in encoder_state_dict.items():
            state_dict[f"taesd_encoder.{key}"] = value

        decoder_state_dict = load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", decoder)
        )
        for key, value in decoder_state_dict.items():
            state_dict[f"taesd_decoder.{key}"] = value

        scale_and_shift = {
            "taesd": (0.18215, 0.0),
            "taesdxl": (0.13025, 0.0),
            "taesd3": (1.5305, 0.0609),
            "taef1": (0.3611, 0.1159),
        }
        if name in scale_and_shift:
            scale, shift = scale_and_shift[name]
            state_dict["vae_scale"] = torch.tensor(scale)
            state_dict["vae_shift"] = torch.tensor(shift)

        return state_dict

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (cls.vae_list(),),
                "device": (["main_device", "cpu"],),
                "weight_dtype": (["bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = categoryFolder

    def load_vae(self, vae_name, device, weight_dtype):
        metadata = None
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[weight_dtype]
        load_device = (
            model_management.get_torch_device()
            if device == "main_device"
            else torch.device("cpu")
        )

        if vae_name == "pixel_space":
            state_dict = {"pixel_space_vae": torch.tensor(1.0)}
        elif vae_name in self.image_taes:
            state_dict = self.load_taesd(vae_name)
        else:
            vae_basename = os.path.splitext(vae_name)[0]
            folder_name = "vae_approx" if vae_basename in self.video_taes else "vae"
            vae_path = folder_paths.get_full_path_or_raise(folder_name, vae_name)
            state_dict, metadata = load_torch_file(vae_path, return_metadata=True)

        is_audio_vae = any(
            key in state_dict
            for key in (
                "vocoder.conv_post.weight",
                "vocoder.vocoder.conv_post.weight",
                "vocoder.resblocks.0.convs1.0.weight",
                "vocoder.vocoder.resblocks.0.convs1.0.weight",
            )
        )
        if is_audio_vae:
            audio_state_dict = state_dict_prefix_replace(
                dict(state_dict),
                {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
                filter_keys=True,
            )
            vae = VAE(sd=audio_state_dict, metadata=metadata)
        else:
            vae = VAE(
                sd=state_dict,
                device=load_device,
                dtype=dtype,
                metadata=metadata,
            )

        vae.throw_exception_if_invalid()
        return (vae,)


class _KJWrappedPreviewer():
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias, rate=8, taeltx=None):
        self.first_preview = True
        self.taeltx = taeltx
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu") if latent_rgb_factors_bias is not None else None

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            PromptServer.instance.send_sync('VHS_latentpreview', {'length':num_images, 'rate': self.rate, 'id': PromptServer.instance.last_node_id})
            self.last_time = new_time + 1/self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]
        Thread(target=self.process_previews, args=(x0, self.c_index,
                                                   num_images)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None
    def process_previews(self, image_tensor, ind, leng):
        max_size = 512
        min_size = 256
        image_tensor = self.decode_latent_to_preview(image_tensor)

        if image_tensor.size(1) < min_size or image_tensor.size(2) < min_size:
            image_tensor = F.interpolate(image_tensor.movedim(-1,0), scale_factor=4, mode='nearest').movedim(0,-1)

        if image_tensor.size(1) > max_size or image_tensor.size(2) > max_size:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (max_size * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,max_size), mode='nearest')
            else:
                width = (max_size * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (max_size, width), mode='nearest')
            image_tensor = image_tensor.movedim(0,-1)

        previews_ubyte = (image_tensor.clamp(0, 1).mul(0xFF)).to(device="cpu", dtype=torch.uint8)

        # Send VHS preview
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big')*2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', (PromptServer.instance.last_node_id or "").encode('ascii')))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            #NOTE: send sync already uses call_soon_threadsafe
            PromptServer.instance.send_sync(BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), PromptServer.instance.client_id)
            if self.taeltx is not None:
                ind = (ind + 1) % ((leng-1) * 8 + 1)
            else:
                ind = (ind + 1) % leng

    def decode_latent_to_preview(self, x0):
        if self.taeltx is not None:
            x0 = x0.unsqueeze(0).to(dtype=self.taeltx.first_stage_model.decoder[1].weight.dtype, device=_KJ_PREVIEW_DEVICE)
            x_sample = self.taeltx.first_stage_model.decode(x0)[0].permute(1, 2, 3, 0)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)

            #low = latent_image.quantile(0.01)
            #high = latent_image.quantile(0.99)
            #latent_image = ((latent_image - low) / (high - low)).clamp(0, 1)
            latent_image = torch.sigmoid(latent_image)
            return latent_image


def _kj_get_ltx_rgb_factors(is_23):
    """Return (latent_rgb_factors, latent_rgb_factors_bias) for LTX (is_23=False) or
    LTX2/LTXAV (is_23=True). Exposed so other nodes can build a _KJWrappedPreviewer with
    the right LTX factors without duplicating the 128-row tables."""
    return _kj__kj_get_ltx_rgb_factors_impl(is_23)


def _kj__kj_get_ltx_rgb_factors_impl(is_23):
    if not is_23:
        latent_rgb_factors = [
                [ 0.0350,  0.0159,  0.0132],
                [ 0.0025, -0.0021, -0.0003],
                [ 0.0286,  0.0028,  0.0020],
                [ 0.0280, -0.0114, -0.0202],
                [-0.0186,  0.0073,  0.0092],
                [ 0.0027,  0.0097, -0.0113],
                [-0.0069, -0.0032, -0.0024],
                [-0.0323, -0.0370, -0.0457],
                [ 0.0174,  0.0164,  0.0106],
                [-0.0097,  0.0061,  0.0035],
                [-0.0130, -0.0042, -0.0012],
                [-0.0102, -0.0002, -0.0091],
                [-0.0025,  0.0063,  0.0161],
                [ 0.0003,  0.0037,  0.0108],
                [ 0.0152,  0.0082,  0.0143],
                [ 0.0317,  0.0203,  0.0312],
                [-0.0092, -0.0233, -0.0119],
                [-0.0405, -0.0226, -0.0023],
                [ 0.0376,  0.0397,  0.0352],
                [ 0.0171, -0.0043, -0.0095],
                [ 0.0482,  0.0341,  0.0213],
                [ 0.0031, -0.0046, -0.0018],
                [-0.0486, -0.0383, -0.0294],
                [-0.0071, -0.0272, -0.0123],
                [ 0.0320,  0.0218,  0.0289],
                [ 0.0327,  0.0088, -0.0116],
                [-0.0098, -0.0240, -0.0111],
                [ 0.0094, -0.0116,  0.0021],
                [ 0.0309,  0.0092,  0.0165],
                [-0.0065, -0.0077, -0.0107],
                [ 0.0179,  0.0114,  0.0038],
                [-0.0018, -0.0030, -0.0026],
                [-0.0002,  0.0076, -0.0029],
                [-0.0131, -0.0059, -0.0170],
                [ 0.0055,  0.0066, -0.0038],
                [ 0.0154,  0.0063,  0.0090],
                [ 0.0186,  0.0175,  0.0188],
                [-0.0166, -0.0381, -0.0428],
                [ 0.0121,  0.0015, -0.0153],
                [ 0.0118,  0.0050,  0.0019],
                [ 0.0125,  0.0259,  0.0231],
                [ 0.0046,  0.0130,  0.0081],
                [ 0.0271,  0.0250,  0.0250],
                [-0.0054, -0.0347, -0.0326],
                [-0.0438, -0.0262, -0.0228],
                [-0.0191, -0.0256, -0.0173],
                [-0.0205, -0.0058,  0.0042],
                [ 0.0404,  0.0434,  0.0346],
                [-0.0242, -0.0177, -0.0146],
                [ 0.0161,  0.0223,  0.0168],
                [-0.0240, -0.0320, -0.0299],
                [-0.0019,  0.0043,  0.0008],
                [-0.0060, -0.0133, -0.0244],
                [-0.0048, -0.0225, -0.0167],
                [ 0.0267,  0.0133,  0.0152],
                [ 0.0222,  0.0167,  0.0028],
                [ 0.0015, -0.0062,  0.0013],
                [-0.0241, -0.0178, -0.0079],
                [ 0.0040, -0.0081, -0.0097],
                [-0.0064,  0.0133, -0.0011],
                [-0.0204, -0.0231, -0.0304],
                [ 0.0011, -0.0011,  0.0145],
                [-0.0283, -0.0259, -0.0260],
                [ 0.0038,  0.0171, -0.0029],
                [ 0.0637,  0.0424,  0.0409],
                [ 0.0092,  0.0163,  0.0188],
                [ 0.0082,  0.0055, -0.0179],
                [-0.0177, -0.0286, -0.0147],
                [ 0.0171,  0.0242,  0.0398],
                [-0.0129,  0.0095, -0.0071],
                [-0.0154,  0.0036,  0.0128],
                [-0.0081, -0.0009,  0.0118],
                [-0.0067, -0.0178, -0.0230],
                [-0.0022, -0.0125, -0.0003],
                [-0.0032, -0.0039, -0.0022],
                [-0.0005, -0.0127, -0.0131],
                [-0.0143, -0.0157, -0.0165],
                [-0.0262, -0.0263, -0.0270],
                [ 0.0063,  0.0127,  0.0178],
                [ 0.0092,  0.0133,  0.0150],
                [-0.0106, -0.0068,  0.0032],
                [-0.0214, -0.0022,  0.0171],
                [-0.0104, -0.0266, -0.0362],
                [ 0.0021,  0.0048, -0.0005],
                [ 0.0345,  0.0431,  0.0402],
                [-0.0275, -0.0110, -0.0195],
                [ 0.0203,  0.0251,  0.0224],
                [ 0.0016, -0.0037, -0.0094],
                [ 0.0241,  0.0198,  0.0114],
                [-0.0003,  0.0027,  0.0141],
                [ 0.0012, -0.0052, -0.0084],
                [ 0.0057, -0.0028, -0.0163],
                [-0.0488, -0.0545, -0.0509],
                [-0.0076, -0.0025, -0.0014],
                [-0.0249, -0.0142, -0.0367],
                [ 0.0136,  0.0041,  0.0135],
                [ 0.0007,  0.0034, -0.0053],
                [-0.0068, -0.0109,  0.0029],
                [ 0.0006, -0.0237, -0.0094],
                [-0.0149, -0.0177, -0.0131],
                [-0.0105,  0.0039,  0.0216],
                [ 0.0242,  0.0200,  0.0180],
                [-0.0339, -0.0153, -0.0195],
                [ 0.0104,  0.0151,  0.0120],
                [-0.0043,  0.0089,  0.0047],
                [ 0.0157, -0.0030,  0.0008],
                [ 0.0126,  0.0102, -0.0040],
                [ 0.0040,  0.0114,  0.0137],
                [ 0.0423,  0.0473,  0.0436],
                [-0.0128, -0.0066, -0.0152],
                [-0.0337, -0.0087, -0.0026],
                [-0.0052,  0.0235,  0.0291],
                [ 0.0079,  0.0154,  0.0260],
                [-0.0539, -0.0377, -0.0358],
                [-0.0188,  0.0062, -0.0035],
                [-0.0186,  0.0041, -0.0083],
                [ 0.0045, -0.0049,  0.0053],
                [ 0.0172,  0.0071,  0.0042],
                [-0.0003, -0.0078, -0.0096],
                [-0.0209, -0.0132, -0.0135],
                [-0.0074,  0.0017,  0.0099],
                [-0.0038,  0.0070,  0.0014],
                [-0.0013, -0.0017,  0.0073],
                [ 0.0030,  0.0105,  0.0105],
                [ 0.0154, -0.0168, -0.0235],
                [-0.0108, -0.0038,  0.0047],
                [-0.0298, -0.0347, -0.0436],
                [-0.0206, -0.0189, -0.0139]
            ]
        latent_rgb_factors_bias = [0.2796, 0.1101, -0.0047]
    else:
        latent_rgb_factors = [[0.002269406570121646, -0.02110900916159153, -0.009850316680967808], [-0.016038373112678528, -0.012462412007153034, -0.01112896017730236], [0.025274179875850677, 0.011209743097424507, 0.025426799431443214], [0.04690725728869438, 0.041542328894138336, 0.03568895906209946], [-0.02388044260442257, -0.0018645941745489836, 0.01858334057033062], [0.03720448538661003, 0.0220357533544302, 0.027937663719058037], [-0.07273884862661362, -0.09326262027025223, -0.11579664051532745], [-0.063837431371212, 0.00026216846890747547, 0.03042735904455185], [0.02903873845934868, 0.042082373052835464, 0.030649805441498756], [0.03777873516082764, 0.0322984978556633, -0.005671461578458548], [-0.0075670829974114895, -0.012113905511796474, -0.01638956367969513], [0.026524530723690987, 0.060518112033605576, 0.059549521654844284], [0.10093028098344803, 0.10073262453079224, 0.0505094900727272], [0.03725508227944374, 0.015382086858153343, 0.005786076188087463], [-0.03139607608318329, -0.01690264232456684, -0.0013519978383556008], [-0.027200624346733093, -0.02517341822385788, -0.008874989114701748], [0.024963486939668655, 0.04293748363852501, 0.05582639202475548], [-0.0364827960729599, -0.026975594460964203, -0.021950015798211098], [0.027655167505145073, 0.025136707350611687, 0.043967027217149734], [0.035822272300720215, 0.013104500249028206, 0.01113432738929987], [0.05353763327002525, 0.013606574386358261, -0.018720127642154694], [-0.013587888330221176, -0.01689346879720688, -0.027842802926898003], [0.059415675699710846, 0.03734271228313446, 0.04562298208475113], [-0.02946414425969124, -0.038338612765073776, 0.001805233070626855], [0.03921474143862724, 0.0651894062757492, 0.10681862384080887], [-0.00744189927354455, 0.007951526902616024, 0.020728807896375656], [-0.04038553684949875, -0.05215264856815338, -0.07213657349348068], [-0.004655141849070787, 0.01305423304438591, 0.026104029268026352], [0.03434251993894577, 0.018448110669851303, 0.013096392154693604], [0.0022075253073126078, -0.0011812079465016723, 0.0002940484555438161], [-0.00043441299931146204, 0.02366728149354458, 0.035889431834220886], [-0.030657343566417694, -0.024926183745265007, -0.012355240061879158], [-0.018955843523144722, -0.017360301688313484, -0.008214764297008514], [-0.01113052573055029, -0.01201171800494194, -0.002986249281093478], [0.018902746960520744, 0.01758778840303421, 0.026414571329951286], [-0.019977254793047905, -0.01605399139225483, -0.019136475399136543], [-0.00300968368537724, -0.017609693109989166, -0.013655650429427624], [0.0022096361499279737, 0.017998533323407173, 0.01815750263631344], [0.05186990648508072, 0.03285299986600876, 0.016072165220975876], [0.012626334093511105, 0.0013884707586839795, -0.012077193707227707], [-0.0037861645687371492, -0.013902144506573677, -0.01911942847073078], [-0.014163163490593433, -0.00513274222612381, -0.014303527772426605], [-0.010461323894560337, 0.009658926166594028, 0.01644069515168667], [-0.008665377274155617, 0.002501955023035407, -0.009703717194497585], [-0.03404829278588295, -0.02546044997870922, -0.014914450235664845], [0.04997691139578819, 0.06592527031898499, 0.073111392557621], [0.027394814416766167, 0.024555068463087082, 0.019957970827817917], [-0.027501430362462997, -0.01673700101673603, -0.03089248389005661], [-0.018696032464504242, -0.0020940247923135757, 0.015244065783917904], [-0.0062704551964998245, -0.0067006442695856094, -0.007532030809670687], [0.014871004968881607, 0.009914354421198368, 0.020960720255970955], [0.03662937879562378, 0.04413224756717682, 0.04220828413963318], [-0.011242181062698364, -0.013539309613406658, -0.016438307240605354], [-0.014854325912892818, 0.0038217694964259863, -0.002461288822814822], [-0.014826249331235886, 0.0009719038498587906, -0.012078499421477318], [-0.029396841302514076, -0.01432017982006073, 0.013018904253840446], [0.02755064144730568, 0.028369395062327385, 0.01640605367720127], [0.12049165368080139, 0.1395745575428009, 0.14566579461097717], [0.019721267744898796, 0.009739740751683712, 0.0023876908235251904], [-0.007320966571569443, 0.0065013207495212555, 0.01603059470653534], [0.007391378283500671, -0.0073603675700724125, -0.01770283281803131], [0.02984853833913803, 0.012391146272420883, 0.010563627816736698], [-0.013479884713888168, -0.008637298829853535, -0.013457189314067364], [0.04127075523138046, 0.03032625839114189, 0.024770958349108696], [-0.06524652987718582, -0.012209279462695122, 0.02087211236357689], [-0.1179763451218605, -0.060323599725961685, -0.07592175155878067], [-0.07122819870710373, -0.04385707899928093, -0.022124603390693665], [-0.04682473465800285, -0.022610662505030632, -0.010107148438692093], [-0.0054328180849552155, -0.010368981398642063, -0.008167334832251072], [0.029181398451328278, 0.030588403344154358, 0.028090540319681168], [0.016619984060525894, 0.004931286443024874, -0.006450849585235119], [0.01035264041274786, 0.002237115055322647, 0.0013903985964134336], [-0.04313831403851509, -0.061772625893354416, -0.08946335315704346], [0.0150345079600811, 0.007781678810715675, 0.0011013159528374672], [-0.013585779815912247, 0.008117705583572388, 0.020367907360196114], [-0.172962948679924, -0.16406646370887756, -0.1668281853199005], [0.0083833709359169, 0.0015236001927405596, -0.01731627807021141], [0.021939430385828018, 0.018004458397626877, 0.014768349006772041], [0.008083095774054527, -0.013463049195706844, -0.022061636671423912], [0.024328550323843956, 0.0128010343760252, 0.014966367743909359], [0.05850301682949066, 0.027980001643300056, 0.02225641906261444], [0.09690416604280472, 0.06929530203342438, 0.03253814950585365], [0.048208240419626236, 0.025294817984104156, 0.023508133366703987], [-0.026432134211063385, -0.040383171290159225, -0.03950457274913788], [-0.021598653867840767, -0.017070941627025604, -0.010933087207376957], [0.011645167134702206, 0.002806191798299551, 0.003779367310926318], [0.10478592664003372, 0.08954174816608429, 0.06555330753326416], [0.015151776373386383, -0.016160616651177406, -0.024905217811465263], [0.019659176468849182, 0.008487952873110771, 0.002426224760711193], [-0.05173315480351448, -0.026337839663028717, -0.02127116546034813], [0.016987523064017296, 0.006270893849432468, 0.0015798212261870503], [0.007938026450574398, -0.005250005517154932, -0.020408453419804573], [0.013017759658396244, 0.01654384844005108, 0.04163840040564537], [-0.009886542335152626, -0.026848411187529564, -0.03070281818509102], [0.01108171883970499, 0.01827266439795494, -0.007332107983529568], [-0.0285995751619339, -0.031727731227874756, -0.03370537981390953], [0.005299570970237255, 0.05678633600473404, 0.02825017087161541], [-0.055322226136922836, -0.09084303677082062, -0.12999044358730316], [0.01844066195189953, 0.031044499948620796, 0.021148500964045525], [-0.004471115302294493, 0.005830412730574608, 0.00911418441683054], [-0.04053843766450882, -0.016424428671598434, -0.0010634599020704627], [0.03858831524848938, 0.007309338077902794, -0.005618985276669264], [0.01423253770917654, -0.0055681923404335976, 3.394074519746937e-05], [0.11455483734607697, 0.14653916656970978, 0.1488018035888672], [-0.005231931805610657, -0.0033921014983206987, -0.000995257287286222], [0.01449565589427948, 0.019586293026804924, 0.04565812274813652], [-0.005179048050194979, -0.011201606132090092, -0.0008710073889233172], [-0.015361929312348366, 0.00778581015765667, -0.008238887414336205], [-0.1147838830947876, -0.09109023958444595, -0.050579313188791275], [0.09037500619888306, 0.09597006440162659, 0.10811734944581985], [0.001873677596449852, -0.01772197335958481, -0.07681205868721008], [-0.020383257418870926, -0.016072455793619156, -0.01077069528400898], [-0.060444317758083344, -0.05499502643942833, -0.06153025105595589], [-0.016717270016670227, 0.026493264362215996, 0.021835654973983765], [0.008203534409403801, 0.00418612826615572, 0.013867748901247978], [0.0789225772023201, 0.05467747151851654, 0.016568133607506752], [-0.15149451792240143, -0.1526806503534317, -0.14325062930583954], [0.00538366474211216, 0.010192245244979858, -0.00449327751994133], [-0.004906965419650078, -0.005569908302277327, -0.02096559666097164], [0.024530155584216118, 0.010962833650410175, 0.0034586559049785137], [0.03551010414958, 0.017310436815023422, 0.007064413744956255], [0.11111932247877121, 0.09825586527585983, 0.08827318251132965], [-0.051722846925258636, -0.047595202922821045, -0.03763044252991676], [-0.02975175902247429, -0.02153967320919037, -0.021425534039735794], [-0.03462936729192734, -0.025198571383953094, -0.017322326079010963], [-0.016921017318964005, -0.012419789098203182, -0.0154880927875638], [-0.08035065978765488, -0.08451078832149506, -0.09623870998620987], [-0.03870908170938492, -0.04211008921265602, -0.04383759945631027]]
        latent_rgb_factors_bias = [-0.6957847476005554, -0.7276281118392944, -0.7405748963356018]
    return latent_rgb_factors, latent_rgb_factors_bias


def _kj_unwrap_upscale_model(latent_upscale_model):
    # newer comfy wraps the upsampler in a ModelPatcher
    return getattr(latent_upscale_model, "model", latent_upscale_model)

def _kj_prepare_ltx_preview_callback(model, steps, x0_output_dict=None, shape=None, latent_upscale_model=None, vae=None, rate=8, taeltx=False, num_keyframes=0, is_23=False):
    latent_rgb_factors, latent_rgb_factors_bias = _kj_get_ltx_rgb_factors(is_23)
    upscaler = _kj_unwrap_upscale_model(latent_upscale_model) if latent_upscale_model is not None else None
    upscaler_dtype = next(upscaler.parameters()).dtype if upscaler is not None else None
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = _KJWrappedPreviewer(latent_rgb_factors, latent_rgb_factors_bias, rate=rate, taeltx=vae if taeltx else None)

    pbar = ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0 is not None and shape is not None:
            cut = math.prod(shape[1:])
            x0 = x0[:, :, :cut].reshape([x0.shape[0]] + list(shape)[1:])

        if num_keyframes > 0:
            x0 = x0[:, :, :-num_keyframes]

        if upscaler is not None:
            x0 = vae.first_stage_model.per_channel_statistics.un_normalize(x0)
            x0 = upscaler(x0.to(upscaler_dtype))
            x0 = vae.first_stage_model.per_channel_statistics.normalize(x0)

        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

class _KJOuterSampleCallbackWrapper:
    def __init__(self, latent_upscale_model=None, vae=None, preview_rate=8, taeltx=False):
        self.latent_upscale_model = latent_upscale_model
        self.vae = vae
        self.preview_rate = preview_rate
        self.taeltx = taeltx
        self.x0_output = {}

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        guider = executor.class_obj
        diffusion_model = guider.model_patcher.model.diffusion_model
        is_23 = not diffusion_model.caption_projection_first_linear

        original_callback = callback
        if self.latent_upscale_model is not None:
            _kj_unwrap_upscale_model(self.latent_upscale_model).to(_KJ_PREVIEW_DEVICE)
        if self.vae is not None and self.taeltx:
            self.vae.first_stage_model.to(_KJ_PREVIEW_DEVICE)

        num_keyframes = 0
        if 'positive' in guider.conds and len(guider.conds['positive']) > 0:
            keyframe_idxs = guider.conds['positive'][0].get('keyframe_idxs')
            if keyframe_idxs is not None:
                num_keyframes = len(torch.unique(keyframe_idxs[0, 0, :, 0]))

        new_callback = _kj_prepare_ltx_preview_callback(guider.model_patcher, len(sigmas) -1, shape=latent_shapes[0] if len(latent_shapes) > 1 else None,
                                        x0_output_dict=self.x0_output, latent_upscale_model=self.latent_upscale_model, vae=self.vae, rate=self.preview_rate, taeltx=self.taeltx, num_keyframes=num_keyframes, is_23=is_23)
        # Wrapper that calls both callbacks
        def combined_callback(step, x0, x, total_steps):
            new_callback(step, x0, x, total_steps)
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)
        out = executor(noise, latent_image, sampler, sigmas, denoise_mask, combined_callback, disable_pbar, seed, latent_shapes=latent_shapes)
        if self.latent_upscale_model is not None:
            _kj_unwrap_upscale_model(self.latent_upscale_model).to(model_management.unet_offload_device())
        return out


class CP_KJNodesLTX2SamplingPreviewOverride:
    DESCRIPTION = (
        "Overrides LTX2 sampling previews until equivalent preview support is "
        "available in ComfyUI core."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preview_rate": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "latent_upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = categoryFolder
    EXPERIMENTAL = True

    def execute(self, model, preview_rate, latent_upscale_model=None, vae=None):
        if latent_upscale_model is not None and vae is None:
            raise ValueError(
                "A VAE is required when latent_upscale_model is connected."
            )

        patched_model = model.clone()
        taeltx = False
        if vae is not None and vae.first_stage_model.__class__.__name__ == "TAEHV":
            taeltx = True
            latent_upscale_model = None

        patched_model.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            "sampling_preview",
            _KJOuterSampleCallbackWrapper(
                latent_upscale_model,
                vae,
                preview_rate,
                taeltx,
            ),
        )
        return (patched_model,)


class _KJAnyType(str):
    def __ne__(self, _value):
        return False


_KJ_ANY_TYPE = _KJAnyType("*")
_KJ_CALCULATOR_INPUT_NAMES = tuple("abcdefghijk")


class _KJCalculatorOptionalInputs(dict):
    def __init__(self):
        input_spec = (_KJ_ANY_TYPE, {"forceInput": True})
        super().__init__({"a": input_spec, "b": input_spec})
        self.input_spec = input_spec

    def __getitem__(self, key):
        if key not in _KJ_CALCULATOR_INPUT_NAMES:
            raise KeyError(key)
        return self.input_spec

    def __contains__(self, key):
        return key in _KJ_CALCULATOR_INPUT_NAMES

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


_KJ_CALCULATOR_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_KJ_CALCULATOR_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


class CP_KJNodesSimpleCalculatorKJ:
    DESCRIPTION = (
        "Evaluates arithmetic, comparisons and boolean expressions using "
        "the connected variables a through k."
    )
    SEARCH_ALIASES = ["math", "arithmetic", "expression", "logic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": (
                    "STRING",
                    {
                        "default": "a + b",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": _KJCalculatorOptionalInputs(),
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    FUNCTION = "calculate"
    CATEGORY = categoryFolder

    def calculate(self, expression, **variables):
        try:
            allowed_names = {
                "pi": math.pi,
                "euler": math.e,
                "True": True,
                "False": False,
            }
            for name, value in variables.items():
                if name not in _KJ_CALCULATOR_INPUT_NAMES:
                    continue
                if not isinstance(value, (int, float, bool)):
                    raise TypeError(
                        f"Variable '{name}' must be an INT, FLOAT or BOOLEAN"
                    )
                allowed_names[name] = value
        except Exception as error:
            logging.error("SimpleCalculatorKJ error: %s", error)
            return (0.0, 0, False)

        def evaluate(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, bool)):
                    return node.value
                raise ValueError(f"Constant {node.value!r} is not allowed")

            if isinstance(node, ast.Name):
                if node.id in allowed_names:
                    return allowed_names[node.id]
                raise ValueError(f"Name '{node.id}' is not allowed")

            if isinstance(node, ast.BinOp):
                operation = _KJ_CALCULATOR_OPERATORS.get(type(node.op))
                if operation is None:
                    raise ValueError(
                        f"Operator {type(node.op).__name__} is not allowed"
                    )
                return operation(evaluate(node.left), evaluate(node.right))

            if isinstance(node, ast.UnaryOp):
                operation = _KJ_CALCULATOR_OPERATORS.get(type(node.op))
                if operation is None:
                    raise ValueError(
                        f"Operator {type(node.op).__name__} is not allowed"
                    )
                return operation(evaluate(node.operand))

            if isinstance(node, ast.Compare):
                left = evaluate(node.left)
                for operation_node, comparator_node in zip(
                    node.ops,
                    node.comparators,
                ):
                    operation = _KJ_CALCULATOR_OPERATORS.get(type(operation_node))
                    if operation is None:
                        raise ValueError(
                            f"Operator {type(operation_node).__name__} is not allowed"
                        )
                    right = evaluate(comparator_node)
                    if not operation(left, right):
                        return False
                    left = right
                return True

            if isinstance(node, ast.BoolOp):
                values = [evaluate(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    return all(values)
                if isinstance(node.op, ast.Or):
                    return any(values)
                raise ValueError(
                    f"Operator {type(node.op).__name__} is not allowed"
                )

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls are allowed")
                function = _KJ_CALCULATOR_FUNCTIONS.get(node.func.id)
                if function is None:
                    raise ValueError(
                        f"Function '{node.func.id}' is not allowed"
                    )
                if node.keywords:
                    raise ValueError("Keyword arguments are not allowed")
                return function(*(evaluate(argument) for argument in node.args))

            raise ValueError(f"Node type {type(node).__name__} is not allowed")

        try:
            result = evaluate(ast.parse(expression, mode="eval").body)
            return (float(result), int(result), bool(result))
        except Exception as error:
            logging.error("SimpleCalculatorKJ error: %s", error)
            return (0.0, 0, False)


class CP_KJNodesLazySwitchKJ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN",),
                "on_false": (_KJ_ANY_TYPE, {"lazy": True}),
                "on_true": (_KJ_ANY_TYPE, {"lazy": True}),
            },
        }

    RETURN_TYPES = (_KJ_ANY_TYPE,)
    FUNCTION = "switch"
    CATEGORY = categoryFolder
    DESCRIPTION = "Controls flow of execution based on a boolean switch."

    def check_lazy_status(self, switch, on_false=None, on_true=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]
        return None

    def switch(self, switch, on_false=None, on_true=None):
        return (on_true if switch else on_false,)


NODE_CLASS_MAPPINGS_IMPORTED_KJNODES_KJNODES = {
    "ImageResizeKJ": CP_KJNodesImageResizeKJ,
    "ImageResizeKJv2": CP_KJNodesImageResizeKJv2,
    "VAELoaderKJ": CP_KJNodesVAELoaderKJ,
    "LTX2SamplingPreviewOverride": CP_KJNodesLTX2SamplingPreviewOverride,
    "SimpleCalculatorKJ": CP_KJNodesSimpleCalculatorKJ,
    "LazySwitchKJ": CP_KJNodesLazySwitchKJ,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_KJNODES_KJNODES = {
    "ImageResizeKJ": "Resize Image (deprecated)",
    "ImageResizeKJv2": "Resize Image v2",
    "VAELoaderKJ": "VAELoader KJ",
    "LTX2SamplingPreviewOverride": "LTX2 Sampling Preview Override",
    "SimpleCalculatorKJ": "Simple Calculator KJ",
    "LazySwitchKJ": "Lazy Switch KJ",
}
