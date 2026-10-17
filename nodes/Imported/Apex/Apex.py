import json
import math
from datetime import datetime

import torch
import torch.nn.functional as F


categoryFolder = "Custom pack/Imported/Apex"


_APEX_RESOLUTION_SETS = {
    "Standard": [
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
        (832, 1280),
        (1280, 832),
        (704, 1504),
        (1504, 704),
        (896, 1344),
        (1344, 896),
        (960, 1280),
        (1280, 960),
        (512, 512),
        (768, 768),
        (640, 640),
    ],
    "Extended": [
        (1024, 1024),
        (1152, 896),
        (896, 1152),
        (1216, 832),
        (832, 1216),
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
        (1728, 576),
        (576, 1728),
        (1920, 512),
        (512, 1920),
        (2048, 512),
        (512, 2048),
        (832, 1280),
        (1280, 832),
        (704, 1504),
        (1504, 704),
        (960, 1536),
        (1536, 960),
        (1088, 1472),
        (1472, 1088),
    ],
    "Flux": [
        (1024, 1024),
        (768, 1344),
        (832, 1216),
        (896, 1152),
        (1152, 896),
        (1216, 832),
        (1344, 768),
        (512, 512),
        (640, 1536),
        (1536, 640),
        (704, 1504),
        (1504, 704),
        (832, 1280),
        (1280, 832),
    ],
    "Portrait": [
        (832, 1216),
        (768, 1344),
        (640, 1536),
        (896, 1152),
        (832, 1280),
        (704, 1504),
        (512, 768),
        (576, 1024),
        (640, 960),
        (720, 1280),
        (768, 1024),
        (896, 1344),
    ],
    "Landscape": [
        (1216, 832),
        (1344, 768),
        (1536, 640),
        (1152, 896),
        (1280, 832),
        (1504, 704),
        (768, 512),
        (1024, 576),
        (960, 640),
        (1280, 720),
        (1024, 768),
        (1344, 896),
    ],
    "Square": [
        (512, 512),
        (640, 640),
        (768, 768),
        (832, 832),
        (896, 896),
        (1024, 1024),
        (1152, 1152),
        (1216, 1216),
        (1280, 1280),
        (1344, 1344),
    ],
}


class CP_ApexSmartResize:
    resolution_sets = _APEX_RESOLUTION_SETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": (
                    ["Standard", "Extended", "Flux", "Portrait", "Landscape", "Square"],
                    {"default": "Standard"},
                ),
                "snap_method": (
                    [
                        "keep_proportion",
                        "closest_area",
                        "closest_ratio",
                        "prefer_larger",
                        "prefer_smaller",
                    ],
                    {"default": "keep_proportion"},
                ),
                "resize_mode": (
                    [
                        "crop_center",
                        "stretch",
                        "fit_pad_black",
                        "fit_pad_white",
                        "fit_pad_edge",
                    ],
                    {"default": "crop_center"},
                ),
                "interpolation": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {"default": "lanczos"},
                ),
                "show_candidates": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Print resolution candidates to the console.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "image",
        "width",
        "height",
        "scale_factor",
        "resolution_info",
        "console_log",
    )
    FUNCTION = "smart_resize"
    CATEGORY = categoryFolder

    def smart_resize(
        self,
        image,
        resolution_set,
        snap_method,
        resize_mode,
        interpolation,
        show_candidates,
    ):
        start_time = datetime.now()
        working_image = image
        orig_w = 0
        orig_h = 0

        try:
            if not isinstance(image, torch.Tensor):
                raise TypeError("image must be a torch.Tensor")
            if image.ndim == 3:
                working_image = image.unsqueeze(0)
            elif image.ndim != 4:
                raise ValueError("image must use HWC or BHWC layout")

            _, orig_h, orig_w, _ = working_image.shape
            if orig_w <= 0 or orig_h <= 0:
                raise ValueError("image dimensions must be greater than zero")

            target_w, target_h, info, candidates_info = self._find_best_resolution(
                orig_w,
                orig_h,
                resolution_set,
                snap_method,
            )
            scale_factor = math.sqrt((target_w * target_h) / (orig_w * orig_h))
            resized_image = self._apply_resize(
                working_image,
                target_w,
                target_h,
                resize_mode,
                interpolation,
            )

            console_data = self._create_console_data(
                orig_w,
                orig_h,
                target_w,
                target_h,
                scale_factor,
                resize_mode,
                resolution_set,
                snap_method,
                candidates_info,
                start_time,
            )
            console_data["processing_time_seconds"] = round(
                (datetime.now() - start_time).total_seconds(),
                3,
            )
            console_output = json.dumps(console_data, indent=2)

            if show_candidates:
                print(f"[ApexSmartResize]\n{console_output}")

            return (
                resized_image,
                target_w,
                target_h,
                scale_factor,
                info,
                console_output,
            )
        except Exception as error:
            error_console = json.dumps(
                {
                    "status": "error",
                    "message": str(error),
                    "original_size": f"{orig_w}x{orig_h}",
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
            return (
                working_image,
                orig_w,
                orig_h,
                1.0,
                f"Error: {error}",
                error_console,
            )

    @staticmethod
    def _create_console_data(
        orig_w,
        orig_h,
        target_w,
        target_h,
        scale_factor,
        resize_mode,
        resolution_set,
        snap_method,
        candidates_info,
        start_time,
    ):
        orig_area = orig_w * orig_h
        target_area = target_w * target_h
        memory_change_mb = ((target_area - orig_area) * 4 * 3) / (1024 * 1024)

        return {
            "action": "Smart Resize Complete",
            "status": "success",
            "timestamp": start_time.isoformat(),
            "input": {
                "size": f"{orig_w}x{orig_h}",
                "aspect_ratio": round(orig_w / orig_h, 3),
                "total_pixels": f"{orig_area:,}",
                "estimated_memory_mb": round((orig_area * 4 * 3) / (1024 * 1024), 1),
            },
            "output": {
                "size": f"{target_w}x{target_h}",
                "aspect_ratio": round(target_w / target_h, 3),
                "total_pixels": f"{target_area:,}",
                "estimated_memory_mb": round((target_area * 4 * 3) / (1024 * 1024), 1),
            },
            "processing": {
                "resolution_set": resolution_set,
                "snap_method": snap_method,
                "resize_mode": resize_mode,
                "scale_factor": round(scale_factor, 3),
                "size_change_percent": round((scale_factor * scale_factor - 1) * 100, 1),
                "memory_change_mb": round(memory_change_mb, 1),
            },
            "candidates": candidates_info,
        }

    def _find_best_resolution(self, orig_w, orig_h, resolution_set, snap_method):
        resolutions = self.resolution_sets[resolution_set]
        orig_area = orig_w * orig_h
        orig_aspect = orig_w / orig_h

        if snap_method == "keep_proportion":
            return self._keep_proportion_snap(orig_w, orig_h, resolutions)

        candidates = []
        for width, height in resolutions:
            area = width * height
            aspect = width / height
            candidates.append(
                {
                    "resolution": f"{width}x{height}",
                    "scale_factor": round(math.sqrt(area / orig_area), 3),
                    "aspect_ratio": round(aspect, 3),
                    "aspect_diff": round(abs(aspect - orig_aspect), 3),
                    "area_diff": abs(area - orig_area),
                    "total_pixels": f"{area:,}",
                    "_width": width,
                    "_height": height,
                    "_area": area,
                }
            )

        if snap_method == "closest_area":
            ranked = sorted(candidates, key=lambda item: item["area_diff"])
            info = f"Closest area match from {resolution_set}"
        elif snap_method == "closest_ratio":
            ranked = sorted(
                candidates,
                key=lambda item: (item["aspect_diff"], item["area_diff"]),
            )
            info = f"Closest aspect ratio from {resolution_set}"
        elif snap_method == "prefer_larger":
            larger = [item for item in candidates if item["_area"] >= orig_area]
            ranked = sorted(
                larger or candidates,
                key=lambda item: (
                    item["_area"] - orig_area
                    if item["_area"] >= orig_area
                    else orig_area - item["_area"]
                ),
            )
            if not larger:
                ranked = sorted(candidates, key=lambda item: item["_area"], reverse=True)
            info = f"Prefer larger from {resolution_set}"
        elif snap_method == "prefer_smaller":
            smaller = [item for item in candidates if item["_area"] <= orig_area]
            ranked = sorted(
                smaller or candidates,
                key=lambda item: (
                    orig_area - item["_area"]
                    if item["_area"] <= orig_area
                    else item["_area"] - orig_area
                ),
            )
            if not smaller:
                ranked = sorted(candidates, key=lambda item: item["_area"])
            info = f"Prefer smaller from {resolution_set}"
        else:
            raise ValueError(f"Unknown snap method: {snap_method}")

        best = ranked[0]
        public_candidates = [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in ranked[:5]
        ]
        candidates_info = {
            "method": snap_method,
            "total_evaluated": len(candidates),
            "top_5": public_candidates,
        }
        return best["_width"], best["_height"], info, candidates_info

    @staticmethod
    def _keep_proportion_snap(orig_w, orig_h, resolutions):
        orig_aspect = orig_w / orig_h
        is_portrait = orig_h > orig_w
        best_match = None
        best_score = float("inf")
        candidates = []

        for target_w, target_h in resolutions:
            if is_portrait != (target_h > target_w):
                continue

            if is_portrait:
                scale_factor = target_h / orig_h
                snapped_side = round((orig_w * scale_factor) / 64) * 64
                target_side = target_w
            else:
                scale_factor = target_w / orig_w
                snapped_side = round((orig_h * scale_factor) / 64) * 64
                target_side = target_h

            if abs(snapped_side - target_side) > 64:
                continue

            aspect_diff = abs((target_w / target_h) - orig_aspect)
            score = aspect_diff * 10 + abs(scale_factor - 1.0) * 2
            candidates.append(
                {
                    "resolution": f"{target_w}x{target_h}",
                    "scale_factor": round(scale_factor, 3),
                    "aspect_diff": round(aspect_diff, 3),
                    "score": round(score, 3),
                }
            )
            if score < best_score:
                best_score = score
                best_match = (target_w, target_h)

        if best_match is None:
            best_match = min(
                resolutions,
                key=lambda size: abs((size[0] / size[1]) - orig_aspect),
            )

        target_w, target_h = best_match
        candidates_info = {
            "method": "keep_proportion",
            "orientation": "portrait" if is_portrait else "landscape",
            "total_evaluated": len(candidates),
            "top_5": sorted(candidates, key=lambda item: item["score"])[:5],
        }
        return (
            target_w,
            target_h,
            f"Keep proportion snap from {len(resolutions)} resolutions",
            candidates_info,
        )

    def _apply_resize(self, image, target_w, target_h, resize_mode, interpolation):
        if resize_mode == "stretch":
            return self._resize_tensor(image, target_w, target_h, interpolation)
        if resize_mode == "crop_center":
            return self._crop_center_resize(image, target_w, target_h, interpolation)
        if resize_mode == "fit_pad_black":
            return self._fit_pad_resize(
                image,
                target_w,
                target_h,
                interpolation,
                pad_color=0.0,
            )
        if resize_mode == "fit_pad_white":
            return self._fit_pad_resize(
                image,
                target_w,
                target_h,
                interpolation,
                pad_color=1.0,
            )
        if resize_mode == "fit_pad_edge":
            return self._fit_pad_edge_resize(
                image,
                target_w,
                target_h,
                interpolation,
            )
        raise ValueError(f"Unknown resize mode: {resize_mode}")

    @staticmethod
    def _resize_tensor(image, width, height, interpolation):
        image_bchw = image.permute(0, 3, 1, 2)
        mode = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic",
        }.get(interpolation)
        if mode is None:
            raise ValueError(f"Unknown interpolation mode: {interpolation}")

        options = {}
        if mode in ("bilinear", "bicubic"):
            options.update(align_corners=False, antialias=True)
        resized = F.interpolate(
            image_bchw,
            size=(height, width),
            mode=mode,
            **options,
        )
        return resized.permute(0, 2, 3, 1)

    def _crop_center_resize(self, image, target_w, target_h, interpolation):
        orig_h, orig_w = image.shape[1:3]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h

        if orig_aspect > target_aspect:
            new_h = target_h
            new_w = max(target_w, math.ceil(target_h * orig_aspect))
        else:
            new_w = target_w
            new_h = max(target_h, math.ceil(target_w / orig_aspect))

        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        crop_x = max(0, (new_w - target_w) // 2)
        crop_y = max(0, (new_h - target_h) // 2)
        return resized[
            :,
            crop_y : crop_y + target_h,
            crop_x : crop_x + target_w,
            :,
        ]

    def _fit_dimensions(self, image, target_w, target_h):
        orig_h, orig_w = image.shape[1:3]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        if orig_aspect > target_aspect:
            return target_w, max(1, int(target_w / orig_aspect))
        return max(1, int(target_h * orig_aspect)), target_h

    def _fit_pad_resize(
        self,
        image,
        target_w,
        target_h,
        interpolation,
        pad_color,
    ):
        new_w, new_h = self._fit_dimensions(image, target_w, target_h)
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        return self._pad_to_target(
            resized,
            target_w,
            target_h,
            mode="constant",
            value=pad_color,
        )

    def _fit_pad_edge_resize(
        self,
        image,
        target_w,
        target_h,
        interpolation,
    ):
        new_w, new_h = self._fit_dimensions(image, target_w, target_h)
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        return self._pad_to_target(
            resized,
            target_w,
            target_h,
            mode="replicate",
        )

    @staticmethod
    def _pad_to_target(image, target_w, target_h, mode, value=0.0):
        current_h, current_w = image.shape[1:3]
        pad_w = max(0, target_w - current_w)
        pad_h = max(0, target_h - current_h)
        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        )
        if not any(padding):
            return image

        image_bchw = image.permute(0, 3, 1, 2)
        if mode == "constant":
            padded = F.pad(image_bchw, padding, mode=mode, value=value)
        else:
            padded = F.pad(image_bchw, padding, mode=mode)
        return padded.permute(0, 2, 3, 1)


NODE_CLASS_MAPPINGS_IMPORTED_APEX_APEX = {
    "ApexSmartResize": CP_ApexSmartResize,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_APEX_APEX = {
    "ApexSmartResize": "Apex Smart Resize",
}
