import glob
import hashlib
import os
import random
import re
import socket
import time
from io import StringIO
from threading import Lock

import comfy.model_management as model_management
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps


categoryFolder = "Custom pack/Imported/WAS"


_ESCAPED_TOKEN = re.compile(r"\\(\[[^\]]*\])")
_CUSTOM_TIME_TOKEN = re.compile(r"\[time\((.*?)\)\]")
_WAS_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")
_WAS_MAX_BATCH_IMAGES = 4096
_WAS_FOLDER_CACHE = (0.0, ())


def _was_folder_roots():
    roots = {}
    getters = (
        ("input", "get_input_directory"),
        ("output", "get_output_directory"),
        ("temp", "get_temp_directory"),
    )
    for name, getter_name in getters:
        getter = getattr(folder_paths, getter_name, None)
        if getter is None:
            continue
        try:
            directory = os.path.realpath(getter())
        except Exception:
            continue
        if directory:
            roots[name] = directory
    return roots


def _was_folder_options():
    global _WAS_FOLDER_CACHE

    now = time.monotonic()
    expires_at, cached = _WAS_FOLDER_CACHE
    if cached and now < expires_at:
        return list(cached)

    roots = _was_folder_roots()
    options = list(roots)
    remaining = 2000
    for name, root in roots.items():
        if remaining <= 0 or not os.path.isdir(root):
            continue
        try:
            for current, directories, _files in os.walk(root, followlinks=False):
                directories[:] = sorted(
                    directory
                    for directory in directories
                    if not directory.startswith(".")
                    and not os.path.islink(os.path.join(current, directory))
                )
                if current == root:
                    continue
                relative = os.path.relpath(current, root).replace(os.sep, "/")
                options.append(f"{relative} [{name}]")
                remaining -= 1
                if remaining <= 0:
                    break
        except OSError:
            continue

    if not options:
        options = ["input"]
    _WAS_FOLDER_CACHE = (now + 5.0, tuple(options))
    return options


def _was_resolve_folder(label):
    chosen = str(label or "").strip()
    if not chosen:
        return None

    if chosen.endswith("]") and " [" in chosen:
        relative, _, root_name = chosen[:-1].rpartition(" [")
    else:
        relative, root_name = "", chosen

    root = _was_folder_roots().get(root_name)
    if root is None:
        return None

    target = os.path.realpath(os.path.join(root, relative))
    try:
        if os.path.commonpath((root, target)) != root:
            return None
    except ValueError:
        return None
    return target


def _was_scan_images(directory, pattern):
    selected_pattern = str(pattern or "*").strip() or "*"
    normalized_pattern = selected_pattern.replace("\\", "/")
    pattern_parts = normalized_pattern.split("/")
    if (
        os.path.isabs(selected_pattern)
        or os.path.splitdrive(selected_pattern)[0]
        or ".." in pattern_parts
    ):
        raise ValueError("pattern must stay inside the selected folder")

    root = os.path.realpath(directory)
    paths = []
    for candidate in glob.iglob(
        os.path.join(root, selected_pattern),
        recursive=True,
    ):
        if not os.path.isfile(candidate):
            continue
        if not candidate.lower().endswith(_WAS_IMAGE_EXTENSIONS):
            continue
        resolved = os.path.realpath(candidate)
        try:
            if os.path.commonpath((root, resolved)) != root:
                raise ValueError("pattern matched a file outside the selected folder")
        except ValueError as error:
            raise ValueError("pattern matched a file outside the selected folder") from error
        paths.append(resolved)

    paths.sort()
    return paths


def _was_image_to_tensor(path, allow_rgba):
    with Image.open(path) as opened:
        opened.seek(0)
        image = ImageOps.exif_transpose(opened)
        image = image.convert("RGBA" if allow_rgba else "RGB")
        pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)


def _was_file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_text_tokens(text):
    held = []

    def hold_escaped_token(match):
        held.append(match.group(1))
        return f"\x00was-escaped-token-{len(held) - 1}\x00"

    expanded = _ESCAPED_TOKEN.sub(hold_escaped_token, text)

    try:
        device = model_management.get_torch_device()
        device_name = model_management.get_torch_device_name(device=device)
    except Exception:
        return text

    try:
        user = os.getlogin() or "null"
    except Exception:
        user = "null"

    tokens = {
        "[time]": str(int(time.time())),
        "[hostname]": socket.gethostname(),
        "[cuda_device]": str(device),
        "[cuda_name]": str(device_name),
        "[user]": user,
    }

    for token, value in tokens.items():
        expanded = expanded.replace(token, value)

    expanded = _CUSTOM_TIME_TOKEN.sub(
        lambda match: time.strftime(match.group(1), time.localtime()),
        expanded,
    )

    for index, literal in enumerate(held):
        expanded = expanded.replace(f"\x00was-escaped-token-{index}\x00", literal)

    return expanded


class WAS_TextMultiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "dynamic_prompts": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_multiline"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "A multiline text box. Lines whose first non-blank character is # are removed, "
        "WAS text tokens are expanded, and dynamic prompt alternatives can be enabled or "
        "disabled."
    )

    def text_multiline(self, text, dynamic_prompts=True):
        del dynamic_prompts

        text = _expand_text_tokens(text)
        kept = []
        for line in StringIO(text):
            if not line.strip().startswith("#"):
                kept.append(line.replace("\n", ""))

        return ("\n".join(kept),)


class WAS_LoadImageBatch:
    _incremental_state = {}
    _incremental_lock = Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["single_image", "incremental_image", "random", "all_images"],
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 150000,
                        "step": 1,
                    },
                ),
                "folder": (_was_folder_options(),),
                "pattern": ("STRING", {"default": "*"}),
                "allow_RGBA_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "filename_text_extension": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_text", "image_list", "filename_list")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "load_image_batch"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Loads one image from a folder by index, in sequence or at random, or returns "
        "every matching image as a list."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        mode="single_image",
        seed=0,
        index=0,
        folder="input",
        pattern="*",
        allow_RGBA_output=False,
        filename_text_extension=True,
        unique_id=None,
    ):
        del seed, allow_RGBA_output, filename_text_extension, unique_id

        if mode not in ("single_image", "all_images"):
            return float("NaN")

        directory = _was_resolve_folder(folder)
        if directory is None or not os.path.isdir(directory):
            return float("NaN")

        try:
            paths = _was_scan_images(directory, pattern)
            if mode == "all_images":
                return repr(
                    [
                        (path, os.stat(path).st_mtime_ns, os.path.getsize(path))
                        for path in paths
                    ]
                )
            if 0 <= index < len(paths):
                return _was_file_digest(paths[index])
        except (OSError, ValueError):
            pass
        return float("NaN")

    @classmethod
    def _next_incremental_index(cls, key, directory, pattern, image_count):
        with cls._incremental_lock:
            state = cls._incremental_state.get(key)
            identity = (directory, pattern)
            if state is None or state[0] != identity:
                position = 0
            else:
                position = state[1] % image_count
            cls._incremental_state[key] = (identity, (position + 1) % image_count)
            return position

    def load_image_batch(
        self,
        mode,
        seed,
        index,
        folder,
        pattern,
        allow_RGBA_output,
        filename_text_extension=True,
        unique_id=None,
    ):
        directory = _was_resolve_folder(folder)
        if directory is None or not os.path.isdir(directory):
            raise ValueError(
                f"'{folder}' does not name an available ComfyUI folder"
            )

        paths = _was_scan_images(directory, pattern)
        if not paths:
            raise ValueError(
                f"'{folder}' contains no supported images matching '{pattern}'"
            )

        if mode == "all_images":
            selected_paths = paths[:_WAS_MAX_BATCH_IMAGES]
            images = [
                _was_image_to_tensor(path, allow_RGBA_output)
                for path in selected_paths
            ]
            names = [os.path.basename(path) for path in selected_paths]
            first_name = names[0]
            if not filename_text_extension:
                first_name = os.path.splitext(first_name)[0]
            return (
                images[0],
                first_name,
                images,
                [os.path.splitext(name)[0] for name in names],
            )

        if mode == "single_image":
            position = index
            if position < 0 or position >= len(paths):
                raise ValueError(
                    f"image index {position} is outside the available range "
                    f"0 to {len(paths) - 1}"
                )
        elif mode == "incremental_image":
            cursor_key = str(unique_id) if unique_id is not None else str(id(self))
            position = self._next_incremental_index(
                cursor_key,
                directory,
                pattern,
                len(paths),
            )
        elif mode == "random":
            position = int(random.Random(seed).random() * len(paths))
        else:
            raise ValueError(f"unsupported Load Image Batch mode: {mode}")

        path = paths[position]
        image = _was_image_to_tensor(path, allow_RGBA_output)
        name = os.path.basename(path)
        filename_text = name if filename_text_extension else os.path.splitext(name)[0]
        return (
            image,
            filename_text,
            [image],
            [os.path.splitext(name)[0]],
        )


NODE_CLASS_MAPPINGS_IMPORTED_WAS_WAS = {
    "Text Multiline": WAS_TextMultiline,
    "Load Image Batch": WAS_LoadImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_WAS_WAS = {
    "Text Multiline": "Text Multiline",
    "Load Image Batch": "Load Image Batch",
}
