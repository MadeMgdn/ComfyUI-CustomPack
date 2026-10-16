import os
import re
import socket
import time
from io import StringIO

import comfy.model_management as model_management


categoryFolder = "Custom pack/Imported/WAS"


_ESCAPED_TOKEN = re.compile(r"\\(\[[^\]]*\])")
_CUSTOM_TIME_TOKEN = re.compile(r"\[time\((.*?)\)\]")


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


NODE_CLASS_MAPPINGS_IMPORTED_WAS_WAS = {
    "Text Multiline": WAS_TextMultiline,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_WAS_WAS = {
    "Text Multiline": "Text Multiline",
}
