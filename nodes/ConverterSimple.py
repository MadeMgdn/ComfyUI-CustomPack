import math


categoryFolder = "Custom pack/Converter/Simple"


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def _to_string(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _to_int(value):
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0
        lowered = text.lower()
        if lowered in ("true", "yes", "y", "on"):
            return 1
        if lowered in ("false", "no", "n", "off", "none", "null"):
            return 0
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except (ValueError, OverflowError):
                return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _to_float(value):
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0.0
        lowered = text.lower()
        if lowered in ("true", "yes", "y", "on"):
            return 1.0
        if lowered in ("false", "no", "n", "off", "none", "null"):
            return 0.0
        try:
            return float(text)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _to_rounded_int(value, rounding):
    rounding_functions = {
        "round": round,
        "ceiling": math.ceil,
        "floor": math.floor,
    }
    rounding_function = rounding_functions.get(rounding, round)

    try:
        return int(rounding_function(_to_float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _to_rounded_float(value, digits):
    try:
        return float(round(_to_float(value), int(digits)))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _to_boolean(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "false", "0", "0.0", "no", "n", "off", "none", "null"):
            return False
        if text in ("true", "1", "1.0", "yes", "y", "on"):
            return True
        try:
            return float(text) != 0.0
        except ValueError:
            pass
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


class CP_AnythingToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_string(value),)


class CP_AnythingToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_int(value),)


class CP_AnythingToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_float(value),)


class CP_AnythingToBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY_TYPE, {"forceInput": True})}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_boolean(value),)


class CP_StringToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (value,)


class CP_StringToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"forceInput": True}),
                "rounding": (
                    ["round", "ceiling", "floor"],
                    {"default": "round"},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value, rounding="round"):
        return (_to_rounded_int(value, rounding),)


class CP_StringToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {"forceInput": True}),
                "digits": (
                    "INT",
                    {"default": 1, "min": 0, "max": 15, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value, digits=1):
        return (_to_rounded_float(value, digits),)


class CP_StringToBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_boolean(value),)


class CP_IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_string(value),)


class CP_IntToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (value,)


class CP_IntToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"forceInput": True}),
                "digits": (
                    "INT",
                    {"default": 1, "min": 0, "max": 15, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value, digits=1):
        return (_to_rounded_float(value, digits),)


class CP_IntToBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"forceInput": True})}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_boolean(value),)


class CP_FloatToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_string(value),)


class CP_FloatToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"forceInput": True}),
                "rounding": (
                    ["round", "ceiling", "floor"],
                    {"default": "round"},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value, rounding="round"):
        return (_to_rounded_int(value, rounding),)


class CP_FloatToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (value,)


class CP_FloatToBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"forceInput": True})}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_boolean(value),)


class CP_BooleanToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_string(value),)


class CP_BooleanToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_int(value),)


class CP_BooleanToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (_to_float(value),)


class CP_BooleanToAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("BOOLEAN", {"forceInput": True})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder

    def convert(self, value):
        return (value,)


NODE_CLASS_MAPPINGS_CONVERTERSIMPLE = {
    "CP Anything To String": CP_AnythingToString,
    "CP Anything To Int": CP_AnythingToInt,
    "CP Anything To Float": CP_AnythingToFloat,
    "CP Anything To Boolean": CP_AnythingToBoolean,
    "CP String To Anything": CP_StringToAnything,
    "CP String To Int": CP_StringToInt,
    "CP String To Float": CP_StringToFloat,
    "CP String To Boolean": CP_StringToBoolean,
    "CP Int To String": CP_IntToString,
    "CP Int To Anything": CP_IntToAnything,
    "CP Int To Float": CP_IntToFloat,
    "CP Int To Boolean": CP_IntToBoolean,
    "CP Float To String": CP_FloatToString,
    "CP Float To Int": CP_FloatToInt,
    "CP Float To Anything": CP_FloatToAnything,
    "CP Float To Boolean": CP_FloatToBoolean,
    "CP Boolean To String": CP_BooleanToString,
    "CP Boolean To Int": CP_BooleanToInt,
    "CP Boolean To Float": CP_BooleanToFloat,
    "CP Boolean To Anything": CP_BooleanToAnything,
}

NODE_DISPLAY_NAME_MAPPINGS_CONVERTERSIMPLE = {
    "CP Anything To String": "CP Anything To String",
    "CP Anything To Int": "CP Anything To Int",
    "CP Anything To Float": "CP Anything To Float",
    "CP Anything To Boolean": "CP Anything To Boolean",
    "CP String To Anything": "CP String To Anything",
    "CP String To Int": "CP String To Int",
    "CP String To Float": "CP String To Float",
    "CP String To Boolean": "CP String To Boolean",
    "CP Int To String": "CP Int To String",
    "CP Int To Anything": "CP Int To Anything",
    "CP Int To Float": "CP Int To Float",
    "CP Int To Boolean": "CP Int To Boolean",
    "CP Float To String": "CP Float To String",
    "CP Float To Int": "CP Float To Int",
    "CP Float To Anything": "CP Float To Anything",
    "CP Float To Boolean": "CP Float To Boolean",
    "CP Boolean To String": "CP Boolean To String",
    "CP Boolean To Int": "CP Boolean To Int",
    "CP Boolean To Float": "CP Boolean To Float",
    "CP Boolean To Anything": "CP Boolean To Anything",
}
