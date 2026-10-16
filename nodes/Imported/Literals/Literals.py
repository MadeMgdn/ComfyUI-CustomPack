categoryFolder = "Custom pack/Imported/Literals"


class IntLiteral:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Number": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "to_int"
    CATEGORY = categoryFolder

    def to_int(self, Number):
        try:
            ret_val = int(Number)
        except Exception as exc:
            raise ValueError("Invalid value provided for INT") from exc
        return (ret_val,)


class FloatLiteral:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Number": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "to_float"
    CATEGORY = categoryFolder

    def to_float(self, Number):
        try:
            ret_val = float(Number)
        except Exception as exc:
            raise ValueError("Invalid value provided for FLOAT") from exc
        return (ret_val,)


class StringLiteral:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "String": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"
    CATEGORY = categoryFolder

    def to_string(self, String):
        return (String,)


NODE_CLASS_MAPPINGS_IMPORTED_LITERALS_LITERALS = {
    "Int": IntLiteral,
    "Float": FloatLiteral,
    "String": StringLiteral,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_LITERALS_LITERALS = {
    "Int": "Int",
    "Float": "Float",
    "String": "String",
}
