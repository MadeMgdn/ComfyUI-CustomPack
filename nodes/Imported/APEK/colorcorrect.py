#Part code get from https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet
from PIL import Image, ImageEnhance, ImageColor, ImageOps
import numpy as np
import torch

categoryFolder = "Custom pack/Imported/AlekPet"

class ColorsCorrectNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.01},
                ),
                "use_color": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "hex_color": (
                    "STRING",
                    {"default": "#00FF33"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct"
    DESCRIPTION = "Node for correcting image colors."      
    CATEGORY = categoryFolder

    @staticmethod
    def hex_to_rgb(hex_color):
        return ImageColor.getcolor(hex_color, "RGB")

    @staticmethod
    def hex_to_hue(hex_color):
        rgb = ImageColor.getcolor(hex_color, "RGB")
        r, g, b = [x / 255.0 for x in rgb]
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360

        return h

    @staticmethod
    def adjust_brightness(image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_saturation(image, factor):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        lut = [pow(x / 255.0, inv_gamma) * 255 for x in range(256)]
        lut = np.array(lut * 3, dtype=np.uint8)
        return image.point(lut)

    @staticmethod
    def degrees_to_hue(degrees):
        degrees = degrees % 360
        hue = degrees / 360.0

        if hue > 0.5:
            hue -= 1.0
        if hue < -0.5:
            hue += 1.0

        return hue

    @staticmethod
    def adjust_hue(image, hue):
        if not (-0.5 <= hue <= 0.5):
            raise ValueError("hue value is not in [-0.5, 0.5].")

        image_array = np.array(image.convert("RGB"), dtype=np.uint8)

        hsv_image = Image.fromarray(image_array).convert("HSV")
        hsv_array = np.array(hsv_image)

        hsv_array[..., 0] = (hsv_array[..., 0].astype(int) + int(hue * 255)) % 256

        rgb_image = Image.fromarray(hsv_array, mode="HSV").convert("RGB")
        return rgb_image

    @staticmethod
    def tint_image(image, hex_color):
        return ImageOps.colorize(image.convert("L"), black="black", white=hex_color)

    def correct(
        self,
        image,
        use_color=True,
        hex_color="#00FF33",
        brightness=1.0,
        contrast=1.0,
        saturation=1.0,
        gamma=1.0,
        hue_degrees=0.0,
    ):
        processed_images = []
        hue_norm = ColorsCorrectNode.degrees_to_hue(hue_degrees)
        
        for img_tensor in image:
            i = 255.0 * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            if use_color:
                img = ColorsCorrectNode.tint_image(img, hex_color)
            
            img = ColorsCorrectNode.adjust_brightness(img, brightness)
            img = ColorsCorrectNode.adjust_contrast(img, contrast)
            img = ColorsCorrectNode.adjust_saturation(img, saturation)
            img = ColorsCorrectNode.adjust_gamma(img, gamma)
            img = ColorsCorrectNode.adjust_hue(img, hue_norm)
            
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            
            img_array = np.array(img).astype(np.float32) / 255.0
            processed_images.append(torch.from_numpy(img_array))

        result = torch.stack(processed_images, dim=0)
        
        return (result,)

NODE_CLASS_MAPPINGS_IMPORTED_APEK_COLORCORRECT = {
    "ColorsCorrectNode": ColorsCorrectNode,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_APEK_COLORCORRECT = {
    "ColorsCorrectNode": "Colors Correction",
}