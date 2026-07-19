import os
import random
from datetime import datetime, timedelta

import folder_paths
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

categoryFolder = "Custom pack/Imported/Augmentation"

class AugNode:
    CATEGORY = categoryFolder
    FUNCTION = "augment"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    OUTPUT_NODE = True

    SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ImageFolder": ("STRING", {"default": ""}),
                "Rotate": ("BOOLEAN", {"default": False}),
                "FlipH": ("BOOLEAN", {"default": False}),
                "FlipV": ("BOOLEAN", {"default": False}),
                "Brightness": ("BOOLEAN", {"default": False}),
                "Blur": ("BOOLEAN", {"default": False}),
                "Saturation": ("BOOLEAN", {"default": False}),
                "Contrast": ("BOOLEAN", {"default": False}),
                "Sharpness": ("BOOLEAN", {"default": False}),
                "Gamma": ("BOOLEAN", {"default": False}),
                "CLAHE": ("BOOLEAN", {"default": False}),
                "HSV": ("BOOLEAN", {"default": False}),
                "SSR": ("BOOLEAN", {"default": False}),
                "Count": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1}),
                "Destination": ("STRING", {"default": ""}),
            }
        }

    def _list_images(self, folder_path):
        return sorted(
            image_name
            for image_name in os.listdir(folder_path)
            if image_name.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
        )

    def _next_output_path(self, output_dir, base_time, time_index):
        while True:
            dt = base_time + timedelta(seconds=time_index)
            filename = dt.strftime("%Y-%m-%d-%H-%M-%S") + ".png"
            full_path = os.path.join(output_dir, filename)
            if not os.path.exists(full_path):
                return full_path, time_index + 1
            time_index += 1

    def _apply_gamma(self, image):
        gamma = random.uniform(0.7, 1.5)
        inverse_gamma = 1.0 / gamma
        lut = [int(((value / 255.0) ** inverse_gamma) * 255) for value in range(256)]
        return image.point(lut * len(image.getbands()))

    def _apply_hsv_adjustment(self, image):
        hue_shift = random.randint(-15, 15)
        saturation_factor = random.uniform(0.8, 1.3)
        hue, saturation, value = image.convert("HSV").split()
        hue = hue.point(lambda pixel: (pixel + hue_shift) % 256)
        saturation = saturation.point(lambda pixel: max(0, min(255, int(pixel * saturation_factor))))
        return Image.merge("HSV", (hue, saturation, value)).convert("RGB")

    def _apply_clahe_like(self, image):
        luma, cb, cr = image.convert("YCbCr").split()
        equalized_luma = ImageOps.equalize(luma)
        return Image.merge("YCbCr", (equalized_luma, cb, cr)).convert("RGB")

    def _apply_ssr(self, image):
        width, height = image.size

        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.9, 1.1)
        max_dx = int(width * 0.08)
        max_dy = int(height * 0.08)
        shift_x = random.randint(-max_dx, max_dx)
        shift_y = random.randint(-max_dy, max_dy)

        output = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        )

        if abs(scale - 1.0) > 1e-6:
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            scaled = output.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

            if scale >= 1.0:
                left = max(0, (new_w - width) // 2)
                top = max(0, (new_h - height) // 2)
                output = scaled.crop((left, top, left + width, top + height))
            else:
                canvas = Image.new("RGB", (width, height), (0, 0, 0))
                left = (width - new_w) // 2
                top = (height - new_h) // 2
                canvas.paste(scaled, (left, top))
                output = canvas

        output = ImageChops.offset(output, shift_x, shift_y)
        if shift_x > 0:
            output.paste((0, 0, 0), (0, 0, shift_x, height))
        elif shift_x < 0:
            output.paste((0, 0, 0), (width + shift_x, 0, width, height))
        if shift_y > 0:
            output.paste((0, 0, 0), (0, 0, width, shift_y))
        elif shift_y < 0:
            output.paste((0, 0, 0), (0, height + shift_y, width, height))

        return output

    def _apply_augmentations(
        self,
        image,
        Rotate,
        FlipH,
        FlipV,
        Brightness,
        Blur,
        Saturation,
        Contrast,
        Sharpness,
        Gamma,
        CLAHE,
        HSV,
        SSR,
    ):
        output = image

        if Rotate:
            output = output.transpose(Image.Transpose.ROTATE_90)
        if FlipH:
            output = ImageOps.mirror(output)
        if FlipV:
            output = ImageOps.flip(output)
        if SSR:
            output = self._apply_ssr(output)

        if Blur:
            output = output.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if Saturation:
            output = ImageEnhance.Color(output).enhance(random.uniform(0.6, 1.4))
        if Contrast:
            output = ImageEnhance.Contrast(output).enhance(random.uniform(0.6, 1.4))
        if Sharpness:
            output = ImageEnhance.Sharpness(output).enhance(random.uniform(1.2, 2.0))
        if Gamma:
            output = self._apply_gamma(output)
        if CLAHE:
            output = self._apply_clahe_like(output)
        if HSV:
            output = self._apply_hsv_adjustment(output)
        if Brightness:
            output = ImageEnhance.Brightness(output).enhance(random.uniform(0.7, 1.3))

        return output

    def augment(
        self,
        ImageFolder,
        Rotate,
        FlipH,
        FlipV,
        Brightness,
        Blur,
        Saturation,
        Contrast,
        Sharpness,
        Gamma,
        CLAHE,
        HSV,
        SSR,
        Count,
        Destination,
    ):
        if not ImageFolder:
            return ("ImageFolder is empty.",)
        if not os.path.isdir(ImageFolder):
            return (f"ImageFolder does not exist: {ImageFolder}",)

        original_images = self._list_images(ImageFolder)
        if not original_images:
            return ("No images found in ImageFolder.",)

        augmentations_needed = Count - len(original_images)
        if augmentations_needed <= 0:
            return (
                f"No additional images needed. Current images: {len(original_images)}, requested total: {Count}.",
            )

        output_dir = ""
        if Destination == "":
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = Destination
        os.makedirs(output_dir, exist_ok=True)

        generated = 0
        generated_this_pass = 0
        base_time = datetime.now()
        time_index = 0

        while generated < augmentations_needed:
            generated_this_pass = 0
            for image_name in original_images:
                if generated >= augmentations_needed:
                    break

                source_path = os.path.join(ImageFolder, image_name)
                try:
                    with Image.open(source_path) as source_image:
                        image = source_image.convert("RGB")
                except Exception:
                    continue

                try:
                    augmented = self._apply_augmentations(
                        image,
                        Rotate,
                        FlipH,
                        FlipV,
                        Brightness,
                        Blur,
                        Saturation,
                        Contrast,
                        Sharpness,
                        Gamma,
                        CLAHE,
                        HSV,
                        SSR,
                    )
                    output_path, time_index = self._next_output_path(output_dir, base_time, time_index)
                    augmented.save(output_path, format="PNG")
                    generated += 1
                    generated_this_pass += 1
                except Exception:
                    continue

            if generated_this_pass == 0:
                return ("Could not generate augmented images from the provided folder.",)

        return (f"Saved {generated} augmented image(s) to: {output_dir}",)

NODE_CLASS_MAPPINGS_IMPORTED_AUGNODE_AUGNODE = {
    "CP AugNode": AugNode,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_AUGNODE_AUGNODE = {
    "CP AugNode": "✏️ Augmentation Images",
}
