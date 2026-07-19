from __future__ import annotations
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
import numpy as np
import torch
import torchvision.transforms.functional as TF
import io
import os
import folder_paths
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from nodes import SaveImage
from PIL.PngImagePlugin import PngInfo

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy.cli_args import args as comfy_args
except Exception:
    comfy_args = None

categoryFolder = "Custom pack/Image"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")
_CP_CIVITAI_HASH_CACHE = {}
_CP_CIVITAI_CHECKPOINT_DB_CACHE = None

#Global
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]
    return TF.rgb_to_grayscale(t.permute(0,3,1,2), num_output_channels=1)[:,0,:,:]


#Crops an image by the specified number of pixels on each side. Zero values ​​are allowed for image overflow.
class CP_ImageCrop:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = categoryFolder

    def crop_image(self, image, top, left, right, bottom):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        width, height = img.size
        left_crop = left
        top_crop = top
        right_crop = width - right
        bottom_crop = height - bottom
        if left_crop >= right_crop:
            left_crop = 0
            right_crop = width
        if top_crop >= bottom_crop:
            top_crop = 0
            bottom_crop = height

        img = img.crop((left_crop, top_crop, right_crop, bottom_crop))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]
        
        return (img,)

#Combines two images by mask. I tried to combine them using a mask, but it didn't work.
class CP_ImageConcatMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "image_concat_mask"
    CATEGORY = categoryFolder

    def image_concat_mask(self, image1, image2=None, mask=None):
        processed_images = []
        masks = []
        
        for idx, img1 in enumerate(image1):
            pil_image1 = tensor2pil(img1)
            width1, height1 = pil_image1.size
            
            if image2 is not None and idx < len(image2):
                pil_image2 = tensor2pil(image2[idx])
                width2, height2 = pil_image2.size
                new_width2 = int(width2 * (height1 / height2))
                pil_image2 = pil_image2.resize((new_width2, height1), Image.Resampling.LANCZOS)
            else:
                pil_image2 = Image.new('RGB', (width1, height1), 'white')
                new_width2 = width1
            combined_image = Image.new('RGB', (width1 + new_width2, height1))
            
            combined_image.paste(pil_image1, (0, 0))
            combined_image.paste(pil_image2, (width1, 0))

            combined_tensor = pil2tensor(combined_image)
            processed_images.append(combined_tensor)
            
            #This *** doesn't work here - Wanted to make a connection by sections
            final_mask = torch.zeros((1, height1, width1 + new_width2))
            final_mask[:, :, width1:] = 1.0
            
            if mask is not None and idx < len(mask):
                input_mask = mask[idx]
                pil_input_mask = tensor2pil(input_mask)
                pil_input_mask = pil_input_mask.resize((new_width2, height1), Image.Resampling.LANCZOS)
                resized_input_mask = pil2tensor(pil_input_mask)
                final_mask[:, :, width1:] *= (1.0 - resized_input_mask)
            
            masks.append(final_mask)
            
        processed_images = torch.cat(processed_images, dim=0)
        masks = torch.cat(masks, dim=0)
        
        return (processed_images, masks)

#Reflects an image and outputs one or another based on a condition (needed for texturing 3D models)
class CP_FlipImagePair:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "use_flip": ("BOOLEAN", {"default": False}),
                "flipImage": (["a", "b"], {"default": "a"}),
                "flipType": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_a_out", "image_b_out")
    FUNCTION = "flip_images"
    CATEGORY = categoryFolder

    def flip_images(self, image_a, image_b, use_flip, flipImage, flipType):
        def tensor_to_pil(tensor):
            i = 255. * tensor.cpu().numpy().squeeze()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return img

        def pil_to_tensor(img):
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return img
            
        img_a = tensor_to_pil(image_a)
        img_b = tensor_to_pil(image_b)
        
        if not use_flip:
            return (image_a, image_b)
            
        if flipImage == "a":
            base_img = img_a
        else:
            base_img = img_b

        if flipType == "horizontal":
            flipped_img = ImageOps.mirror(base_img)
        else:
            flipped_img = ImageOps.flip(base_img)

        base_tensor = pil_to_tensor(base_img)
        flipped_tensor = pil_to_tensor(flipped_img)
        
        if flipImage == "a":
            return (base_tensor, flipped_tensor)
        else:
            return (flipped_tensor, base_tensor)

#Draws a bbox frame on the recognized area of ​​the image and displays information about the size of the occupied space (needed for preparing the dataset before training)
class CP_DrawDetections:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detections": ("BBOX_DETECTOR",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "expand_left": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "expand_right": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "expand_top": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "expand_bottom": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
            },
            "optional": {
                "font_size": ("INT", {"default": 15, "min": 1, "max": 100}),
                "color": ("COLOR", {"default": "red"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = categoryFolder

    def draw(self, image, detections, threshold, dilation, crop_factor, 
             expand_left, expand_right, expand_top, expand_bottom, 
             font_size=15, color="red"):
        pil_image = self.tensor_to_pil(image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        img_width, img_height = pil_image.size
        total_pixels = img_width * img_height
        
        shape, seg_items = detections.detect(
            image=image,
            threshold=threshold,
            dilation=dilation,
            crop_factor=crop_factor
        )

        for seg in seg_items:
            label = seg.label
            conf_val = seg.confidence
            if hasattr(conf_val, 'item'):
                conf_val = conf_val.item()
            elif isinstance(conf_val, (list, tuple, np.ndarray)) and len(conf_val) == 1:
                conf_val = conf_val[0]
            confidence = float(conf_val)
            
            x1, y1, x2, y2 = seg.bbox
            
            x1 = max(0, x1 - expand_left)
            x2 = min(img_width, x2 + expand_right)
            y1 = max(0, y1 - expand_top)
            y2 = min(img_height, y2 + expand_bottom)
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            width_percent = (bbox_width / img_width) * 100
            height_percent = (bbox_height / img_height) * 100
            area_percent = (bbox_area / total_pixels) * 100
            position = ""
            if area_percent > 12:
                position = "Close-Up"
            elif area_percent >= 3 and area_percent <= 12:
                position = "Middle"
            else:
                position = "Common"
            
            info_text = [
                f"{label.upper()} {confidence:.0%}; {position}",
                f"Width: {int(bbox_width)}px ({width_percent:.1f}%)",
                f"Height: {int(bbox_height)}px ({height_percent:.1f}%)",
                f"Area: {int(bbox_area)}px² ({area_percent:.1f}%)"
            ]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text_block_height = len(info_text) * font_size + 4
            text_bg = [
                x1 - 0,
                y1 - text_block_height - 0,
                x1 + max(len(line) for line in info_text) * font_size * 0.6,
                y1 - 0
            ]
            draw.rectangle(text_bg, fill=color)
            
            for i, line in enumerate(info_text):
                draw.text(
                    (x1, y1 - text_block_height + i * font_size),
                    line,
                    fill="white",
                    font=font
                )

        return (self.pil_to_tensor(pil_image),)

    def tensor_to_pil(self, tensor):
        tensor = tensor.squeeze(0).cpu().numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

    def pil_to_tensor(self, image):
        image = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image).unsqueeze(0)

#Adds jpeg artifacts
class CP_AddJpegArtifacts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_artifacts": ("BOOLEAN", {
                    "default": True,
                    "label_off": "OFF",
                    "label_on": "ON"
                }),
                "quality": ("INT", {
                    "default": 75,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "subsampling": (["default", "4:4:4", "4:2:2", "4:2:0", "4:1:1"], {
                    "default": "4:2:0"
                }),
                "dct_method": (["slow", "fast"], {
                    "default": "slow"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "apply_jpeg_artifacts"
    CATEGORY = categoryFolder

    def apply_jpeg_artifacts(self, image, enable_artifacts, quality, subsampling, dct_method):
        if not enable_artifacts:
            return self.return_original_image(image)
        subsampling_map = {
            "default": -1,
            "4:4:4": 0,
            "4:2:2": 1,
            "4:2:0": 2,
            "4:1:1": 3,
        }
        pil_subsampling = subsampling_map[subsampling]
        
        batch_size = image.shape[0]
        out_images = []
        out_masks = []
        
        for i in range(batch_size):
            img_tensor = image[i].cpu()
            img_array = (img_tensor.numpy() * 255.0).astype(np.uint8)
            has_alpha = img_array.shape[2] == 4
            
            if has_alpha:
                rgb_array = img_array[:, :, :3]
                alpha_array = img_array[:, :, 3]
                rgb_image = Image.fromarray(rgb_array, mode='RGB')
                alpha_image = Image.fromarray(alpha_array, mode='L')
                alpha_mask = torch.from_numpy(np.array(alpha_image).astype(np.float32) / 255.0)
                alpha_preserved = alpha_image.copy()
            else:
                rgb_array = img_array
                rgb_image = Image.fromarray(rgb_array, mode='RGB')
                alpha_mask = torch.ones((img_array.shape[0], img_array.shape[1]), dtype=torch.float32)
            buffer = io.BytesIO()
            
            save_kwargs = {
                'format': 'JPEG',
                'quality': quality,
                'subsampling': pil_subsampling,
                'optimize': True,
            }
            
            if dct_method == "slow":
                save_kwargs['dct_method'] = 0
            else:
                save_kwargs['dct_method'] = 1
            
            try:
                rgb_image.save(buffer, **save_kwargs)
            except (TypeError, ValueError) as e:
                save_kwargs.pop('dct_method', None)
                save_kwargs.pop('subsampling', None)
                rgb_image.save(buffer, **save_kwargs)
            
            buffer.seek(0)
            
            jpeg_image = Image.open(buffer)
            jpeg_array = np.array(jpeg_image.convert('RGB'))
            
            if has_alpha:
                alpha_array_restored = np.array(alpha_preserved)
                rgba_array = np.dstack([jpeg_array, alpha_array_restored])
                rgba_tensor = torch.from_numpy(rgba_array.astype(np.float32) / 255.0)
                out_images.append(rgba_tensor)
                out_masks.append(alpha_mask)
            else:
                rgb_tensor = torch.from_numpy(jpeg_array.astype(np.float32) / 255.0)
                out_images.append(rgb_tensor)
                out_masks.append(alpha_mask)
        out_images_batch = torch.stack(out_images, dim=0)
        out_masks_batch = torch.stack(out_masks, dim=0)
        return (out_images_batch, out_masks_batch)
    
    def return_original_image(self, image):
        batch_size = image.shape[0]
        out_masks = []
        for i in range(batch_size):
            img_tensor = image[i].cpu()
            img_array = (img_tensor.numpy() * 255.0).astype(np.uint8)
            has_alpha = img_array.shape[2] == 4
            if has_alpha:
                alpha_array = img_array[:, :, 3]
                alpha_mask = torch.from_numpy(alpha_array.astype(np.float32) / 255.0)
            else:
                alpha_mask = torch.ones((img_array.shape[0], img_array.shape[1]), dtype=torch.float32)
            out_masks.append(alpha_mask)
        out_masks_batch = torch.stack(out_masks, dim=0)
        return (image, out_masks_batch)

#Crops the image on the sides
class CP_ImageCrop:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "crop_image"
    CATEGORY = categoryFolder
    
    def crop_image(self, image, top, left, right, bottom):
        batch_size, height, width, channels = image.shape
        
        new_height = height - top - bottom
        new_width = width - left - right
        
        if new_height <= 0:
            raise ValueError(f"Invalid crop: resulting height {new_height} <= 0. "
                           f"Top ({top}) + bottom ({bottom}) >= original height ({height})")
        if new_width <= 0:
            raise ValueError(f"Invalid crop: resulting width {new_width} <= 0. "
                           f"Left ({left}) + right ({right}) >= original width ({width})")

        cropped_images = []
        for i in range(batch_size):
            img = image[i]
            cropped = img[top:height-bottom, left:width-right, :]
            if cropped.shape[2] != channels:
                raise ValueError(f"Channel mismatch after cropping: expected {channels}, got {cropped.shape[2]}")
            cropped_images.append(cropped)
        result = torch.stack(cropped_images, dim=0)
        
        return (result,)

#Rotate image. Part code get from https://github.com/WASasquatch/was-node-suite-comfyui
class CP_ImageRotate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["transpose", "internal",],),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 90}),
                "sampler": (["nearest", "bilinear", "bicubic"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rotate_image"
    CATEGORY = categoryFolder

    def rotate_image(self, images, mode, rotation, sampler):

        batch_tensor = []
        for image in images:
            image = tensor2pil(image)

            if rotation > 360:
                rotation = int(360)
            if (rotation % 90 != 0):
                rotation = int((rotation//90)*90)

            if sampler:
                if sampler == 'nearest':
                    sampler = Image.NEAREST
                elif sampler == 'bicubic':
                    sampler = Image.BICUBIC
                elif sampler == 'bilinear':
                    sampler = Image.BILINEAR
                else:
                    sampler == Image.BILINEAR

            if mode == 'internal':
                image = image.rotate(rotation, sampler)
            else:
                rot = int(rotation / 90)
                for _ in range(rot):
                    image = image.transpose(2)

            batch_tensor.append(pil2tensor(image))

        batch_tensor = torch.cat(batch_tensor, dim=0)
        return (batch_tensor, )

#Crop imeage to mulitple Xpx
class CP_CropImageToME:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "side_width": (["left", "right"], {"default": "right"}),
                "side_height": (["top", "bottom"], {"default": "bottom"}),
                "divideBy": ([1, 8, 16, 64], {"default": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = categoryFolder

    def crop_image(self, image, side_width, side_height, divideBy):
        B, H, W, C = image.shape
        k = divideBy

        target_w = (W // k) * k
        target_h = (H // k) * k

        if target_w == W and target_h == H:
            return (image,)

        crop_w = W - target_w
        crop_h = H - target_h

        if side_width == "left":
            x_start = crop_w
            x_end = W
        else: 
            x_start = 0
            x_end = W - crop_w

        if side_height == "top":
            y_start = crop_h
            y_end = H
        else:
            y_start = 0
            y_end = H - crop_h

        cropped = image[:, y_start:y_end, x_start:x_end, :]

        return (cropped,)

#Save image in bulk work for create dataset train lora
class CP_ImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "image.png"}),
                "subfolder": ("STRING", {"default": ""}),
                "filePrefix": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ()
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    CATEGORY = categoryFolder

    def save_image(self, image, filename, subfolder, filePrefix):
        base_name = os.path.splitext(filename)[0]
        if base_name == "":
            base_name = "image"

        output_dir = folder_paths.get_output_directory()
        saved_images = []
        if image.dim() == 3:
            image = image.unsqueeze(0)
        dst = os.path.join(output_dir, subfolder)
        os.makedirs(dst, exist_ok=True)

        batch_size = image.shape[0]
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            if batch_size > 1:
                final_filename = f"{base_name}{filePrefix}_{i+1:05d}.png"
            else:
                final_filename = f"{base_name}{filePrefix}.png"
            full_path = os.path.join(dst, final_filename)
            pil_image.save(full_path, compress_level=4)
            saved_images.append({
                "filename": final_filename,
                "subfolder": subfolder,
                "type": "output"
            })
        return {"ui": {"images": saved_images}}

#Assembly images into grid for compares
class CP_ImageGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_folder": ("STRING", {"default": "", "multiline": False}),
                "subfolder_1": ("STRING", {"default": "", "multiline": False}),
                "subfolder_2": ("STRING", {"default": "", "multiline": False}),
                "subfolder_3": ("STRING", {"default": "", "multiline": False}),
                "subfolder_4": ("STRING", {"default": "", "multiline": False}),
                "subfolder_5": ("STRING", {"default": "", "multiline": False}),
                "subfolder_6": ("STRING", {"default": "", "multiline": False}),
                "subfolder_7": ("STRING", {"default": "", "multiline": False}),
                "subfolder_8": ("STRING", {"default": "", "multiline": False}),
                "subfolder_9": ("STRING", {"default": "", "multiline": False}),
                "subfolder_10": ("STRING", {"default": "", "multiline": False}),
                "border_thickness": ("INT", {"default": 1, "min": 0, "max": 128, "step": 1}),
                "border_color": ("COLOR", {"default": "red"}),
                "columns": ("INT", {"default": 10, "min": 1, "max": 128, "step": 1}),
                "max_cell_resolution": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 1}),
                "background_color": ("COLOR", {"default": "black"}),
                "font_row_text": ("STRING", {"default": "E[N]", "multiline": False}),
                "font_row_start": ("FLOAT", {"default": 10.0, "min": -1000000.0, "max": 1000000.0, "step": 0.01}),
                "font_row_step": ("FLOAT", {"default": 3.0, "min": -1000000.0, "max": 1000000.0, "step": 0.01}),
                "font_row_color": ("COLOR", {"default": "white"}),
                "font_row_size": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
                "font_row_path": ("STRING", {"default": "arial.ttf", "multiline": False}),
                "font_col_text": ("STRING", {"default": "[N]", "multiline": False}),
                "font_col_color": ("COLOR", {"default": "white"}),
                "font_col_size": ("INT", {"default": 24, "min": 1, "max": 1024, "step": 1}),
                "font_col_path": ("STRING", {"default": "arial.ttf", "multiline": False}),
                "font_col_start": ("FLOAT", {"default": 0.5, "min": -1000000.0, "max": 1000000.0, "step": 0.01}),
                "font_col_step": ("FLOAT", {"default": 0.1, "min": -1000000.0, "max": 1000000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_grid"
    CATEGORY = categoryFolder

    def build_grid(
        self,
        base_folder,
        subfolder_1,
        subfolder_2,
        subfolder_3,
        subfolder_4,
        subfolder_5,
        subfolder_6,
        subfolder_7,
        subfolder_8,
        subfolder_9,
        subfolder_10,
        border_thickness,
        border_color,
        columns,
        max_cell_resolution,
        background_color,
        font_row_text,
        font_row_start,
        font_row_step,
        font_row_color,
        font_row_size,
        font_row_path,
        font_col_text,
        font_col_color,
        font_col_size,
        font_col_path,
        font_col_start,
        font_col_step,
    ):
        subfolders = [
            subfolder_1, subfolder_2, subfolder_3, subfolder_4, subfolder_5,
            subfolder_6, subfolder_7, subfolder_8, subfolder_9, subfolder_10
        ]
        subfolders = [str(folder).strip() for folder in subfolders if str(folder).strip()]

        row_count = len(subfolders) if len(subfolders) > 0 else 1
        columns = max(1, int(columns))
        cell_size = max(8, int(max_cell_resolution))
        border = max(0, int(border_thickness))

        background_rgb = self._parse_color(background_color, (0, 0, 0))
        border_rgb = self._parse_color(border_color, (255, 0, 0))
        row_color_rgb = self._parse_color(font_row_color, (255, 255, 255))
        col_color_rgb = self._parse_color(font_col_color, (255, 255, 255))

        row_font = self._load_font(font_row_path, max(1, int(font_row_size)))
        col_font = self._load_font(font_col_path, max(1, int(font_col_size)))

        row_labels = [
            self._apply_mask(font_row_text, font_row_start + (font_row_step * i))
            for i in range(row_count)
        ]
        col_labels = [
            self._apply_mask(font_col_text, font_col_start + (font_col_step * i))
            for i in range(columns)
        ]

        measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "black"))
        row_text_width = 0
        for label in row_labels:
            width, _ = self._measure_text(measure, label, row_font)
            row_text_width = max(row_text_width, width)

        col_text_height = 0
        for label in col_labels:
            _, height = self._measure_text(measure, label, col_font)
            col_text_height = max(col_text_height, height)

        text_padding = 10
        left_col_width = max(40, row_text_width + (text_padding * 2))
        top_row_height = max(30, col_text_height + (text_padding * 2))

        widths = [left_col_width] + [cell_size] * columns
        heights = [top_row_height] + [cell_size] * row_count
        total_width = sum(widths) + border * (len(widths) + 1)
        total_height = sum(heights) + border * (len(heights) + 1)

        total_width = max(1, total_width)
        total_height = max(1, total_height)

        canvas = Image.new("RGB", (total_width, total_height), background_rgb)
        draw = ImageDraw.Draw(canvas)

        x_positions = []
        current_x = border
        for width in widths:
            x_positions.append(current_x)
            current_x += width + border

        y_positions = []
        current_y = border
        for height in heights:
            y_positions.append(current_y)
            current_y += height + border

        if border > 0:
            x_line = 0
            for i in range(len(widths) + 1):
                x_end = min(total_width - 1, x_line + border - 1)
                draw.rectangle((x_line, 0, x_end, total_height - 1), fill=border_rgb)
                if i < len(widths):
                    x_line += border + widths[i]

            y_line = 0
            for i in range(len(heights) + 1):
                y_end = min(total_height - 1, y_line + border - 1)
                draw.rectangle((0, y_line, total_width - 1, y_end), fill=border_rgb)
                if i < len(heights):
                    y_line += border + heights[i]

        supported_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
        row_files = []
        for folder_name in subfolders:
            folder_path = os.path.join(base_folder, folder_name)
            files = []
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    path = os.path.join(folder_path, file_name)
                    if os.path.isfile(path):
                        _, ext = os.path.splitext(file_name)
                        if ext.lower() in supported_ext:
                            files.append(path)
            files.sort(key=lambda x: os.path.basename(x).lower())
            row_files.append(files[:columns])

        for col_index, label in enumerate(col_labels):
            x0 = x_positions[col_index + 1]
            y0 = y_positions[0]
            w = widths[col_index + 1]
            h = heights[0]
            self._draw_text_centered(draw, x0, y0, w, h, label, col_font, col_color_rgb)

        for row_index, label in enumerate(row_labels):
            x0 = x_positions[0]
            y0 = y_positions[row_index + 1]
            w = widths[0]
            h = heights[row_index + 1]
            self._draw_text_centered(draw, x0, y0, w, h, label, row_font, row_color_rgb)

        for row_index in range(row_count):
            files = row_files[row_index] if row_index < len(row_files) else []
            for col_index, image_path in enumerate(files):
                try:
                    with Image.open(image_path) as src:
                        src = ImageOps.exif_transpose(src).convert("RGB")
                        img = self._resize_to_cell(src, cell_size)
                except Exception:
                    continue

                cell_x = x_positions[col_index + 1]
                cell_y = y_positions[row_index + 1]
                offset_x = (cell_size - img.width) // 2
                offset_y = (cell_size - img.height) // 2
                canvas.paste(img, (cell_x + offset_x, cell_y + offset_y))

        return (pil2tensor(canvas),)

    def _parse_color(self, color_value, fallback):
        if color_value is None:
            return fallback

        def to_channel(value):
            try:
                value = float(value)
            except Exception:
                return None
            if 0.0 <= value <= 1.0:
                value *= 255.0
            return int(max(0, min(255, round(value))))

        if isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
            channels = [to_channel(color_value[i]) for i in range(3)]
            if all(channel is not None for channel in channels):
                return tuple(channels)

        if isinstance(color_value, dict):
            key_sets = [
                ("r", "g", "b"),
                ("red", "green", "blue"),
            ]
            lower_map = {str(k).lower(): v for k, v in color_value.items()}
            for key_r, key_g, key_b in key_sets:
                if key_r in lower_map and key_g in lower_map and key_b in lower_map:
                    channels = [to_channel(lower_map[key_r]), to_channel(lower_map[key_g]), to_channel(lower_map[key_b])]
                    if all(channel is not None for channel in channels):
                        return tuple(channels)
            for key in ("hex", "value", "color"):
                if key in lower_map:
                    parsed = self._parse_color(lower_map[key], None)
                    if parsed is not None:
                        return parsed
            return fallback

        if isinstance(color_value, (int, float)):
            channel = to_channel(color_value)
            if channel is not None:
                return (channel, channel, channel)
            return fallback

        color_text = str(color_value).strip()
        if not color_text:
            return fallback

        rgb_text = color_text.lower()
        if rgb_text.startswith("rgb(") and rgb_text.endswith(")"):
            rgb_text = rgb_text[4:-1]
        rgb_text = rgb_text.strip()
        if (
            (rgb_text.startswith("(") and rgb_text.endswith(")"))
            or (rgb_text.startswith("[") and rgb_text.endswith("]"))
        ):
            rgb_text = rgb_text[1:-1].strip()

        if "," in rgb_text:
            parts = [part.strip() for part in rgb_text.split(",")]
            if len(parts) >= 3:
                channels = [to_channel(parts[0]), to_channel(parts[1]), to_channel(parts[2])]
                if all(channel is not None for channel in channels):
                    return tuple(channels)

        if len(color_text) == 6 and color_text.isalnum():
            color_text = f"#{color_text}"

        try:
            return ImageColor.getrgb(color_text)
        except Exception:
            return fallback

    def _load_font(self, font_path, font_size):
        requested = str(font_path).strip() if font_path is not None else ""
        candidates = []
        if requested:
            candidates.append(requested)
            name, ext = os.path.splitext(requested)
            if not ext:
                candidates.append(f"{requested}.ttf")
                candidates.append(f"{requested}.otf")
                candidates.append(f"{requested}.ttc")

            windows_dir = os.environ.get("WINDIR", "C:\\Windows")
            fonts_dir = os.path.join(windows_dir, "Fonts")
            for candidate in list(candidates):
                if not os.path.isabs(candidate):
                    candidates.append(os.path.join(fonts_dir, candidate))

        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception:
                pass

        return ImageFont.load_default()

    def _measure_text(self, draw, text, font):
        if not text:
            return (0, 0)
        left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
        return (max(0, right - left), max(0, bottom - top))

    def _format_number(self, value):
        numeric = round(float(value), 10)
        if abs(numeric - round(numeric)) < 1e-10:
            return str(int(round(numeric)))
        return f"{numeric:.10f}".rstrip("0").rstrip(".")

    def _apply_mask(self, text_mask, value):
        mask = str(text_mask) if text_mask is not None else ""
        value_text = self._format_number(value)
        return mask.replace("[N]", value_text)

    def _draw_text_centered(self, draw, x, y, width, height, text, font, color):
        if text is None:
            return
        text = str(text)
        tw, th = self._measure_text(draw, text, font)
        text_x = x + max(0, (width - tw) // 2)
        text_y = y + max(0, (height - th) // 2)
        draw.text((text_x, text_y), text, fill=color, font=font)

    def _resize_to_cell(self, image, cell_size):
        width, height = image.size
        if width <= 0 or height <= 0:
            return Image.new("RGB", (cell_size, cell_size), "black")

        scale = float(cell_size) / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

#Add text on bottom size image with white background and black text
class CP_TextIncluder:
    _font_cache: Dict[str, str] | None = None
    _fallback_label = "Default PIL"

    @classmethod
    def INPUT_TYPES(cls):
        fonts = cls._get_font_labels()
        default_font = fonts[0] if fonts else cls._fallback_label
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "font_name": (fonts, {"default": default_font}),
                "font_size": ("INT", {"default": 36, "min": 8, "max": 256, "step": 1}),
                "padding": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1}),
                "line_spacing": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "append_text"
    CATEGORY = categoryFolder

    @classmethod
    def _get_font_labels(cls) -> List[str]:
        if cls._font_cache is None:
            cls._font_cache = cls._discover_cyrillic_fonts()
        labels = list(cls._font_cache.keys())
        if cls._fallback_label not in labels:
            labels.append(cls._fallback_label)
        return labels

    @staticmethod
    def _looks_like_cyrillic_font(name: str) -> bool:
        lowered = name.lower().replace(" ", "")
        cyrillic_keywords = (
            "arial",
            "tahoma",
            "times",
            "verdana",
            "segoe",
            "calibri",
            "cambria",
            "candara",
            "consola",
            "cour",
            "georgia",
            "trebuchet",
            "lucida",
            "bookman",
            "garamond",
            "dejavu",
            "noto",
            "liberation",
            "ptsans",
            "ptserif",
            "ubuntu",
            "roboto",
            "fira",
            "opensans",
            "sourcesans",
            "sourceserif",
            "cyr",
        )
        return any(keyword in lowered for keyword in cyrillic_keywords)

    @classmethod
    def _discover_cyrillic_fonts(cls) -> Dict[str, str]:
        candidates: Dict[str, str] = {}
        font_dirs = [
            Path("C:/Windows/Fonts"),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
        ]
        extensions = {".ttf", ".otf", ".ttc"}

        for font_dir in font_dirs:
            if not font_dir.exists():
                continue
            for path in font_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                stem = path.stem
                if not cls._looks_like_cyrillic_font(stem):
                    continue
                label = stem
                if label in candidates:
                    label = f"{stem} ({path.name})"
                candidates[label] = str(path)

        if not candidates:
            return {}
        return dict(sorted(candidates.items(), key=lambda item: item[0].lower()))

    @classmethod
    def _load_font(cls, font_name: str, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if cls._font_cache is None:
            cls._font_cache = cls._discover_cyrillic_fonts()
        path = cls._font_cache.get(font_name) if cls._font_cache else None
        if path:
            try:
                return ImageFont.truetype(path, size=font_size)
            except OSError:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
        if not text:
            return 0
        return int(draw.textlength(text, font=font))

    @classmethod
    def _split_long_word(
        cls, draw: ImageDraw.ImageDraw, word: str, font: ImageFont.ImageFont, max_width: int
    ) -> List[str]:
        pieces: List[str] = []
        current = ""
        for char in word:
            probe = current + char
            if not current or cls._text_width(draw, probe, font) <= max_width:
                current = probe
            else:
                pieces.append(current)
                current = char
        if current:
            pieces.append(current)
        return pieces

    @classmethod
    def _wrap_paragraph(
        cls, draw: ImageDraw.ImageDraw, paragraph: str, font: ImageFont.ImageFont, max_width: int
    ) -> List[str]:
        if paragraph == "":
            return [""]
        words = paragraph.split()
        if not words:
            return [""]

        lines: List[str] = []
        current = ""
        for word in words:
            if current:
                candidate = f"{current} {word}"
            else:
                candidate = word

            if cls._text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
                current = ""

            if cls._text_width(draw, word, font) <= max_width:
                current = word
                continue

            chunks = cls._split_long_word(draw, word, font, max_width)
            if len(chunks) > 1:
                lines.extend(chunks[:-1])
            current = chunks[-1]

        if current or not lines:
            lines.append(current)
        return lines

    @classmethod
    def _wrap_text(
        cls, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
    ) -> List[str]:
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = normalized.split("\n")
        wrapped_lines: List[str] = []
        for paragraph in paragraphs:
            wrapped_lines.extend(cls._wrap_paragraph(draw, paragraph, font, max_width))
        return wrapped_lines if wrapped_lines else [""]

    @staticmethod
    def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        return max(1, bbox[3] - bbox[1])

    @classmethod
    def _render_text_strip(
        cls,
        width: int,
        text: str,
        font_name: str,
        font_size: int,
        padding: int,
        line_spacing: int,
    ) -> np.ndarray:
        font = cls._load_font(font_name, font_size)
        draft = Image.new("RGB", (max(1, width), 8), "white")
        draw = ImageDraw.Draw(draft)
        max_text_width = max(1, width - (padding * 2))
        lines = cls._wrap_text(draw, text, font, max_text_width)
        line_height = cls._line_height(draw, font)

        content_height = len(lines) * line_height
        if len(lines) > 1:
            content_height += (len(lines) - 1) * line_spacing
        strip_height = content_height + (padding * 2)
        strip_height = max(strip_height, line_height + 2)

        strip = Image.new("RGB", (width, strip_height), "white")
        strip_draw = ImageDraw.Draw(strip)
        y = padding
        for line in lines:
            strip_draw.text((padding, y), line, fill="black", font=font)
            y += line_height + line_spacing
        return np.asarray(strip, dtype=np.float32) / 255.0

    @classmethod
    def _adapt_strip_channels(cls, strip_rgb: torch.Tensor, channels: int) -> torch.Tensor:
        if channels == 3:
            return strip_rgb
        if channels == 4:
            alpha = torch.ones(
                (strip_rgb.shape[0], strip_rgb.shape[1], 1),
                dtype=strip_rgb.dtype,
                device=strip_rgb.device,
            )
            return torch.cat([strip_rgb, alpha], dim=2)
        raise ValueError("TextIncluder supports IMAGE tensors with 3 or 4 channels.")

    def append_text(
        self,
        image: torch.Tensor,
        text: str,
        font_name: str,
        font_size: int,
        padding: int,
        line_spacing: int,
    ) -> Tuple[torch.Tensor]:
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor shape [B, H, W, C].")

        batch, _, width, channels = image.shape
        if batch < 1:
            raise ValueError("Input IMAGE batch is empty.")

        strip_np = self._render_text_strip(
            width=width,
            text=text,
            font_name=font_name,
            font_size=font_size,
            padding=padding,
            line_spacing=line_spacing,
        )
        strip_rgb = torch.from_numpy(strip_np).to(device=image.device, dtype=image.dtype)
        strip = self._adapt_strip_channels(strip_rgb, channels)
        strip = strip.unsqueeze(0).repeat(batch, 1, 1, 1)

        merged = torch.cat([image.clamp(0.0, 1.0), strip], dim=1)
        return (merged,)

#Square ratio images for datasets
class CP_ImageAutoAlignmentSquare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "align_image_to_square"
    CATEGORY = categoryFolder

    def align_image_to_square(self, image):
        batch_size, height, width, channels = image.shape
        target_side = max(height, width)

        if height == width:
            return (image,)

        pad_top = (target_side - height) // 2
        pad_bottom = target_side - height - pad_top
        pad_left = (target_side - width) // 2
        pad_right = target_side - width - pad_left

        aligned = torch.zeros(
            (batch_size, target_side, target_side, channels),
            dtype=image.dtype,
            device=image.device
        )
        if channels == 4:
            aligned[:, :, :, 3] = 1.0
        aligned[:, pad_top:target_side - pad_bottom, pad_left:target_side - pad_right, :] = image
        return (aligned,)

#Resize image with rgba images
class CP_ResizeImageAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384, "step": 1}),
                "method": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"],),
                "mode": (["stretch", "fit", "fill"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_image"
    CATEGORY = categoryFolder

    def _get_resample(self, method):
        mapping = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        return mapping.get(method, Image.Resampling.BICUBIC)

    def _tensor_to_pil_preserve_channels(self, image_tensor, channels):
        data = torch.clamp(image_tensor, 0.0, 1.0).cpu().numpy()
        pixels = np.clip(np.round(data * 255.0), 0, 255).astype(np.uint8)

        if channels == 4:
            return Image.fromarray(pixels, mode="RGBA")
        if channels == 3:
            return Image.fromarray(pixels, mode="RGB")
        if channels == 1:
            return Image.fromarray(pixels[:, :, 0], mode="L")

        raise ValueError(f"Unsupported channel count: {channels}")

    def _pil_to_tensor_preserve_channels(self, image, channels):
        if channels == 4 and image.mode != "RGBA":
            image = image.convert("RGBA")
        elif channels == 3 and image.mode != "RGB":
            image = image.convert("RGB")
        elif channels == 1 and image.mode != "L":
            image = image.convert("L")

        array = np.array(image).astype(np.float32) / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).unsqueeze(0)

    def _create_canvas(self, channels, width, height):
        if channels == 4:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if channels == 3:
            return Image.new("RGB", (width, height), (0, 0, 0))
        if channels == 1:
            return Image.new("L", (width, height), 0)
        raise ValueError(f"Unsupported channel count: {channels}")

    def resize_image(self, image, width, height, method, mode):
        resample = self._get_resample(method)
        result = []
        channels = image.shape[-1]

        for img in image:
            pil_image = self._tensor_to_pil_preserve_channels(img, channels)
            src_width, src_height = pil_image.size

            if mode == "stretch":
                resized = pil_image.resize((width, height), resample=resample)
                result.append(self._pil_to_tensor_preserve_channels(resized, channels))
                continue

            if mode == "fit":
                scale = min(float(width) / float(src_width), float(height) / float(src_height))
            else:
                scale = max(float(width) / float(src_width), float(height) / float(src_height))

            target_width = max(1, int(round(src_width * scale)))
            target_height = max(1, int(round(src_height * scale)))
            resized_source = pil_image.resize((target_width, target_height), resample=resample)

            if mode == "fit":
                canvas = self._create_canvas(channels, width, height)
                paste_x = (width - target_width) // 2
                paste_y = (height - target_height) // 2
                if channels == 4:
                    canvas.paste(resized_source, (paste_x, paste_y), resized_source)
                else:
                    canvas.paste(resized_source, (paste_x, paste_y))
                resized = canvas
            else:
                crop_x = (target_width - width) // 2
                crop_y = (target_height - height) // 2
                resized = resized_source.crop((crop_x, crop_y, crop_x + width, crop_y + height))

            result.append(self._pil_to_tensor_preserve_channels(resized, channels))

        return (torch.cat(result, dim=0),)

#Ported from ComfyUI v0.24.0 ColorTransfer to the classic custom node API.
class CP_ColorTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_ref": ("IMAGE",),
                "method": (["reinhard_lab", "mkl_lab", "histogram"], {"default": "reinhard_lab"}),
                "source_stats": (["per_frame", "uniform", "target_frame"], {"default": "per_frame"}),
                "target_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "color_transfer"
    CATEGORY = categoryFolder

    @staticmethod
    def _get_runtime(image_target):
        from comfy import model_management
        from comfy import utils as comfy_utils

        device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()
        intermediate_dtype_fn = getattr(model_management, "intermediate_dtype", None)
        if callable(intermediate_dtype_fn):
            intermediate_dtype = intermediate_dtype_fn()
        else:
            intermediate_dtype = image_target.dtype

        return device, intermediate_device, intermediate_dtype, comfy_utils

    @staticmethod
    def _try_kornia_color():
        try:
            import kornia
        except ImportError:
            return None
        return kornia.color

    @staticmethod
    def _rgb_to_lab_fallback(image):
        if image.shape[1] != 3:
            raise ValueError("Color Transfer LAB methods require RGB images with 3 channels.")

        rgb = image.clamp(0.0, 1.0)
        linear = torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            torch.pow((rgb + 0.055) / 1.055, 2.4),
        )

        r = linear[:, 0:1, :, :]
        g = linear[:, 1:2, :, :]
        b = linear[:, 2:3, :, :]
        x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

        delta = 6.0 / 29.0
        threshold = delta ** 3

        def pivot(t):
            return torch.where(
                t > threshold,
                torch.pow(t.clamp_min(0.0), 1.0 / 3.0),
                t / (3.0 * delta * delta) + 4.0 / 29.0,
            )

        fx = pivot(x)
        fy = pivot(y)
        fz = pivot(z)
        l = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        lab_b = 200.0 * (fy - fz)
        return torch.cat((l, a, lab_b), dim=1)

    @staticmethod
    def _lab_to_rgb_fallback(image):
        if image.shape[1] != 3:
            raise ValueError("Color Transfer LAB methods require Lab tensors with 3 channels.")

        l = image[:, 0:1, :, :]
        a = image[:, 1:2, :, :]
        lab_b = image[:, 2:3, :, :]
        fy = (l + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - lab_b / 200.0

        delta = 6.0 / 29.0

        def pivot_inv(t):
            return torch.where(
                t > delta,
                torch.pow(t, 3.0),
                3.0 * delta * delta * (t - 4.0 / 29.0),
            )

        x = pivot_inv(fx) * 0.95047
        y = pivot_inv(fy)
        z = pivot_inv(fz) * 1.08883

        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        linear = torch.cat((r, g, b), dim=1).clamp(0.0, 1.0)

        rgb = torch.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055,
        )
        return rgb.clamp(0.0, 1.0)

    @classmethod
    def _rgb_to_lab(cls, image):
        kornia_color = cls._try_kornia_color()
        if kornia_color is not None:
            return kornia_color.rgb_to_lab(image)
        return cls._rgb_to_lab_fallback(image)

    @classmethod
    def _lab_to_rgb(cls, image):
        kornia_color = cls._try_kornia_color()
        if kornia_color is not None:
            return kornia_color.lab_to_rgb(image)
        return cls._lab_to_rgb_fallback(image)

    @classmethod
    def _to_lab(cls, images, i, device):
        return cls._rgb_to_lab(
            images[i:i + 1].to(device, dtype=torch.float32).permute(0, 3, 1, 2)
        )

    @classmethod
    def _pool_stats(cls, images, device, is_reinhard, eps):
        n, channels = images.shape[0], images.shape[3]
        hw = images.shape[1] * images.shape[2]
        mean = torch.zeros(channels, 1, device=device, dtype=torch.float32)
        for i in range(n):
            mean += cls._to_lab(images, i, device).view(channels, -1).mean(dim=-1, keepdim=True)
        mean /= n

        acc = torch.zeros(channels, 1 if is_reinhard else channels, device=device, dtype=torch.float32)
        for i in range(n):
            centered = cls._to_lab(images, i, device).view(channels, -1) - mean
            if is_reinhard:
                acc += (centered * centered).mean(dim=-1, keepdim=True)
            else:
                acc += centered @ centered.T / hw

        if is_reinhard:
            return mean, torch.sqrt(acc / n).clamp_min_(eps)
        return mean, acc / n

    @staticmethod
    def _frame_stats(lab_flat, hw, is_reinhard, eps):
        mean = lab_flat.mean(dim=-1, keepdim=True)
        if is_reinhard:
            return mean, lab_flat.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(eps)
        centered = lab_flat - mean
        return mean, centered @ centered.T / hw

    @staticmethod
    def _mkl_matrix(cov_s, cov_r, eps):
        eig_val_s, eig_vec_s = torch.linalg.eigh(cov_s)
        sqrt_val_s = torch.sqrt(eig_val_s.clamp_min(0)).clamp_min_(eps)

        scaled_v = eig_vec_s * sqrt_val_s.unsqueeze(0)
        mid = scaled_v.T @ cov_r @ scaled_v
        eig_val_m, eig_vec_m = torch.linalg.eigh(mid)
        sqrt_m = torch.sqrt(eig_val_m.clamp_min(0))

        inv_sqrt_s = 1.0 / sqrt_val_s
        inv_scaled_v = eig_vec_s * inv_sqrt_s.unsqueeze(0)
        m_half = (eig_vec_m * sqrt_m.unsqueeze(0)) @ eig_vec_m.T
        return inv_scaled_v @ m_half @ inv_scaled_v.T

    @staticmethod
    def _histogram_lut(src, ref, bins=256):
        s_bins = (src * (bins - 1)).long().clamp(0, bins - 1)
        r_bins = (ref * (bins - 1)).long().clamp(0, bins - 1)
        s_hist = torch.zeros(src.shape[0], bins, device=src.device, dtype=src.dtype)
        r_hist = torch.zeros(src.shape[0], bins, device=src.device, dtype=src.dtype)
        s_hist.scatter_add_(1, s_bins, torch.ones_like(src))
        r_hist.scatter_add_(1, r_bins, torch.ones_like(ref))

        s_cdf = s_hist.cumsum(1)
        s_cdf = s_cdf / s_cdf[:, -1:]
        r_cdf = r_hist.cumsum(1)
        r_cdf = r_cdf / r_cdf[:, -1:]
        return torch.searchsorted(r_cdf, s_cdf).clamp_max_(bins - 1).float() / (bins - 1)

    @classmethod
    def _pooled_cdf(cls, images, device, num_bins=256):
        channels = images.shape[3]
        hist = torch.zeros(channels, num_bins, device=device, dtype=torch.float32)
        for i in range(images.shape[0]):
            frame = images[i].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(channels, -1)
            bins = (frame * (num_bins - 1)).long().clamp(0, num_bins - 1)
            hist.scatter_add_(1, bins, torch.ones_like(frame))
        cdf = hist.cumsum(1)
        return cdf / cdf[:, -1:]

    @classmethod
    def _build_histogram_transform(cls, image_target, image_ref, device, stats_mode, target_index, batch_size):
        if stats_mode == "per_frame":
            return None

        r_cdf = cls._pooled_cdf(image_ref, device)
        if stats_mode == "target_frame":
            ti = min(target_index, batch_size - 1)
            s_cdf = cls._pooled_cdf(image_target[ti:ti + 1], device)
        else:
            s_cdf = cls._pooled_cdf(image_target, device)
        return torch.searchsorted(r_cdf, s_cdf).clamp_max_(255).float() / 255.0

    @classmethod
    def _build_lab_transform(cls, image_target, image_ref, device, stats_mode, target_index, is_reinhard):
        eps = 1e-6
        batch_size, height, width, channels = image_target.shape
        batch_ref = image_ref.shape[0]
        single_ref = batch_ref == 1
        hw = height * width
        hw_ref = image_ref.shape[1] * image_ref.shape[2]

        if single_ref or stats_mode in ("uniform", "target_frame"):
            ref_mean, ref_sc = cls._pool_stats(image_ref, device, is_reinhard, eps)

        if stats_mode in ("uniform", "target_frame"):
            if stats_mode == "target_frame":
                ti = min(target_index, batch_size - 1)
                s_lab = cls._to_lab(image_target, ti, device).view(channels, -1)
                s_mean, s_sc = cls._frame_stats(s_lab, hw, is_reinhard, eps)
            else:
                s_mean, s_sc = cls._pool_stats(image_target, device, is_reinhard, eps)

            if is_reinhard:
                scale = ref_sc / s_sc
                offset = ref_mean - scale * s_mean
                return lambda src_flat, **_: src_flat * scale + offset

            transform_matrix = cls._mkl_matrix(s_sc, ref_sc, eps)
            offset = ref_mean - transform_matrix @ s_mean
            return lambda src_flat, **_: transform_matrix @ src_flat + offset

        def per_frame_transform(src_flat, frame_idx):
            s_mean, s_sc = cls._frame_stats(src_flat, hw, is_reinhard, eps)

            if single_ref:
                r_mean, r_sc = ref_mean, ref_sc
            else:
                ri = min(frame_idx, batch_ref - 1)
                ref_lab = cls._to_lab(image_ref, ri, device).view(channels, -1)
                r_mean, r_sc = cls._frame_stats(ref_lab, hw_ref, is_reinhard, eps)

            centered = src_flat - s_mean
            if is_reinhard:
                return centered * (r_sc / s_sc) + r_mean
            transform_matrix = cls._mkl_matrix(centered @ centered.T / hw, r_sc, eps)
            return transform_matrix @ centered + r_mean

        return per_frame_transform

    @staticmethod
    def _make_progress_bar(comfy_utils, total):
        try:
            return comfy_utils.ProgressBar(total)
        except Exception:
            return None

    @staticmethod
    def _update_progress_bar(progress_bar):
        if progress_bar is not None:
            progress_bar.update(1)

    @classmethod
    def color_transfer(cls, image_target, image_ref, method, source_stats, target_index, strength=1.0):
        if strength == 0 or image_ref is None:
            return (image_target,)

        device, intermediate_device, intermediate_dtype, comfy_utils = cls._get_runtime(image_target)
        batch_size, height, width, channels = image_target.shape
        batch_ref = image_ref.shape[0]
        progress_bar = cls._make_progress_bar(comfy_utils, batch_size)
        out = torch.empty(batch_size, height, width, channels, device=intermediate_device, dtype=intermediate_dtype)

        if method == "histogram":
            uniform_lut = cls._build_histogram_transform(
                image_target, image_ref, device, source_stats, target_index, batch_size
            )

            for i in range(batch_size):
                src = image_target[i].to(device, dtype=torch.float32).permute(2, 0, 1)
                src_flat = src.reshape(channels, -1)
                if uniform_lut is not None:
                    lut = uniform_lut
                else:
                    ri = min(i, batch_ref - 1)
                    ref = image_ref[ri].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(channels, -1)
                    lut = cls._histogram_lut(src_flat, ref)

                bin_idx = (src_flat * 255).long().clamp(0, 255)
                matched = lut.gather(1, bin_idx).view(channels, height, width)
                result = matched if strength == 1.0 else torch.lerp(src, matched, strength)
                out[i] = result.permute(1, 2, 0).clamp_(0, 1).to(
                    device=intermediate_device,
                    dtype=intermediate_dtype,
                )
                cls._update_progress_bar(progress_bar)
        else:
            transform = cls._build_lab_transform(
                image_target,
                image_ref,
                device,
                source_stats,
                target_index,
                is_reinhard=method == "reinhard_lab",
            )
            for i in range(batch_size):
                src_frame = cls._to_lab(image_target, i, device)
                corrected = transform(src_frame.view(channels, -1), frame_idx=i)
                if strength == 1.0:
                    result = cls._lab_to_rgb(corrected.view(1, channels, height, width))
                else:
                    corrected_frame = corrected.view(1, channels, height, width)
                    result = cls._lab_to_rgb(torch.lerp(src_frame, corrected_frame, strength))

                out[i] = result.squeeze(0).permute(1, 2, 0).clamp_(0, 1).to(
                    device=intermediate_device,
                    dtype=intermediate_dtype,
                )
                cls._update_progress_bar(progress_bar)

        return (out,)

#Saves PNG images with Civitai-readable metadata and ComfyUI workflow chunks.
class CP_CivitaiImageSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "path": ("STRING", {"default": "output", "multiline": False}),
                "filename": ("STRING", {"default": "%date:yyyy-MM-dd-hh-mm-ss%", "multiline": False}),
                "quality": (["lossless", "100", "95", "90", "85", "80", "75", "70", "60", "50"], {"default": "lossless"}),
                "save_with_alpha": ("BOOLEAN", {"default": True}),
                "enable_inner_counter": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "positive": (any_type, {"forceInput": True}),
                "negative": (any_type, {"forceInput": True}),
                "steps": (any_type, {"forceInput": True}),
                "sampler": (any_type, {"forceInput": True}),
                "scheduler": (any_type, {"forceInput": True}),
                "cfg": (any_type, {"forceInput": True}),
                "seed": (any_type, {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    SAMPLER_NAMES = {
        "euler": "Euler",
        "euler_ancestral": "Euler a",
        "heun": "Heun",
        "dpm_2": "DPM2",
        "dpm_2_ancestral": "DPM2 a",
        "lms": "LMS",
        "dpm_fast": "DPM fast",
        "dpm_adaptive": "DPM adaptive",
        "ddim": "DDIM",
        "uni_pc": "UniPC",
        "lcm": "LCM",
        "dpmpp_2s_ancestral": "DPM++ 2S a",
        "dpmpp_sde": "DPM++ SDE",
        "dpmpp_2m": "DPM++ 2M",
        "dpmpp_2m_sde": "DPM++ 2M SDE",
        "dpmpp_3m_sde": "DPM++ 3M SDE",
    }

    SCHEDULER_NAMES = {
        "normal": "",
        "simple": "Simple",
        "karras": "Karras",
        "exponential": "Exponential",
        "sgm_uniform": "SGM Uniform",
        "ddim_uniform": "DDIM Uniform",
        "beta": "Beta",
    }

    @staticmethod
    def _metadata_enabled():
        return not getattr(comfy_args, "disable_metadata", False)

    @staticmethod
    def _json_dumps(value):
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _has_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        return True

    @classmethod
    def _to_string(cls, value):
        if value is None or isinstance(value, torch.Tensor):
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                return None
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            for key in ("text", "prompt", "value", "name", "model", "hash"):
                if key in value:
                    return cls._to_string(value[key])
            return None
        if isinstance(value, (list, tuple)):
            parts = [cls._to_string(item) for item in value]
            parts = [part for part in parts if part]
            return ", ".join(parts) if parts and len(parts) <= 8 else None
        return str(value)

    @staticmethod
    def _nodes_from_prompt(prompt):
        if not isinstance(prompt, dict):
            return {}
        return {
            str(node_id): node
            for node_id, node in prompt.items()
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict)
        }

    @staticmethod
    def _widget_value(widgets, index):
        if not isinstance(widgets, list):
            return None
        try:
            index = int(index)
        except Exception:
            return None
        if index < 0 or index >= len(widgets):
            return None
        return widgets[index]

    @classmethod
    def _nodes_from_workflow(cls, workflow):
        if not isinstance(workflow, dict):
            return {}

        workflow_nodes = workflow.get("nodes")
        if not isinstance(workflow_nodes, list):
            return {}

        link_map = {}
        for link in workflow.get("links", []) or []:
            if isinstance(link, (list, tuple)) and len(link) >= 5:
                link_map[str(link[0])] = [str(link[1]), int(link[2])]
            elif isinstance(link, dict):
                link_id = link.get("id")
                source_id = link.get("origin_id", link.get("source_id"))
                source_slot = link.get("origin_slot", link.get("source_slot", 0))
                if link_id is not None and source_id is not None:
                    link_map[str(link_id)] = [str(source_id), int(source_slot)]

        extra = workflow.get("extra") or {}
        widget_idx_map = extra.get("widget_idx_map") if isinstance(extra, dict) else {}
        widget_idx_map = widget_idx_map if isinstance(widget_idx_map, dict) else {}

        nodes = {}
        for node in workflow_nodes:
            if not isinstance(node, dict) or "id" not in node:
                continue
            node_id = str(node["id"])
            inputs = {}

            for input_data in node.get("inputs", []) or []:
                if not isinstance(input_data, dict):
                    continue
                name = input_data.get("name")
                if not name:
                    continue
                link_id = input_data.get("link")
                if link_id is not None and str(link_id) in link_map:
                    inputs[name] = link_map[str(link_id)]

            widgets = node.get("widgets_values")
            node_widget_map = widget_idx_map.get(node_id)
            if node_widget_map is None:
                node_widget_map = widget_idx_map.get(int(node_id)) if node_id.isdigit() else None
            if isinstance(node_widget_map, dict):
                for name, index in node_widget_map.items():
                    if name not in inputs:
                        value = cls._widget_value(widgets, index)
                        if value is not None:
                            inputs[name] = value

            for name, index in cls._known_widget_indexes(node.get("type", "")).items():
                if name not in inputs:
                    value = cls._widget_value(widgets, index)
                    if value is not None:
                        inputs[name] = value

            nodes[node_id] = {
                "inputs": inputs,
                "class_type": node.get("type", ""),
                "_meta": {"title": node.get("title", "")},
            }

        return nodes

    @staticmethod
    def _known_widget_indexes(class_type):
        class_type = str(class_type).lower()
        if "checkpointloader" in class_type:
            return {"ckpt_name": 0}
        if "unetloader" in class_type or "load diffusion model" in class_type:
            return {"unet_name": 0}
        if "loraloader" in class_type or "lora loader" in class_type:
            return {"lora_name": 0, "strength_model": 1, "strength_clip": 2}
        if "ksamplerselect" in class_type:
            return {"sampler_name": 0}
        if "ksampleradvanced" in class_type:
            return {"seed": 1, "steps": 2, "cfg": 3, "sampler_name": 4, "scheduler": 5}
        if class_type == "ksampler" or "ksampler" in class_type:
            return {"seed": 0, "steps": 2, "cfg": 3, "sampler_name": 4, "scheduler": 5, "denoise": 6}
        if "basicscheduler" in class_type:
            return {"scheduler": 0, "steps": 1, "denoise": 2}
        if "cliptextencode" in class_type:
            return {"text": 0}
        return {}

    @staticmethod
    def _is_link(value, nodes):
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and str(value[0]) in nodes
            and isinstance(value[1], int)
        )

    @classmethod
    def _linked_inputs(cls, inputs, nodes):
        for value in inputs.values():
            if cls._is_link(value, nodes):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if cls._is_link(item, nodes):
                        yield item

    @classmethod
    def _resolve_linked_value(cls, value, nodes, role=None, depth=0):
        if depth > 30:
            return None
        if not cls._is_link(value, nodes):
            return value

        node = nodes.get(str(value[0]), {})
        inputs = node.get("inputs", {})
        output_slot = value[1]
        keys = []
        if role:
            keys.append(role)
        if output_slot == 0:
            keys.extend(("positive", "text", "value", "Number", "number", "float", "int"))
        elif output_slot == 1:
            keys.extend(("negative", "text", "value", "Number", "number", "float", "int"))
        keys.extend((
            "text", "value", "Number", "number", "float", "int", "seed", "noise_seed",
            "steps", "cfg", "sampler_name", "scheduler", "ckpt_name", "unet_name",
            "diffusion_model_name", "diffusion_model", "lora_name",
            "string", "positive", "negative", "any_01", "conditioning_1",
            "conditioning_optional_1",
        ))

        for key in keys:
            if key in inputs:
                resolved = cls._resolve_linked_value(inputs[key], nodes, role=role, depth=depth + 1)
                if cls._has_value(resolved):
                    return resolved

        for linked in cls._linked_inputs(inputs, nodes):
            resolved = cls._resolve_linked_value(linked, nodes, role=role, depth=depth + 1)
            if cls._has_value(resolved):
                return resolved
        return None

    @classmethod
    def _collect_ancestors(cls, start_link, nodes):
        ancestors = set()
        distances = {}
        queue = []

        if cls._is_link(start_link, nodes):
            queue.append((str(start_link[0]), 0))
        else:
            queue.extend((node_id, 0) for node_id in nodes)

        while queue:
            node_id, distance = queue.pop(0)
            if node_id in ancestors:
                continue
            ancestors.add(node_id)
            distances[node_id] = distance
            for linked in cls._linked_inputs(nodes.get(node_id, {}).get("inputs", {}), nodes):
                queue.append((str(linked[0]), distance + 1))

        return ancestors, distances

    @staticmethod
    def _node_title(node):
        return str(node.get("_meta", {}).get("title", "")).lower()

    @staticmethod
    def _class_type(node):
        return str(node.get("class_type", "")).lower()

    @classmethod
    def _pick_sampler_node(cls, nodes, ancestors, distances):
        candidates = []
        for node_id in ancestors:
            node = nodes.get(node_id, {})
            inputs = node.get("inputs", {})
            class_type = cls._class_type(node)
            score = 0
            if "sampler" in class_type or "ksampler" in class_type:
                score += 4
            if "positive" in inputs and "negative" in inputs:
                score += 4
            for key in ("steps", "cfg", "seed", "noise_seed", "sampler_name", "sampler", "sigmas"):
                if key in inputs:
                    score += 1
            if score:
                candidates.append((score, -distances.get(node_id, 999999), node_id))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return nodes.get(candidates[0][2])

    @classmethod
    def _search_scalar_in_link(cls, link, nodes, keys):
        if not cls._is_link(link, nodes):
            return None
        visited = set()
        queue = [str(link[0])]
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            inputs = nodes.get(node_id, {}).get("inputs", {})
            for key in keys:
                if key in inputs:
                    value = cls._resolve_linked_value(inputs[key], nodes)
                    if cls._has_value(value):
                        return value
            for linked in cls._linked_inputs(inputs, nodes):
                queue.append(str(linked[0]))
        return None

    @classmethod
    def _extract_text_from_link(cls, link, nodes, role):
        if not cls._is_link(link, nodes):
            return None
        visited = set()
        queue = [str(link[0])]
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = nodes.get(node_id, {})
            inputs = node.get("inputs", {})
            title = cls._node_title(node)
            class_type = cls._class_type(node)

            if role in inputs:
                text = cls._to_string(cls._resolve_linked_value(inputs[role], nodes, role=role))
                if text:
                    return text
            if "cliptextencode" in class_type and "text" in inputs:
                text = cls._to_string(cls._resolve_linked_value(inputs["text"], nodes, role=role))
                if text:
                    return text
            if role in title and "text" in inputs:
                text = cls._to_string(cls._resolve_linked_value(inputs["text"], nodes, role=role))
                if text:
                    return text

            for linked in cls._linked_inputs(inputs, nodes):
                queue.append(str(linked[0]))
        return None

    @classmethod
    def _search_text_by_title(cls, nodes, ancestors, role):
        for node_id in ancestors:
            node = nodes.get(node_id, {})
            inputs = node.get("inputs", {})
            title = cls._node_title(node)
            class_type = cls._class_type(node)
            if role in title and ("text" in inputs or role in inputs):
                key = "text" if "text" in inputs else role
                text = cls._to_string(cls._resolve_linked_value(inputs[key], nodes, role=role))
                if text:
                    return text
            if "cliptextencode" in class_type and role in title and "text" in inputs:
                text = cls._to_string(cls._resolve_linked_value(inputs["text"], nodes, role=role))
                if text:
                    return text
        return None

    @staticmethod
    def _is_base_model_loader_node(node):
        class_type = CP_CivitaiImageSave._class_type(node)
        title = CP_CivitaiImageSave._node_title(node)
        text = f"{class_type} {title}"
        return (
            "checkpoint" in text
            or "unetloader" in text
            or "unet loader" in text
            or "diffusion model" in text
            or "load diffusion" in text
        )

    @classmethod
    def _model_from_node(cls, node, nodes):
        inputs = node.get("inputs", {})
        for key in ("ckpt_name", "unet_name", "diffusion_model_name", "diffusion_model"):
            if key in inputs:
                model_name = cls._to_string(cls._resolve_linked_value(inputs[key], nodes))
                if model_name:
                    return model_name

        if "model_name" in inputs and cls._is_base_model_loader_node(node):
            model_name = cls._to_string(cls._resolve_linked_value(inputs["model_name"], nodes))
            if model_name:
                return model_name

        return None

    @classmethod
    def _extract_model_from_link(cls, link, nodes):
        if not cls._is_link(link, nodes):
            return None
        visited = set()
        queue = [str(link[0])]
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = nodes.get(node_id, {})
            model_name = cls._model_from_node(node, nodes)
            if model_name:
                return model_name
            inputs = node.get("inputs", {})
            for linked in cls._linked_inputs(inputs, nodes):
                queue.append(str(linked[0]))
        return None

    @classmethod
    def _lora_from_node(cls, node, nodes):
        inputs = node.get("inputs", {})
        class_type = cls._class_type(node)
        if "lora_name" not in inputs and not ("lora" in class_type and "name" in inputs):
            return None

        lora_name = cls._to_string(cls._resolve_linked_value(inputs.get("lora_name", inputs.get("name")), nodes))
        if not lora_name:
            return None

        return {
            "name": os.path.basename(lora_name),
            "path": lora_name,
            "strength_model": cls._to_string(cls._resolve_linked_value(inputs.get("strength_model"), nodes)),
            "strength_clip": cls._to_string(cls._resolve_linked_value(inputs.get("strength_clip"), nodes)),
        }

    @staticmethod
    def _dedupe_loras(loras):
        result = []
        seen = set()
        for lora in loras:
            if not isinstance(lora, dict):
                continue
            key = (
                str(lora.get("path") or lora.get("name") or ""),
                str(lora.get("strength_model") or ""),
                str(lora.get("strength_clip") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(lora)
        return result

    @classmethod
    def _extract_loras_from_nodes(cls, nodes, node_ids=None):
        loras = []
        ids = list(node_ids) if node_ids is not None else list(nodes.keys())
        for node_id in ids:
            node = nodes.get(str(node_id), {})
            lora = cls._lora_from_node(node, nodes)
            if lora:
                loras.append(lora)
        return cls._dedupe_loras(loras)

    @classmethod
    def _extract_loras_from_link(cls, link, nodes):
        if not cls._is_link(link, nodes):
            return []
        loras = []
        visited = set()

        def walk(node_id):
            node_id = str(node_id)
            if node_id in visited:
                return
            visited.add(node_id)
            node = nodes.get(node_id, {})
            inputs = node.get("inputs", {})
            lora = cls._lora_from_node(node, nodes)
            if lora:
                loras.append(lora)
            model_link = inputs.get("model")
            if cls._is_link(model_link, nodes):
                walk(model_link[0])
                return
            for linked in cls._linked_inputs(inputs, nodes):
                walk(linked[0])

        walk(link[0])
        return cls._dedupe_loras(loras)

    @classmethod
    def _extract_metadata_from_nodes(cls, nodes, unique_id):
        if not nodes:
            return {}

        save_node = nodes.get(str(unique_id)) if unique_id is not None else None
        save_inputs = save_node.get("inputs", {}) if isinstance(save_node, dict) else {}
        image_link = save_inputs.get("image", save_inputs.get("images"))
        ancestors, distances = cls._collect_ancestors(image_link, nodes)
        sampler_node = cls._pick_sampler_node(nodes, ancestors, distances)
        data = {"loras": []}

        if sampler_node is not None:
            inputs = sampler_node.get("inputs", {})
            data["positive"] = cls._extract_text_from_link(inputs.get("positive"), nodes, "positive")
            data["negative"] = cls._extract_text_from_link(inputs.get("negative"), nodes, "negative")
            data["steps"] = cls._resolve_linked_value(inputs.get("steps"), nodes)
            if not cls._has_value(data["steps"]):
                data["steps"] = cls._search_scalar_in_link(inputs.get("sigmas"), nodes, ("steps",))
            data["sampler_name"] = cls._resolve_linked_value(inputs.get("sampler_name"), nodes)
            if not cls._has_value(data["sampler_name"]):
                data["sampler_name"] = cls._search_scalar_in_link(inputs.get("sampler"), nodes, ("sampler_name", "sampler"))
            data["scheduler"] = cls._resolve_linked_value(inputs.get("scheduler"), nodes)
            if not cls._has_value(data["scheduler"]):
                data["scheduler"] = cls._search_scalar_in_link(inputs.get("sigmas"), nodes, ("scheduler",))
            data["cfg"] = cls._resolve_linked_value(inputs.get("cfg"), nodes)
            data["seed"] = cls._resolve_linked_value(inputs.get("seed", inputs.get("noise_seed")), nodes)
            data["model"] = cls._extract_model_from_link(inputs.get("model"), nodes)
            data["loras"] = cls._extract_loras_from_link(inputs.get("model"), nodes)

        if not data.get("positive"):
            data["positive"] = cls._search_text_by_title(nodes, ancestors, "positive")
        if not data.get("negative"):
            data["negative"] = cls._search_text_by_title(nodes, ancestors, "negative")
        if not data.get("model"):
            for node_id in ancestors:
                node = nodes.get(node_id, {})
                model_name = cls._model_from_node(node, nodes)
                if model_name:
                    data["model"] = model_name
                    break
        if not data.get("model"):
            for node in nodes.values():
                model_name = cls._model_from_node(node, nodes)
                if model_name:
                    data["model"] = model_name
                    break
        if not data.get("loras"):
            data["loras"] = cls._extract_loras_from_nodes(nodes, ancestors) or cls._extract_loras_from_nodes(nodes)
        return data

    @classmethod
    def _merge_graph_extracts(cls, primary, fallback):
        if not primary:
            return fallback or {}
        if not fallback:
            return primary

        merged = dict(primary)
        for key, value in fallback.items():
            if key == "loras":
                if not merged.get("loras"):
                    merged["loras"] = value
                continue
            if not cls._has_value(merged.get(key)) and cls._has_value(value):
                merged[key] = value
        return merged

    @classmethod
    def _extract_graph_metadata(cls, prompt, extra_pnginfo=None, unique_id=None):
        prompt_data = cls._extract_metadata_from_nodes(cls._nodes_from_prompt(prompt), unique_id)

        workflow = None
        if isinstance(extra_pnginfo, dict):
            workflow = extra_pnginfo.get("workflow")
        workflow_data = cls._extract_metadata_from_nodes(cls._nodes_from_workflow(workflow), unique_id)

        return cls._merge_graph_extracts(prompt_data, workflow_data)

    @classmethod
    def _parse_lora_metadata(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                value = json.loads(text)
            except Exception:
                loras = []
                for line in text.splitlines():
                    parts = [part.strip() for part in line.split(",") if part.strip()]
                    if not parts:
                        continue
                    item = {"name": parts[0]}
                    for part in parts[1:]:
                        if "=" in part:
                            key, val = part.split("=", 1)
                            item[key.strip()] = val.strip()
                    loras.append(item)
                return loras
        if isinstance(value, dict):
            if isinstance(value.get("loras"), list):
                return value["loras"]
            return [
                dict(val, name=key) if isinstance(val, dict) and "name" not in val else
                (val if isinstance(val, dict) else {"name": key, "hash": val})
                for key, val in value.items()
            ]
        if isinstance(value, (list, tuple)):
            loras = []
            for item in value:
                if isinstance(item, dict):
                    loras.append(item)
                else:
                    name = cls._to_string(item)
                    if name:
                        loras.append({"name": name})
            return loras
        return []

    @staticmethod
    def _output_directory():
        if folder_paths is not None:
            try:
                return folder_paths.get_output_directory()
            except Exception:
                pass
        return os.path.join(os.getcwd(), "output")

    @classmethod
    def _resolve_save_dir(cls, path):
        output_dir = os.path.abspath(cls._output_directory())
        path = (path or "output").strip()
        if not path or os.path.normcase(path) == os.path.normcase("output"):
            return output_dir
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(output_dir, path))

    @staticmethod
    def _find_full_path(folder_name, filename):
        if not filename:
            return None
        if os.path.isfile(filename):
            return filename
        if folder_paths is not None:
            try:
                full_path = folder_paths.get_full_path(folder_name, filename)
                if full_path and os.path.isfile(full_path):
                    return full_path
            except Exception:
                pass
        return None

    @classmethod
    def _find_model_full_path(cls, filename):
        for folder_name in ("checkpoints", "diffusion_models", "unet"):
            full_path = cls._find_full_path(folder_name, filename)
            if full_path:
                return os.path.abspath(full_path)
        return None

    @staticmethod
    def _sha256_short(path):
        if not path or not os.path.isfile(path):
            return None
        abs_path = os.path.abspath(path)
        if abs_path in _CP_CIVITAI_HASH_CACHE:
            return _CP_CIVITAI_HASH_CACHE[abs_path]
        sha256 = hashlib.sha256()
        with open(abs_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                sha256.update(chunk)
        value = sha256.hexdigest()[:10]
        _CP_CIVITAI_HASH_CACHE[abs_path] = value
        return value

    @staticmethod
    def _checkpoint_db_path():
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_root, "db", "checkpoints.json")

    @staticmethod
    def _file_stat_info(path):
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return {
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1000000000))),
        }

    @classmethod
    def _load_checkpoint_db(cls):
        global _CP_CIVITAI_CHECKPOINT_DB_CACHE
        if _CP_CIVITAI_CHECKPOINT_DB_CACHE is not None:
            return _CP_CIVITAI_CHECKPOINT_DB_CACHE

        db_path = cls._checkpoint_db_path()
        try:
            with open(db_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            data = {}

        if not isinstance(data, dict):
            data = {}

        clean_data = {}
        for key, value in data.items():
            if isinstance(key, str) and os.path.isabs(key) and isinstance(value, dict):
                clean_data[os.path.abspath(key)] = value

        _CP_CIVITAI_CHECKPOINT_DB_CACHE = clean_data
        return _CP_CIVITAI_CHECKPOINT_DB_CACHE

    @classmethod
    def _save_checkpoint_db(cls, data):
        global _CP_CIVITAI_CHECKPOINT_DB_CACHE
        db_path = cls._checkpoint_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        tmp_path = f"{db_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, db_path)
        _CP_CIVITAI_CHECKPOINT_DB_CACHE = data

    @classmethod
    def _checkpoint_hash_from_db(cls, path):
        abs_path = os.path.abspath(path)
        entry = cls._load_checkpoint_db().get(abs_path)
        if not isinstance(entry, dict):
            return None

        hash_value = cls._to_string(entry.get("hash"))
        if not hash_value:
            return None

        current_stat = cls._file_stat_info(abs_path)
        if current_stat is None:
            return hash_value

        if entry.get("size") == current_stat["size"] and entry.get("mtime_ns") == current_stat["mtime_ns"]:
            _CP_CIVITAI_HASH_CACHE[abs_path] = hash_value
            return hash_value

        return None

    @classmethod
    def _write_checkpoint_info(cls, path, hash_value):
        abs_path = os.path.abspath(path)
        stat_info = cls._file_stat_info(abs_path)
        if stat_info is None:
            return

        db = dict(cls._load_checkpoint_db())
        db[abs_path] = {
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "hash": hash_value,
            "size": stat_info["size"],
            "mtime_ns": stat_info["mtime_ns"],
        }
        cls._save_checkpoint_db(db)

    @classmethod
    def _checkpoint_hash_with_cache(cls, path):
        if not path or not os.path.isfile(path):
            return None

        abs_path = os.path.abspath(path)
        hash_value = cls._checkpoint_hash_from_db(abs_path)
        if hash_value:
            return hash_value

        hash_value = cls._sha256_short(abs_path)
        if hash_value:
            cls._write_checkpoint_info(abs_path, hash_value)
        return hash_value

    @classmethod
    def _format_sampler(cls, sampler, scheduler=None):
        sampler_text = cls._to_string(sampler)
        if not sampler_text:
            return None
        sampler_text = sampler_text.strip()
        display = cls.SAMPLER_NAMES.get(sampler_text.lower(), sampler_text)
        scheduler_text = cls._to_string(scheduler)
        scheduler_text = scheduler_text.strip() if scheduler_text else ""
        scheduler_display = cls.SCHEDULER_NAMES.get(scheduler_text.lower(), scheduler_text)
        if scheduler_display and scheduler_display.lower() not in display.lower():
            display = f"{display} {scheduler_display}"
        return display

    @staticmethod
    def _date_format_to_strftime(fmt):
        for source, target in (
            ("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
            ("hh", "%H"), ("HH", "%H"), ("mm", "%M"), ("ss", "%S"),
        ):
            fmt = fmt.replace(source, target)
        return fmt

    @classmethod
    def _replace_filename_tokens(cls, filename, fields, batch_number=None, counter=None):
        now = datetime.now()
        result = re.sub(
            r"%date:([^%]+)%",
            lambda match: now.strftime(cls._date_format_to_strftime(match.group(1))),
            filename,
        )
        for token, value in {
            "year": now.strftime("%Y"),
            "month": now.strftime("%m"),
            "day": now.strftime("%d"),
            "hour": now.strftime("%H"),
            "minute": now.strftime("%M"),
            "second": now.strftime("%S"),
            "seed": fields.get("seed"),
            "steps": fields.get("steps"),
            "sampler": fields.get("sampler"),
            "cfg": fields.get("cfg"),
            "model": os.path.splitext(os.path.basename(fields.get("model") or ""))[0],
            "width": fields.get("width"),
            "height": fields.get("height"),
            "batch_num": batch_number,
            "counter": f"{counter:05}" if isinstance(counter, int) else counter,
        }.items():
            if value is not None:
                result = result.replace(f"%{token}%", str(value))
        return result

    @staticmethod
    def _sanitize_filename(filename):
        filename = (filename or "image").strip()
        filename = filename.replace("\\", "_").replace("/", "_")
        filename = re.sub(r'[<>:"|?*]', "_", filename)
        filename = re.sub(r"\s+", " ", filename).strip()
        return filename or "image"

    @staticmethod
    def _strip_png_extension(filename):
        root, ext = os.path.splitext(filename)
        return root if ext.lower() == ".png" else filename

    @staticmethod
    def _existing_counter(save_dir, base_name):
        pattern = re.compile(re.escape(base_name) + r"_(\d{5})\.png$", re.IGNORECASE)
        max_counter = 0
        if os.path.isdir(save_dir):
            try:
                names = os.listdir(save_dir)
            except OSError:
                names = []
            for name in names:
                match = pattern.match(name)
                if match:
                    max_counter = max(max_counter, int(match.group(1)))
        return max_counter + 1

    @staticmethod
    def _png_compress_level(quality):
        if quality == "lossless":
            return 4
        try:
            quality_value = max(0, min(100, int(quality)))
        except Exception:
            return 4
        return max(0, min(9, round((100 - quality_value) / 100 * 9)))

    @staticmethod
    def _tensor_to_pil_for_save(image_tensor, save_with_alpha):
        image_np = image_tensor.detach().cpu().numpy()
        image_np = np.clip(np.round(image_np * 255.0), 0, 255).astype(np.uint8)
        if image_np.ndim == 2:
            return Image.fromarray(image_np, mode="L")
        channels = image_np.shape[-1]
        if channels == 4:
            pil_image = Image.fromarray(image_np, mode="RGBA")
            return pil_image if save_with_alpha else pil_image.convert("RGB")
        if channels == 3:
            return Image.fromarray(image_np, mode="RGB")
        if channels == 1:
            return Image.fromarray(image_np[:, :, 0], mode="L")
        return Image.fromarray(image_np[:, :, :3], mode="RGB")

    @classmethod
    def _merge_metadata(cls, graph_data, width, height, positive=None, negative=None, steps=None,
                        sampler=None, scheduler=None, cfg=None, seed=None, model=None, model_hash=None,
                        lora_metadata=None):
        fields = dict(graph_data or {})
        fields["width"] = width
        fields["height"] = height
        for key, value in {
            "positive": positive,
            "negative": negative,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "model": model,
            "model_hash": model_hash,
        }.items():
            text = cls._to_string(value)
            if text:
                fields[key] = text

        sampler_override = cls._to_string(sampler)
        scheduler_override = cls._to_string(scheduler)
        if scheduler_override:
            fields["scheduler"] = scheduler_override
        fields["sampler"] = cls._format_sampler(
            sampler_override or fields.get("sampler_name"),
            fields.get("scheduler"),
        )
        fields["steps"] = cls._to_string(fields.get("steps"))
        fields["cfg"] = cls._to_string(fields.get("cfg"))
        fields["seed"] = cls._to_string(fields.get("seed"))
        fields["model"] = cls._to_string(fields.get("model"))

        loras = list(fields.get("loras") or [])
        loras.extend(cls._parse_lora_metadata(lora_metadata))
        fields["loras"] = loras

        if not fields.get("model_hash") and fields.get("model"):
            fields["model_hash"] = cls._checkpoint_hash_with_cache(cls._find_model_full_path(fields["model"]))

        for lora in fields["loras"]:
            if not isinstance(lora, dict):
                continue
            if not lora.get("hash"):
                full_path = cls._find_full_path("loras", lora.get("path") or lora.get("name") or lora.get("model"))
                lora["hash"] = cls._sha256_short(full_path)
            if "strength_model" not in lora and "strength mode" in lora:
                lora["strength_model"] = lora.get("strength mode")
            if "strength_clip" not in lora and "strength clip" in lora:
                lora["strength_clip"] = lora.get("strength clip")
        return fields

    @classmethod
    def _build_parameters(cls, fields):
        positive = cls._to_string(fields.get("positive")) or ""
        negative = cls._to_string(fields.get("negative")) or ""
        details = []
        for key, label in (("steps", "Steps"), ("sampler", "Sampler"), ("cfg", "CFG scale"), ("seed", "Seed")):
            value = cls._to_string(fields.get(key))
            if value:
                details.append(f"{label}: {value}")
        if fields.get("width") and fields.get("height"):
            details.append(f"Size: {fields['width']}x{fields['height']}")
        if fields.get("model"):
            details.append(f"Model: {fields['model']}")
        if fields.get("model_hash"):
            details.append(f"Model hash: {fields['model_hash']}")

        hashes = {}
        if fields.get("model_hash"):
            hashes["model"] = fields["model_hash"]

        for index, lora in enumerate(fields.get("loras") or []):
            if not isinstance(lora, dict):
                continue
            name = cls._to_string(lora.get("name") or lora.get("model") or lora.get("path"))
            if not name:
                continue
            name = os.path.basename(name)
            details.append(f"Lora_{index} Model name: {name}")
            lora_hash = cls._to_string(lora.get("hash") or lora.get("model_hash"))
            if lora_hash:
                details.append(f"Lora_{index} Model hash: {lora_hash}")
                hashes[f"lora:{os.path.splitext(name)[0]}"] = lora_hash
            strength_model = cls._to_string(lora.get("strength_model") or lora.get("strength"))
            if strength_model:
                details.append(f"Lora_{index} Strength model: {strength_model}")
            strength_clip = cls._to_string(lora.get("strength_clip"))
            if strength_clip:
                details.append(f"Lora_{index} Strength clip: {strength_clip}")

        if hashes:
            details.append(f"Hashes: {cls._json_dumps(hashes)}")

        parts = [positive]
        if negative:
            parts.append(f"Negative prompt: {negative}")
        if details:
            parts.append(", ".join(details))
        return "\n".join(parts)

    @classmethod
    def _make_pnginfo(cls, parameters, prompt=None, extra_pnginfo=None):
        if not cls._metadata_enabled():
            return None
        metadata = PngInfo()
        metadata.add_text("parameters", parameters)
        if prompt is not None:
            metadata.add_text("prompt", cls._json_dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(str(key), cls._json_dumps(value))
        return metadata

    @classmethod
    def _relative_subfolder(cls, save_dir):
        output_dir = os.path.abspath(cls._output_directory())
        save_dir = os.path.abspath(save_dir)
        try:
            if os.path.commonpath((output_dir, save_dir)) == output_dir:
                rel = os.path.relpath(save_dir, output_dir)
                return "" if rel == "." else rel
        except Exception:
            pass
        return ""

    def save_images(self, image, path, filename, quality, save_with_alpha, enable_inner_counter,
                    positive=None, negative=None, steps=None, sampler=None, scheduler=None, cfg=None, seed=None,
                    model=None, model_hash=None, lora_metadata=None, prompt=None,
                    extra_pnginfo=None, unique_id=None):
        if image is None or len(image) == 0:
            raise ValueError("CP CivitaiImageSave: image input is empty.")

        save_dir = self._resolve_save_dir(path)
        os.makedirs(save_dir, exist_ok=True)
        height = int(image[0].shape[0])
        width = int(image[0].shape[1])

        fields = self._merge_metadata(
            self._extract_graph_metadata(prompt, extra_pnginfo, unique_id),
            width,
            height,
            positive=positive,
            negative=negative,
            steps=steps,
            sampler=sampler,
            scheduler=scheduler,
            cfg=cfg,
            seed=seed,
            model=model,
            model_hash=model_hash,
            lora_metadata=lora_metadata,
        )

        base_template = self._strip_png_extension(filename or "%date:yyyy-MM-dd-hh-mm-ss%")
        counter_start = None
        if enable_inner_counter or "%counter%" in base_template:
            preview_name = self._sanitize_filename(self._replace_filename_tokens(base_template, fields, batch_number=0, counter=None))
            counter_start = self._existing_counter(save_dir, preview_name)

        parameters = self._build_parameters(fields)
        pnginfo = self._make_pnginfo(parameters, prompt=prompt, extra_pnginfo=extra_pnginfo)
        compress_level = self._png_compress_level(quality)
        subfolder = self._relative_subfolder(save_dir)
        results = []

        for batch_number, image_tensor in enumerate(image):
            counter = counter_start + batch_number if counter_start is not None else None
            base_name = self._replace_filename_tokens(base_template, fields, batch_number=batch_number, counter=counter)
            base_name = self._sanitize_filename(base_name)
            if enable_inner_counter:
                file_name = f"{base_name}_{counter:05}.png"
            elif len(image) > 1 and "%batch_num%" not in base_template:
                file_name = f"{base_name}_{batch_number:05}.png"
            else:
                file_name = f"{base_name}.png"

            file_path = os.path.join(save_dir, file_name)
            pil_image = self._tensor_to_pil_for_save(image_tensor, save_with_alpha)
            pil_image.save(file_path, pnginfo=pnginfo, compress_level=compress_level)
            results.append({"filename": file_name, "subfolder": subfolder, "type": "output"})

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS_IMAGE = {
    "CP ImageCropper": CP_ImageCrop,
    "CP ImageConcatMask": CP_ImageConcatMask,
    "CP FlipImagePair": CP_FlipImagePair,
    "CP DrawDetections": CP_DrawDetections,
    "CP AddJpegArtifacts": CP_AddJpegArtifacts,
    "CP ImageCrop": CP_ImageCrop,
    "CP ImageRotate": CP_ImageRotate,
    "CP CropImageToME": CP_CropImageToME,
    "CP ImageSaver": CP_ImageSaver,
    "CP ImageGrid": CP_ImageGrid,
    "CP TextIncluder": CP_TextIncluder,
    "CP ImageAutoAlignmentSquare": CP_ImageAutoAlignmentSquare,
    "CP ResizeImageAlpha": CP_ResizeImageAlpha,
    "CP ColorTransfer": CP_ColorTransfer,
    "CP CivitaiImageSave": CP_CivitaiImageSave,
}

NODE_DISPLAY_NAME_MAPPINGS_IMAGE = {
    "CP ImageCropper": "Image crop",
    "CP ImageConcatMask": "Image concat mask",
    "CP FlipImagePair": "Flip image pair",
    "CP DrawDetections": "Draw detections",
    "CP AddJpegArtifacts": "Add jpeg artifacts",
    "CP ImageCrop": "Crop image by sides",
    "CP ImageRotate": "Rotate image",
    "CP CropImageToME": "Image crop to ME",
    "CP ImageSaver": "Image Saver Custom",
    "CP ImageGrid": "Image grid",
    "CP TextIncluder": "Text Includer",
    "CP ImageAutoAlignmentSquare": "Image auto alignment square",
    "CP ResizeImageAlpha": "Resize image alpha",
    "CP ColorTransfer": "Color Transfer",
    "CP CivitaiImageSave": "Civitai image save",
}
