from __future__ import annotations
import os
import folder_paths
from datetime import datetime
import torch
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
import math
import glob
from typing import Any
import requests
import threading

categoryFolder = "Custom pack/Tools"

#Returns the image resolution greater than or equal to the specified one (if checking is enabled) and returns the original if there is no checking.
class CP_AspectRatioCheck:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT",),
                "height": ("INT",),
                "aspect_ratio": ("INT", {"default": 512, "min": 8, "max": 8192, "step": 8}),
                "check": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "calculate"
    CATEGORY = categoryFolder

    def calculate(self, width, height, aspect_ratio, check):
        if check:
            if aspect_ratio > width and aspect_ratio > height:
                output = aspect_ratio
            else:
                output = max(width, height)
        else:
            output = aspect_ratio
        return (output,)

#Returns a float from 0 to 1 with an accuracy of 2 digits. Needed to set precise denoise.
class CP_FloatZeroToOne:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = categoryFolder

    def get_value(self, value):
        return (value,)

#Holds an image in an array (deprecated)
class CP_KeepImage:
    def __init__(self):
        pass
    v = [None, None, None, None, None,
         None, None, None, None, None,
         None, None, None, None, None,
         None, None, None, None, None,
         None, None, None, None, None,
         None, None, None, None, None,
         None, None, None]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default":0, "min": 0, "max": 32, "step": 1}),
            },
            "optional": {
                "img": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "CP_KeepImg"
    CATEGORY = categoryFolder

    def CP_KeepImg(self, index, img=None):
        if isinstance(img, torch.Tensor):
            CP_KeepImage.v[index] = img
        return (CP_KeepImage.v[index],)

#Holds a number (deprecated - used to hold seeds)
class CP_KeepInteger:
    def __init__(self):
        pass
    v = [0] * 33
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default":0, "min": 0, "max": 32, "step": 1}),
            },
            "optional": {
                "int_input": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
            },
        }
    RETURN_TYPES = ("INT",)
    FUNCTION = "CP_KeepInt"
    CATEGORY = categoryFolder

    def CP_KeepInt(self, index, int_input=None):
        if isinstance(int_input, int):
            CP_KeepInteger.v[index] = int_input
        return (CP_KeepInteger.v[index],)

#Allows you to submit lore in one list. Needed for testing lore from different eras.
class CP_LoraList:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True, "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("string",)
    RETURN_NAMES = ("range",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "LoraList"
    CATEGORY = categoryFolder

    def LoraList(self, text):
        r = []
        t = text[0].split("\n")
        for i in t:
            r.append(i)
        return (r,)

#Returns a list of available image resolutions for generation
class CP_ImageResolutionSelector:
    @classmethod
    def INPUT_TYPES(s):
        resolutions = [
            ("416x608"),
            ("512x512"),
            ("512x768"),
            ("608x480"),
            ("640x1536"),
            ("672x512"),
            ("768x512"),
            ("768x768"),
            ("768x1344"),
            ("832x1216"),
            ("896x1152"),
            ("1024x1024"),
            ("1152x896"),
            ("1216x832"),
            ("1280x1280"),
            ("1344x768"),
            ("1536x640"),
        ]
        return {
            "required": {
                "resolution": (resolutions, {"default": "512x512"}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = categoryFolder

    def get_resolution(self, resolution):
        width, height = map(int, resolution.split("x"))
        return (width, height)

#Allows you to submit a single list of seeds. Sometimes useful for testing.
class CP_SeedList:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": True, "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("range",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "SeedList"
    CATEGORY = categoryFolder

    def SeedList(self, text):
        r = []
        t = text[0].split("\n")
        for i in t:
            r.append(int(i))
        return (r,)

#Allows you to specify a string and word dictionaries to add or remove words in a string, separated by commas. Needed for automatic prompting before training the lora
class CP_StringProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True}),
                "concat_string": ("STRING", {"forceInput": True}),
                "input_exclude": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_string",)
    FUNCTION = "process_string"
    CATEGORY = categoryFolder

    def process_string(self, input_string, concat_string, input_exclude):
        input_str = str(input_string) if input_string is not None else ""
        exclude_str = str(input_exclude) if input_exclude is not None else ""
        
        words = [word.strip() for word in input_str.split(",") if word.strip()]
        
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        exclude_terms = []
        for term in exclude_str.split(","):
            term = str(term).strip() if term is not None else ""
            if term:
                exclude_terms.append(term)
        
        filtered_words = []
        for word in unique_words:
            word_str = str(word) if word is not None else ""
            exclude = False
            
            for exclude_term in exclude_terms:
                try:
                    if exclude_term and word_str and exclude_term.lower() in word_str.lower():
                        exclude = True
                        break
                except AttributeError:
                    continue
            
            if not exclude and word_str:
                filtered_words.append(word_str)
        
        processed_words = [word.replace("_", " ") for word in filtered_words]
        
        if concat_string:
            concat_str = str(concat_string) if concat_string is not None else ""
            if concat_str:
                processed_words.insert(0, concat_str)
        
        output_string = ", ".join(processed_words)
        
        return (output_string,)

#Submits a list of words (deprecated - used to pass a list of samplers)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class CP_StringRange:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"string": ("STRING", {"default": "", "multiline": True})},
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("range",)
    OUTPUT_IS_LIST = (True)
    FUNCTION = "CP_StrRange"
    CATEGORY = categoryFolder

    def CP_StrRange(self, string):
        r = []
        if string != "":
            values = string.split('\n')
            r.extend(values)
            print(r)
        return(r,)

#Returns a string of choice. Needed for multi-prompting.
class CP_StringSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "string_1": ("STRING", {"forceInput": True}),
                "string_2": ("STRING", {"forceInput": True}),
                "string_3": ("STRING", {"forceInput": True}),
                "string_4": ("STRING", {"forceInput": True}),
                "string_5": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_string"
    CATEGORY = categoryFolder

    def select_string(self, input_index, **kwargs):
        input_name = f"string_{input_index}"
        selected_string = kwargs.get(input_name, "")
        
        if selected_string is None:
            selected_string = ""
            
        return (selected_string,)

#Selects the upscale factor value by flag
class CP_UpscaleFactorSelect:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flag": ("BOOLEAN", {"default": False}),
                "on_true": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01}),
                "on_false": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("upscaleRatio",)
    FUNCTION = "evaluate"
    CATEGORY = categoryFolder
    
    def evaluate(self, flag, on_true, on_false):
        upscaleRatio = on_true if flag else on_false
        return (upscaleRatio,)

#Selects the value of steps by flag
class CP_StepsSelect:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flag": ("BOOLEAN", {"default": False}),
                "on_true": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "on_false": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)
    FUNCTION = "evaluate"
    CATEGORY = categoryFolder
    
    def evaluate(self, flag, on_true, on_false):
        steps = on_true if flag else on_false
        return (steps,)

#Selects the image size value by flag
class CP_IntPosSelect:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flag": ("BOOLEAN", {"default": False}),
                "on_true": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
                "on_false": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)
    FUNCTION = "evaluate"
    CATEGORY = categoryFolder
    
    def evaluate(self, flag, on_true, on_false):
        steps = on_true if flag else on_false
        return (steps,)

#Displays the second image if the first one is not passed or a black square.
class CP_ImageCheck:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image2": ("IMAGE",),
            },
            "optional": {
                "image1": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "check_images"
    CATEGORY = categoryFolder
    
    def check_images(self, image2, image1=None):
        if image2.shape[0] > 1:
            raise ValueError("Custom pack: Second image is batch > 1")
        
        if image1 is not None and image1.shape[0] > 1:
            raise ValueError("Custom pack: First image is batch > 1")
        
        if image1 is None:
            return (image2,)
        
        img1_tensor = image1[0]
        img1_np = img1_tensor.cpu().numpy()
        is_black = np.all(img1_np == 0)
        
        if is_black:
            return (image2,)
        else:
            return (image1,)

#Generates a short seed for Numpy 1.26 (from 0 to 4294967295)
class CP_SeedGeneratorShort:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff})}}

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_seed"
    CATEGORY = categoryFolder

    def get_seed(self, seed):
        return (seed,)

#Assembles paths for dataset preparation from base, dataset and folder-name segments.
class CP_AssemblyFoldersDatasetPreparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "basePath": ("STRING", {"default": "Z:\\SD\\C0164\\output", "multiline": False}),
                "dataset": ("STRING", {"default": "Dataset-LanayaSFM", "multiline": False}),
                "srcSub": ("STRING", {"default": "SRC", "multiline": False}),
                "masklabel": ("STRING", {"default": "Masklabel", "multiline": False}),
                "transparent": ("STRING", {"default": "Transparent", "multiline": False}),
                "transparentResized": ("STRING", {"default": "TransparentResized", "multiline": False}),
                "augmentation": ("STRING", {"default": "Augmentation", "multiline": False}),
                "caption": ("STRING", {"default": "Caption", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("srcFull", "masklabelFull", "transparentFull", "transparentResizedFull", "aumentationFull", "captionFull", "masklabelSub", "transparentSub", "transparentResizedSub")
    FUNCTION = "assemble"
    CATEGORY = categoryFolder

    def assemble(self, basePath, dataset, srcSub, masklabel, transparent, transparentResized, augmentation, caption):
        slash = "\\"
        base_dataset = f"{basePath}{slash}{dataset}"

        src_with_slash = f"{srcSub}{slash}"
        masklabel_with_slash = f"{masklabel}{slash}"
        transparent_with_slash = f"{transparent}{slash}"
        transparentResized_with_slash = f"{transparentResized}{slash}"
        augmentation_with_slash = f"{augmentation}{slash}"
        caption_with_slash = f"{caption}{slash}"

        srcFull = f"{base_dataset}{slash}{src_with_slash}"
        masklabelFull = f"{base_dataset}{slash}{masklabel_with_slash}"
        transparentFull = f"{base_dataset}{slash}{transparent_with_slash}"
        transparentResizedFull = f"{base_dataset}{slash}{transparentResized_with_slash}"
        aumentationFull = f"{base_dataset}{slash}{augmentation_with_slash}"
        captionFull = f"{base_dataset}{slash}{caption_with_slash}"

        masklabelSub = f"{dataset}{slash}{masklabel_with_slash}"
        transparentSub = f"{dataset}{slash}{transparent_with_slash}"
        transparentResizedSub = f"{dataset}{slash}{transparentResized_with_slash}"

        return (srcFull, masklabelFull, transparentFull, transparentResizedFull, aumentationFull, captionFull, masklabelSub, transparentSub, transparentResizedSub)

#Help node. Lists available width and height resolutions for generation, allowing for free resolution settings.
class CP_CustomResolution:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        size_options = list(range(256, 4097, 32))

        return {
            "required": {
                "width": (size_options, {"default": 512}),
                "height": (size_options, {"default": 512}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "getter"
    CATEGORY = categoryFolder

    def getter(self, width, height):
        return(width, height,)

#Calc resolution by pixel limit
class CP_ResolutionByPixelCount:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_pixels": ("INT", {
                    "default": 1048576,
                    "min": 1,
                    "max": 0xffffffff,
                    "step": 1,
                    "display": "number"
                }),
                "divideBy": ([1, 8, 16, 64], {"default": 16}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "min_R", "max_R")
    FUNCTION = "calculate"
    CATEGORY = categoryFolder

    def calculate(self, images, max_pixels, divideBy):
        if len(images.shape) == 4:
            h, w = images.shape[1], images.shape[2]
        elif len(images.shape) == 3:
            h, w = images.shape[0], images.shape[1]
        else:
            raise ValueError(f"Unsupported tensor form: {images.shape}")

        l = max_pixels
        r = w / h
        k = divideBy
        nw = int(math.floor(math.sqrt(l * r)))
        nh = int(math.floor(nw / r))
        nw = nw - nw % k
        nh = nh - nh % k

        if (nw * nh > l):
            nw = nw - k
            nh = int(math.floor(nw / r))

        min_r = min(nw, nh)
        max_r = max(nw, nh)

        return (nw, nh, min_r, max_r)

#Calc multiplier for interpolation video
class CP_InterpolateMultiplier:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"forceInput": True}),
                "limit": ("INT", {"default": 50}),
                "isInterpolate": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("multiplier", "fps")
    FUNCTION = "calculate"
    CATEGORY = categoryFolder

    def calculate(self, fps, limit, isInterpolate):
        m = 0
        f = 0
        if limit > 0:
            m = limit // fps
        else:
            m = 1
        
        if isInterpolate == True:
            f = fps * m
        else:
            f = fps

        return (float(m), float(f))

#Get info and calc custom resolution
class CP_ImageInfoAndCustomResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscaleFactor": ("INT", {"default": 2048, "min": 512, "max": 100000, "step": 8})
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "minResolution", "maxResolution", "upscaleRatioByMin", "upscaleRatioByMax")
    FUNCTION = "calculate"
    CATEGORY = categoryFolder

    def calculate(self, image, upscaleFactor):
        if isinstance(image, torch.Tensor):
            _, h, w, _ = image.shape
        else:
            h, w = image.shape[:2]
        
        width = int(w)
        height = int(h)
        minResolution = min(width, height)
        maxResolution = max(width, height)

        if maxResolution == 0 or upscaleFactor == 0:
            upscaleRatioByMin = 1.0
        else:
            upscaleRatioByMin = upscaleFactor / minResolution
        
        if maxResolution == 0 or upscaleFactor == 0:
            upscaleRatioByMax = 1.0
        else:
            upscaleRatioByMax = upscaleFactor / maxResolution


        return (width, height, minResolution, maxResolution, upscaleRatioByMin, upscaleRatioByMax)

#Calc outpaint resolution
class CP_ImageOutpaintCalcResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1})
            }
        }
    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "extended_up_factor", "extended_down_factor", "extended_left_factor", "extended_right_factor")
    FUNCTION = "calc"
    CATEGORY = categoryFolder

    def calc(self, image, top, left, right, bottom):
        if isinstance(image, torch.Tensor):
            _, h, w, _ = image.shape
        else:
            h, w = image.shape[:2]
        
        width = int(w) + left + right
        height = int(h) + top + bottom
        digit = 2
        u = 1.0 + round(top, digit) / height
        d = 1.0 + round(bottom, digit) / width
        l = 1.0 + round(left, digit) / height
        r = 1.0 + round(right, digit) / width
        
        return (width, height, u, d, l, r)

#Simple int switcher
class CP_SimpleIndexSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "do_it"
    CATEGORY = categoryFolder
    
    def do_it(self, index):
        return (index,)

#Reverse boolean
class CP_ReverseBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flag": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "reverseBoolean"
    CATEGORY = categoryFolder

    def reverseBoolean(self, flag):
        return(not flag,)

#Save string as txt file
class CP_SaveTextToFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "encoding": (["utf-8", "win-1251"], {"default": "utf-8"}),
                "useTimePrefix": ("BOOLEAN", {"default": False}),
                "destination": ("STRING", {"default": ""}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_text"
    CATEGORY = categoryFolder

    def save_text(self, data, encoding, useTimePrefix, destination, prefix=""):
        if prefix and prefix.strip():
            base = os.path.splitext(os.path.basename(prefix))[0]
        else:
            base = ""
        
        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        if base:
            file_name = f"{base}_{date_str}.txt"
        else:
            file_name = f"{date_str}.txt"
        
        if not useTimePrefix:
            file_name = f"{base}.txt"

        if destination == "":
            full_dir = os.path.join(folder_paths.get_output_directory(), "Txts")
        else:
            full_dir = destination
        os.makedirs(full_dir, exist_ok=True)
        file_path = os.path.join(full_dir, file_name)
        try:
            with open(file_path, "w", encoding=encoding) as f:
                f.write(data)
            print(f"[{categoryFolder}] file saved into {file_path}")
        except Exception as e:
            raise RuntimeError(f"[{categoryFolder}] Error save text file: {e}")

        return (file_path,)

#Include watermark and text by coordinates
class CP_BatchImageWatermarkText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "C:/images/", "multiline": False}),
                "watermark_path": ("STRING", {"default": "watermark.png", "multiline": False}),
                "watermark_x": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "watermark_y": ("INT", {"default": 20, "min": 0, "max": 10000, "step": 1}),
                "watermark_width": ("INT", {"default": 120, "min": 1, "max": 10000, "step": 1}),
                "watermark_height": ("INT", {"default": 80, "min": 1, "max": 10000, "step": 1}),
                "text_prefix": ("STRING", {"default": "Steps: ", "multiline": False}),
                "start_value": ("FLOAT", {"default": 0.0, "min": -1000000.0, "max": 1000000.0, "step": 0.1}),
                "step_value": ("FLOAT", {"default": 23.0, "min": -1000000.0, "max": 1000000.0, "step": 0.1}),
                "text_x": ("INT", {"default": 30, "min": 0, "max": 10000, "step": 1}),
                "text_y": ("INT", {"default": 900, "min": 0, "max": 10000, "step": 1}),
                "font_size": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
                "font_color": ("STRING", {"default": "red", "multiline": False}),
                "font_path": ("STRING", {"default": "times.ttf", "multiline": False}),
                "output_folder": ("STRING", {"default": "output", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "process_images"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    def process_images(self, input_folder, watermark_path, watermark_x, watermark_y,
                       watermark_width, watermark_height, text_prefix, start_value,
                       step_value, text_x, text_y, font_size, font_color, font_path,
                       output_folder):

        input_folder = os.path.normpath(input_folder)
        watermark_path = os.path.normpath(watermark_path)
        output_sub = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_folder = os.path.join(folder_paths.get_output_directory(), output_sub)

        if not os.path.isdir(input_folder):
            raise ValueError(f"ERROR: directory {input_folder} not found.")

        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

        image_files = sorted(set(image_files))
        if not image_files:
            raise ValueError(f"ERRPR: images not found in {input_folder}.")

        try:
            watermark = Image.open(watermark_path).convert("RGBA")
        except Exception as e:
            raise ValueError(f"ERROR: watermark not found in {watermark_path}\n{e}")

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            print(f"WARNING: font {font_path} not found, used default.")
        os.makedirs(output_folder, exist_ok=True)

        saved_paths = []

        for idx, img_path in enumerate(image_files):
            try:
                base_img = Image.open(img_path).convert("RGBA")
            except Exception as e:
                print(f"ERROR: can't open {img_path}: {e}")
                continue

            wm_copy = watermark.copy()
            wm_copy.thumbnail((watermark_width, watermark_height), Image.Resampling.LANCZOS)
            layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            layer.paste(wm_copy, (watermark_x, watermark_y), wm_copy)
            base_img = Image.alpha_composite(base_img, layer)
            draw = ImageDraw.Draw(base_img)
            current_value = start_value + idx * step_value
            if current_value.is_integer():
                value_str = str(int(current_value))
            else:
                value_str = f"{current_value:.2f}".rstrip('0').rstrip('.')
            text = f"{text_prefix}{value_str}"
            draw.text((text_x, text_y), text, fill=font_color, font=font)

            if base_img.mode == "RGBA":
                rgb_img = Image.new("RGB", base_img.size, (255, 255, 255))
                rgb_img.paste(base_img, mask=base_img.split()[3])
                save_img = rgb_img
            else:
                save_img = base_img
            filename = f"{idx:05d}.png"
            out_path = os.path.join(output_folder, filename)
            save_img.save(out_path, "PNG")
            saved_paths.append(out_path)
        result_str = "\n".join(saved_paths)
        return (result_str,)

#Transfer token prompt into natural language prompt
DEFAULT_SYSTEM_PROMPT = """You convert tokenized image prompts into natural language prompts for FLUX and Z-Image models.
Rules:
1. Preserve all important visual details, style words, camera hints, lighting, composition, quality markers, and negative constraints.
2. Expand shorthand tokens into clear natural language.
3. Remove duplicates and obvious contradictions.
4. Return only prompt text in English, without meta-commentary."""

OUTPUT_TEMPLATE_DEFAULT = "default"
OUTPUT_TEMPLATE_ZIMAGE_STRUCT = "z-image struct"
LLM_DEVICE_GPU = "gpu"
LLM_DEVICE_CPU = "cpu"
MODEL_RESIDENCY_KEEP = "keep_loaded"
MODEL_RESIDENCY_UNLOAD = "unload_after_run"
MODEL_RESIDENCY_FULL_UNLOAD = "full_unload"
RAM_KEEPER_CONTEXT_SIZE = 512

STRUCT_SECTION_ORDER = (
    "Subject",
    "Scene",
    "Composition",
    "Lighting",
    "Style",
    "Constraints",
)
TOKEN_COVERAGE_RETRY_ROUNDS = 1
MAX_MISSING_TOKEN_INJECTION = 24


class CP_LocalLLMTokenPromptToNatural:
    _llama_cache: dict[tuple[str, int, int], Any] = {}
    _ram_keeper_keys: set[tuple[str, int, int]] = set()
    _cache_lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "token_prompt": ("STRING", {"multiline": True, "default": ""}),
                "output_template": (
                    [OUTPUT_TEMPLATE_DEFAULT, OUTPUT_TEMPLATE_ZIMAGE_STRUCT],
                    {"default": OUTPUT_TEMPLATE_DEFAULT},
                ),
                "backend": (["ollama", "llama_cpp"], {"default": "ollama"}),
                "model_name": ("STRING", {"default": "llama3.1:8b-instruct-q4_K_M"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model_path": ("STRING", {"default": ""}),
                "llm_device": ([LLM_DEVICE_GPU, LLM_DEVICE_CPU], {"default": LLM_DEVICE_GPU}),
                "model_residency": (
                    [MODEL_RESIDENCY_KEEP, MODEL_RESIDENCY_UNLOAD, MODEL_RESIDENCY_FULL_UNLOAD],
                    {"default": MODEL_RESIDENCY_KEEP},
                ),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.05, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 220, "min": 16, "max": 2048, "step": 1}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647, "step": 1}),
                "timeout_sec": ("INT", {"default": 120, "min": 5, "max": 600, "step": 1}),
                "context_size": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 256}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200, "step": 1}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("natural_prompt",)
    FUNCTION = "convert"
    CATEGORY = "text/llm"

    @staticmethod
    def _build_user_prompt(token_prompt: str, output_template: str, token_checklist: list[str]) -> str:
        checklist_block = ""
        if token_checklist:
            checklist_lines = [f"{idx}. {token}" for idx, token in enumerate(token_checklist, start=1)]
            checklist_block = (
                "\nMandatory token checklist (do not drop any item, keep semantics explicit):\n"
                + "\n".join(checklist_lines)
            )

        if output_template == OUTPUT_TEMPLATE_ZIMAGE_STRUCT:
            return (
                "Rewrite the token prompt below into a natural language prompt for text-to-image generation.\n"
                "You must preserve all token semantics from the checklist.\n"
                "Return the answer in English and in this exact structure (all six items are mandatory):\n"
                "1. Subject: ...\n"
                "2. Scene: ...\n"
                "3. Composition: ...\n"
                "4. Lighting: ...\n"
                "5. Style: ...\n"
                "6. Constraints: ...\n"
                "If any item is unknown, write: not specified.\n\n"
                f"Token prompt:\n{token_prompt}{checklist_block}"
            )
        return (
            "Rewrite the token prompt below into a fluent natural language prompt for text-to-image generation.\n"
            "Do not omit any relevant token concept; use as many sentences as needed for full coverage.\n"
            "Return only the final prompt text in English.\n\n"
            f"Token prompt:\n{token_prompt}{checklist_block}"
        )

    @staticmethod
    def _clean_output(text: str, preserve_newlines: bool = False) -> str:
        cleaned = (text or "").strip().strip('"').strip("'")
        if preserve_newlines:
            cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
            lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
            return "\n".join(lines)
        return " ".join(cleaned.split())

    @staticmethod
    def _split_prompt_tokens(token_prompt: str) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []
        stack: list[str] = []
        opening = {"(": ")", "[": "]", "{": "}", "<": ">"}
        closing = {")": "(", "]": "[", "}": "{", ">": "<"}

        for char in token_prompt:
            if char in opening:
                stack.append(char)
                buffer.append(char)
                continue
            if char in closing:
                if stack and stack[-1] == closing[char]:
                    stack.pop()
                buffer.append(char)
                continue
            if char in {",", "\n"} and not stack:
                fragment = "".join(buffer).strip()
                if fragment:
                    tokens.append(fragment)
                buffer = []
                continue
            buffer.append(char)

        tail = "".join(buffer).strip()
        if tail:
            tokens.append(tail)
        return tokens

    @staticmethod
    def _normalize_token_fragment(fragment: str) -> str:
        value = (fragment or "").strip().strip('"').strip("'")
        if not value:
            return ""
        # Remove common weight suffixes like "(detail:1.2)" -> "detail".
        value = re.sub(r"\s*:\s*-?\d+(?:\.\d+)?\s*$", "", value)
        # Unwrap simple emphasis brackets.
        for _ in range(3):
            if len(value) >= 2 and (
                (value[0] == "(" and value[-1] == ")")
                or (value[0] == "[" and value[-1] == "]")
                or (value[0] == "{" and value[-1] == "}")
            ):
                value = value[1:-1].strip()
            else:
                break
        value = value.replace("_", " ")
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @classmethod
    def _extract_token_checklist(cls, token_prompt: str) -> list[str]:
        raw_parts = cls._split_prompt_tokens(token_prompt)
        checklist: list[str] = []
        seen: set[str] = set()

        for part in raw_parts:
            normalized = cls._normalize_token_fragment(part)
            if len(normalized) < 2:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            checklist.append(normalized)
        return checklist

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        normalized = (text or "").lower().replace("_", " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def _is_token_covered(cls, token: str, output_text: str) -> bool:
        token_norm = cls._normalize_text_for_match(token)
        output_norm = cls._normalize_text_for_match(output_text)
        if not token_norm:
            return True
        if token_norm in output_norm:
            return True

        token_words = [word for word in token_norm.split() if len(word) >= 3]
        if not token_words:
            token_words = token_norm.split()
        if not token_words:
            return True

        unique_words = list(dict.fromkeys(token_words))
        hits = 0
        for word in unique_words:
            if re.search(rf"\b{re.escape(word)}\b", output_norm):
                hits += 1

        threshold = max(1, (len(unique_words) + 1) // 2)
        return hits >= threshold

    @classmethod
    def _find_missing_tokens(cls, token_checklist: list[str], output_text: str) -> list[str]:
        missing: list[str] = []
        for token in token_checklist:
            if not cls._is_token_covered(token, output_text):
                missing.append(token)
        return missing

    @staticmethod
    def _build_retry_prompt(
        token_prompt: str,
        output_template: str,
        token_checklist: list[str],
        previous_output: str,
        missing_tokens: list[str],
    ) -> str:
        missing_lines = [f"{idx}. {token}" for idx, token in enumerate(missing_tokens, start=1)]
        if output_template == OUTPUT_TEMPLATE_ZIMAGE_STRUCT:
            return (
                "Revise the draft prompt below.\n"
                "Keep the same 6-item output structure exactly:\n"
                "1. Subject\n2. Scene\n3. Composition\n4. Lighting\n5. Style\n6. Constraints\n"
                "You must explicitly include all missing token semantics.\n\n"
                f"Original token prompt:\n{token_prompt}\n\n"
                "Current draft:\n"
                f"{previous_output}\n\n"
                "Missing token checklist:\n"
                + "\n".join(missing_lines)
            )
        return (
            "Revise the draft prompt below.\n"
            "Keep it natural, but ensure all missing token semantics are explicitly included.\n\n"
            f"Original token prompt:\n{token_prompt}\n\n"
            "Current draft:\n"
            f"{previous_output}\n\n"
            "Missing token checklist:\n"
            + "\n".join(missing_lines)
        )

    @staticmethod
    def _append_missing_tokens(output_text: str, output_template: str, missing_tokens: list[str]) -> str:
        if not missing_tokens:
            return output_text
        limited = missing_tokens[:MAX_MISSING_TOKEN_INJECTION]
        suffix = ", ".join(limited)
        if len(missing_tokens) > MAX_MISSING_TOKEN_INJECTION:
            suffix = f"{suffix}, and more"

        if output_template == OUTPUT_TEMPLATE_ZIMAGE_STRUCT:
            structured = CP_LocalLLMTokenPromptToNatural._force_zimage_structure(output_text)
            lines = structured.split("\n")
            updated_lines: list[str] = []
            for line in lines:
                if line.lower().startswith("6. constraints:"):
                    head, _, tail = line.partition(":")
                    tail_text = tail.strip()
                    if tail_text and tail_text.lower() != "not specified":
                        tail_text = f"{tail_text}; Include explicitly: {suffix}."
                    else:
                        tail_text = f"Include explicitly: {suffix}."
                    updated_lines.append(f"{head}: {tail_text}")
                else:
                    updated_lines.append(line)
            return "\n".join(updated_lines)

        base = output_text.strip()
        separator = "" if base.endswith((".", "!", "?")) else "."
        return f"{base}{separator} Include explicitly: {suffix}."

    @staticmethod
    def _postprocess_output(raw_text: str, output_template: str) -> str:
        output = CP_LocalLLMTokenPromptToNatural._clean_output(
            raw_text,
            preserve_newlines=output_template == OUTPUT_TEMPLATE_ZIMAGE_STRUCT,
        )
        if output_template == OUTPUT_TEMPLATE_ZIMAGE_STRUCT:
            output = CP_LocalLLMTokenPromptToNatural._force_zimage_structure(output)
        return output

    def _generate_with_coverage_guard(
        self,
        token_prompt: str,
        output_template: str,
        generate_once,
    ) -> str:
        token_checklist = self._extract_token_checklist(token_prompt)
        user_prompt = self._build_user_prompt(token_prompt, output_template, token_checklist)
        output = self._postprocess_output(generate_once(user_prompt), output_template)
        if not output:
            return output

        missing = self._find_missing_tokens(token_checklist, output)
        for _ in range(TOKEN_COVERAGE_RETRY_ROUNDS):
            if not missing:
                break
            retry_prompt = self._build_retry_prompt(
                token_prompt=token_prompt,
                output_template=output_template,
                token_checklist=token_checklist,
                previous_output=output,
                missing_tokens=missing,
            )
            revised_output = self._postprocess_output(generate_once(retry_prompt), output_template)
            if not revised_output:
                break
            revised_missing = self._find_missing_tokens(token_checklist, revised_output)
            if len(revised_missing) <= len(missing):
                output = revised_output
                missing = revised_missing

        if missing:
            output = self._append_missing_tokens(output, output_template, missing)
        return output

    @staticmethod
    def _force_zimage_structure(text: str) -> str:
        values: dict[str, str] = {name: "" for name in STRUCT_SECTION_ORDER}
        section_pattern = re.compile(
            r"^\s*(?:\d+\.\s*)?(subject|scene|composition|lighting|style|constraints)\s*:\s*(.*)$",
            re.IGNORECASE,
        )

        current_key = None
        for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            match = section_pattern.match(line_stripped)
            if match:
                matched_key = match.group(1).strip().lower()
                content = match.group(2).strip()
                for canonical_name in STRUCT_SECTION_ORDER:
                    if canonical_name.lower() == matched_key:
                        current_key = canonical_name
                        values[current_key] = content
                        break
            elif current_key:
                prior = values[current_key]
                values[current_key] = f"{prior} {line_stripped}".strip()

        normalized_lines = []
        for idx, section_name in enumerate(STRUCT_SECTION_ORDER, start=1):
            value = values[section_name].strip() or "not specified"
            normalized_lines.append(f"{idx}. {section_name}: {value}")
        return "\n".join(normalized_lines)

    @classmethod
    def _resolve_model_path(cls, model_path: str, model_name: str) -> str:
        raw_path = (model_path or "").strip().strip('"').strip("'")
        raw_name = (model_name or "").strip().strip('"').strip("'")

        if not raw_path:
            raise ValueError("model_path is required when backend='llama_cpp'.")

        expanded_path = os.path.expandvars(os.path.expanduser(raw_path))
        normalized_path = os.path.normpath(expanded_path)

        candidate_path = normalized_path
        if os.path.isdir(normalized_path):
            if not raw_name:
                raise ValueError(
                    "model_path points to a directory. Set model_name to your GGUF filename."
                )
            candidate_path = os.path.join(normalized_path, raw_name)

        if not os.path.exists(candidate_path):
            raise ValueError(f"GGUF model file does not exist: {candidate_path}")
        if not os.path.isfile(candidate_path):
            raise ValueError(f"GGUF model path is not a file: {candidate_path}")
        if not candidate_path.lower().endswith(".gguf"):
            raise ValueError(f"Model file must be .gguf: {candidate_path}")

        return os.path.normpath(candidate_path)

    @classmethod
    def _resolve_effective_gpu_layers(cls, llm_device: str, gpu_layers: int) -> int:
        if llm_device == LLM_DEVICE_CPU:
            return 0
        # In GPU mode, non-positive values mean "offload as many layers as possible".
        if gpu_layers <= 0:
            return 999
        return gpu_layers

    @staticmethod
    def _supports_llama_gpu_offload() -> bool | None:
        try:
            import llama_cpp
        except Exception:
            return None

        supports_fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(supports_fn):
            try:
                return bool(supports_fn())
            except Exception:
                return None
        return None

    @classmethod
    def _close_llama_instance(cls, instance: Any) -> None:
        close_fn = getattr(instance, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    @classmethod
    def _unload_llama_by_key(cls, cache_key: tuple[str, int, int]) -> None:
        instance = None
        with cls._cache_lock:
            instance = cls._llama_cache.pop(cache_key, None)
            cls._ram_keeper_keys.discard(cache_key)
        if instance is not None:
            cls._close_llama_instance(instance)

    @classmethod
    def _ensure_ram_keeper(cls, model_path: str, model_name: str) -> None:
        try:
            _, keeper_key = cls._load_llama(
                model_path=model_path,
                model_name=model_name,
                context_size=RAM_KEEPER_CONTEXT_SIZE,
                effective_gpu_layers=0,
            )
            with cls._cache_lock:
                cls._ram_keeper_keys.add(keeper_key)
        except Exception:
            # Best effort: inference result should not fail if RAM parking is unavailable.
            pass

    @classmethod
    def _drop_ram_keepers_for_model_path(cls, resolved_model_path: str) -> None:
        to_remove: list[tuple[str, int, int]] = []
        with cls._cache_lock:
            for key in list(cls._ram_keeper_keys):
                if key[0] == resolved_model_path:
                    to_remove.append(key)
                    cls._ram_keeper_keys.discard(key)
        for key in to_remove:
            cls._unload_llama_by_key(key)

    @classmethod
    def _load_llama(
        cls,
        model_path: str,
        model_name: str,
        context_size: int,
        effective_gpu_layers: int,
    ):
        resolved_model_path = cls._resolve_model_path(model_path=model_path, model_name=model_name)

        cache_key = (resolved_model_path, context_size, effective_gpu_layers)

        with cls._cache_lock:
            cached = cls._llama_cache.get(cache_key)
            if cached is not None:
                return cached, cache_key

            try:
                from llama_cpp import Llama
            except Exception as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install dependencies from requirements.txt."
                ) from exc

            if effective_gpu_layers > 0:
                supports_gpu = cls._supports_llama_gpu_offload()
                if supports_gpu is False:
                    raise RuntimeError(
                        "Current llama-cpp-python build does not support GPU offload. "
                        "Install a GPU-enabled llama-cpp-python wheel/build."
                    )

            instance = Llama(
                model_path=resolved_model_path,
                n_ctx=context_size,
                n_gpu_layers=effective_gpu_layers,
                verbose=False,
            )
            cls._llama_cache[cache_key] = instance
            return instance, cache_key

    def _call_ollama(
        self,
        token_prompt: str,
        model_name: str,
        endpoint: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        timeout_sec: int,
        output_template: str,
        llm_device: str,
        model_residency: str,
        system_prompt: str,
    ) -> str:
        url = f"{endpoint.rstrip('/')}/api/generate"
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }
        if seed >= 0:
            options["seed"] = seed
        if llm_device == LLM_DEVICE_CPU:
            options["num_gpu"] = 0

        def generate_once(user_prompt: str) -> str:
            payload = {
                "model": model_name,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": options,
            }
            if model_residency == MODEL_RESIDENCY_KEEP:
                payload["keep_alive"] = "30m"
            elif model_residency == MODEL_RESIDENCY_UNLOAD:
                payload["keep_alive"] = 0
            try:
                response = requests.post(url, json=payload, timeout=timeout_sec)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"Ollama request failed: {exc}") from exc
            data = response.json()
            return data.get("response", "")

        output = self._generate_with_coverage_guard(
            token_prompt=token_prompt,
            output_template=output_template,
            generate_once=generate_once,
        )
        if not output:
            raise RuntimeError("Ollama returned an empty response.")
        return output

    def _call_llama_cpp(
        self,
        token_prompt: str,
        model_path: str,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        context_size: int,
        gpu_layers: int,
        output_template: str,
        llm_device: str,
        model_residency: str,
        system_prompt: str,
    ) -> str:
        resolved_model_path = self._resolve_model_path(model_path=model_path, model_name=model_name)
        effective_gpu_layers = self._resolve_effective_gpu_layers(llm_device=llm_device, gpu_layers=gpu_layers)
        cache_key = None
        try:
            llm, cache_key = self._load_llama(
                model_path=model_path,
                model_name=model_name,
                context_size=context_size,
                effective_gpu_layers=effective_gpu_layers,
            )
        except Exception as exc:
            raise RuntimeError(
                "llama.cpp model init failed. Check model_path/model_name and ensure llama-cpp-python is up to date."
            ) from exc

        def generate_once(user_prompt: str) -> str:
            request_payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if seed >= 0:
                request_payload["seed"] = seed
            try:
                result = llm.create_chat_completion(**request_payload)
            except Exception as exc:
                raise RuntimeError(f"llama.cpp inference failed: {exc}") from exc
            try:
                return result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("llama.cpp returned an unexpected response format.") from exc

        try:
            output = self._generate_with_coverage_guard(
                token_prompt=token_prompt,
                output_template=output_template,
                generate_once=generate_once,
            )
        finally:
            if cache_key is not None:
                if model_residency == MODEL_RESIDENCY_UNLOAD:
                    if llm_device == LLM_DEVICE_GPU:
                        self._ensure_ram_keeper(model_path=model_path, model_name=model_name)
                    self._unload_llama_by_key(cache_key)
                elif model_residency == MODEL_RESIDENCY_FULL_UNLOAD:
                    self._unload_llama_by_key(cache_key)
                    self._drop_ram_keepers_for_model_path(resolved_model_path)
        if not output:
            raise RuntimeError("llama.cpp returned an empty response.")
        return output

    def convert(
        self,
        token_prompt: str,
        output_template: str,
        backend: str,
        model_name: str,
        endpoint: str,
        model_path: str,
        llm_device: str,
        model_residency: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        timeout_sec: int,
        context_size: int,
        gpu_layers: int,
        system_prompt: str,
    ):
        normalized_prompt = token_prompt.strip()
        if not normalized_prompt:
            return ("",)
        normalized_template = (output_template or OUTPUT_TEMPLATE_DEFAULT).strip().lower()
        if normalized_template not in (OUTPUT_TEMPLATE_DEFAULT, OUTPUT_TEMPLATE_ZIMAGE_STRUCT):
            raise ValueError(f"Unsupported output_template: {output_template}")
        normalized_device = (llm_device or LLM_DEVICE_GPU).strip().lower()
        if normalized_device not in (LLM_DEVICE_GPU, LLM_DEVICE_CPU):
            raise ValueError(f"Unsupported llm_device: {llm_device}")
        normalized_residency = (model_residency or MODEL_RESIDENCY_KEEP).strip().lower()
        if normalized_residency not in (
            MODEL_RESIDENCY_KEEP,
            MODEL_RESIDENCY_UNLOAD,
            MODEL_RESIDENCY_FULL_UNLOAD,
        ):
            raise ValueError(f"Unsupported model_residency: {model_residency}")

        if backend == "ollama":
            natural_prompt = self._call_ollama(
                token_prompt=normalized_prompt,
                model_name=model_name,
                endpoint=endpoint,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                timeout_sec=timeout_sec,
                output_template=normalized_template,
                llm_device=normalized_device,
                model_residency=normalized_residency,
                system_prompt=system_prompt,
            )
        elif backend == "llama_cpp":
            natural_prompt = self._call_llama_cpp(
                token_prompt=normalized_prompt,
                model_path=model_path,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                context_size=context_size,
                gpu_layers=gpu_layers,
                output_template=normalized_template,
                llm_device=normalized_device,
                model_residency=normalized_residency,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return (natural_prompt,)

#Seed batch generator
class CP_SeedIncrementBatchGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffffffffff6, "control_before_generate": True}),
            },
        }

    RETURN_TYPES = ("INT",) * 10
    RETURN_NAMES = (
        "SeedShift0",
        "SeedShift1",
        "SeedShift2",
        "SeedShift3",
        "SeedShift4",
        "SeedShift5",
        "SeedShift6",
        "SeedShift7",
        "SeedShift8",
        "SeedShift9",
    )
    FUNCTION = "generate"
    CATEGORY = categoryFolder

    def generate(self, seed):
        return tuple(seed + shift for shift in range(10))

#vae forward node (crutch for VHS anything anywhere)
class CP_VaeForwarding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "forward"
    CATEGORY = categoryFolder

    def forward(self, vae):
        return (vae,)

NODE_CLASS_MAPPINGS_TOOLS = {
    "CP AspectRatioCheck" : CP_AspectRatioCheck,
    "CP Float 0 to 1" : CP_FloatZeroToOne,
    "CP Keep Image": CP_KeepImage,
    "CP Keep Integer": CP_KeepInteger,
    "CP Lora List": CP_LoraList,
    "CP ImageResolutionSelector": CP_ImageResolutionSelector,
    "CP Seed List": CP_SeedList,
    "CP StringProcessor": CP_StringProcessor,
    "CP StringSelector": CP_StringSelector,
    "CP UpscaleFactorSelect": CP_UpscaleFactorSelect,
    "CP StepsSelect": CP_StepsSelect,
    "CP IntPosSelect": CP_IntPosSelect,
    "CP ImageCheck": CP_ImageCheck,
    "CP SeedGeneratorShort": CP_SeedGeneratorShort,
    "CP CustomResolution": CP_CustomResolution,
    "CP ResolutionByPixelCount": CP_ResolutionByPixelCount,
    "CP InterpolateMultiplier": CP_InterpolateMultiplier,
    "CP ImageInfoAndCustomResolution": CP_ImageInfoAndCustomResolution,
    "CP ImageOutpaintCalcResolution": CP_ImageOutpaintCalcResolution,
    "CP SimpleIndexSwitch": CP_SimpleIndexSwitch,
    "CP ReverseBoolean": CP_ReverseBoolean,
    "CP SaveTextToFile": CP_SaveTextToFile,
    "CP BatchImageWatermarkText": CP_BatchImageWatermarkText,
    "CP AssemblyFoldersDatasetPreparation": CP_AssemblyFoldersDatasetPreparation,
    "CP LocalLLMTokenPromptToNatural": CP_LocalLLMTokenPromptToNatural,
    "CP SeedIncrementBatchGen": CP_SeedIncrementBatchGen,
    "CP VaeForwarding": CP_VaeForwarding,
}

NODE_DISPLAY_NAME_MAPPINGS_TOOLS = {
    "CP AspectRatioCheck" : "Aspect ratio check",
    "CP Float 0 to 1" : "Float 0 to 1",
    "CP Keep Image": "Keep image",
    "CP Keep Integer": "Keep integer",
    "CP Lora List": "Lora list",
    "CP ImageResolutionSelector": "Image resolution selector",
    "CP Seed List": "Seed list",
    "CP StringProcessor": "String processor",
    "CP StringSelector": "String selector",
    "CP UpscaleFactorSelect": "Upscale factor select by flag",
    "CP StepsSelect": "Steps select by flag",
    "CP IntPosSelect": "Int positive select by flag",
    "CP ImageCheck": "Image Check",
    "CP SeedGeneratorShort": "Seed generator (short)",
    "CP CustomResolution": "Select resolution",
    "CP ResolutionByPixelCount": "Resolution by pixel count",
    "CP InterpolateMultiplier": "Calc interpolate multiplier",
    "CP ImageInfoAndCustomResolution": "Image info and custom resolution",
    "CP ImageOutpaintCalcResolution": " Image outpaint calc resolution",
    "CP SimpleIndexSwitch": "Simple Index Switch",
    "CP ReverseBoolean": "Reverse boolean",
    "CP SaveTextToFile": "Save Text To File",
    "CP BatchImageWatermarkText": "Batch Image Watermark Text",
    "CP AssemblyFoldersDatasetPreparation": "Assembly folders dataset preparation",
    "CP LocalLLMTokenPromptToNatural": "Token Prompt -> Natural Prompt (Local LLM)",
    "CP SeedIncrementBatchGen": "Seed Increment Batch Gen",
    "CP VaeForwarding": "VAE Forwarding",
}
