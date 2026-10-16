from __future__ import annotations
import ast
import json
import os
import operator as op
import random
import folder_paths
from datetime import datetime
import torch
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
import math
import glob

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


class CP_ShowAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = categoryFolder

    @staticmethod
    def _to_text(value):
        if isinstance(value, str):
            return value

        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2, default=str)
            except (TypeError, ValueError):
                pass

        return str(value)

    def show(self, anything):
        text = self._to_text(anything)
        return {"ui": {"text": [text]}, "result": (text,)}


_MATH_EXPRESSION_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
    ast.BitAnd: op.and_,
    ast.BitOr: op.or_,
    ast.Invert: op.invert,
    ast.And: lambda left, right: 1 if left and right else 0,
    ast.Or: lambda left, right: 1 if left or right else 0,
    ast.Not: lambda value: 0 if value else 1,
    ast.RShift: op.rshift,
    ast.LShift: op.lshift,
}

_MATH_EXPRESSION_FUNCTIONS = {
    "round": ((1, 2), lambda value, digits=None: round(value, digits)),
    "ceil": ((1, 1), math.ceil),
    "floor": ((1, 1), math.floor),
    "min": ((2, None), min),
    "max": ((2, None), max),
    "randomint": ((2, 2), random.randint),
    "randomchoice": ((2, None), lambda *values: random.choice(values)),
    "sqrt": ((1, 1), math.sqrt),
    "int": ((1, 1), int),
    "iif": (
        (3, 3),
        lambda condition, true_value, false_value: true_value if condition else false_value,
    ),
}

_MATH_EXPRESSION_COMPARATORS = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
    ast.Lt: op.lt,
    ast.LtE: op.le,
}


class CP_MathExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": False},
                ),
            },
            "optional": {
                "a": (any,),
                "b": (any,),
                "c": (any,),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    FUNCTION = "evaluate"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, expression, **_kwargs):
        if "random" in expression:
            return float("nan")
        return expression

    @staticmethod
    def _get_widget_value(extra_pnginfo, prompt, node_name, widget_name):
        workflow = (extra_pnginfo or {}).get("workflow") or {"nodes": []}
        node_id = None

        for workflow_node in workflow.get("nodes", []):
            properties = workflow_node.get("properties") or {}
            names = (
                properties.get("Node name for S&R"),
                workflow_node.get("type"),
                workflow_node.get("title"),
            )
            if node_name in names:
                node_id = workflow_node.get("id")
                break

        if node_id is None:
            raise NameError(f"Node not found: {node_name}.{widget_name}")

        prompt_node = (prompt or {}).get(str(node_id), {})
        inputs = prompt_node.get("inputs", {})
        if widget_name not in inputs:
            raise NameError(f"Widget not found: {node_name}.{widget_name}")

        value = inputs[widget_name]
        if isinstance(value, list):
            raise ValueError(
                "Converted widgets are not supported via named reference; "
                "use an input instead."
            )
        return value

    @staticmethod
    def _get_size(target, property_name):
        if isinstance(target, dict) and "samples" in target:
            samples = target["samples"]
            return samples.shape[3 if property_name == "width" else 2] * 8

        if hasattr(target, "shape") and len(target.shape) >= 3:
            return target.shape[2 if property_name == "width" else 1]

        raise TypeError("width/height can only be read from an IMAGE or LATENT input")

    def evaluate(self, expression, prompt=None, extra_pnginfo=None, a=None, b=None, c=None):
        parsed = ast.parse(
            expression.replace("\n", " ").replace("\r", ""),
            mode="eval",
        ).body
        lookup = {"a": a, "b": b, "c": c}

        def eval_binary(operator_node, left_node, right_node):
            operator = _MATH_EXPRESSION_OPERATORS.get(type(operator_node))
            if operator is None:
                raise NotImplementedError(
                    f"Operator {type(operator_node).__name__} is not supported"
                )

            left = eval_expr(left_node)
            right = eval_expr(right_node)
            left = left if isinstance(left, int) else float(left)
            right = right if isinstance(right, int) else float(right)
            return operator(left, right)

        def eval_expr(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise TypeError(f"Unsupported constant: {node.value!r}")

            if isinstance(node, ast.BinOp):
                return eval_binary(node.op, node.left, node.right)

            if isinstance(node, ast.BoolOp):
                result = eval_expr(node.values[0])
                for value_node in node.values[1:]:
                    operator = _MATH_EXPRESSION_OPERATORS[type(node.op)]
                    result = operator(result, eval_expr(value_node))
                return result

            if isinstance(node, ast.UnaryOp):
                operator = _MATH_EXPRESSION_OPERATORS.get(type(node.op))
                if operator is None:
                    raise NotImplementedError(
                        f"Operator {type(node.op).__name__} is not supported"
                    )
                return operator(eval_expr(node.operand))

            if isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Name):
                    raise TypeError("Only direct input or node widget references are supported")

                source_name = node.value.id
                if source_name in lookup and node.attr in ("width", "height"):
                    return self._get_size(lookup[source_name], node.attr)
                return self._get_widget_value(
                    extra_pnginfo,
                    prompt,
                    source_name,
                    node.attr,
                )

            if isinstance(node, ast.Name):
                if node.id not in lookup:
                    raise NameError(f"Name not found: {node.id}")

                value = lookup[node.id]
                if isinstance(value, (int, float, complex)):
                    return value
                raise TypeError(
                    "Complex types (LATENT/IMAGE) must reference width or height, "
                    f"for example {node.id}.width"
                )

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise NameError("Invalid function call")
                if node.keywords:
                    raise SyntaxError("Keyword arguments are not supported")

                function_data = _MATH_EXPRESSION_FUNCTIONS.get(node.func.id)
                if function_data is None:
                    raise NameError(f"Invalid function call: {node.func.id}")

                (minimum, maximum), function = function_data
                argument_count = len(node.args)
                if argument_count < minimum or (
                    maximum is not None and argument_count > maximum
                ):
                    expected = (
                        f"{minimum} or more"
                        if maximum is None
                        else f"{minimum} to {maximum}"
                    )
                    raise SyntaxError(
                        f"Invalid function call: {node.func.id} requires "
                        f"{expected} arguments"
                    )

                return function(*(eval_expr(argument) for argument in node.args))

            if isinstance(node, ast.Compare):
                left = eval_expr(node.left)
                for operator_node, comparator_node in zip(node.ops, node.comparators):
                    comparator = _MATH_EXPRESSION_COMPARATORS.get(type(operator_node))
                    if comparator is None:
                        raise NotImplementedError(
                            f"Operator {type(operator_node).__name__} is not supported"
                        )
                    right = eval_expr(comparator_node)
                    if not comparator(left, right):
                        return 0
                    left = right
                return 1

            raise TypeError(f"Unsupported expression element: {type(node).__name__}")

        result = eval_expr(parsed)
        return {
            "ui": {"value": [result]},
            "result": (int(result), float(result)),
        }


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
    "CP SeedIncrementBatchGen": CP_SeedIncrementBatchGen,
    "CP VaeForwarding": CP_VaeForwarding,
    "CP ShowAnything": CP_ShowAnything,
    "MathExpression|pysssss": CP_MathExpression,
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
    "CP SeedIncrementBatchGen": "Seed Increment Batch Gen",
    "CP VaeForwarding": "VAE Forwarding",
    "CP ShowAnything": "Show Anything",
    "MathExpression|pysssss": "Math Expression",
}
