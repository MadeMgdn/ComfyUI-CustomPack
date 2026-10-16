# Divide and Conquer code by Steudio.
# Utility nodes include adaptations credited in the original ComfyUI_Steudio sources.

import math
import os

import comfy.samplers
import comfy.utils
import numpy as np
import torch
from comfy import model_management
from PIL import Image, ImageDraw, ImageFilter, ImageOps


categoryFolder = "Custom pack/Imported/Steudio"


OVERLAP_DICT = {
    "None": 0,
    "1/64 Tile": 0.015625,
    "1/32 Tile": 0.03125,
    "1/16 Tile": 0.0625,
    "1/8 Tile": 0.125,
    "1/4 Tile": 0.25,
    "1/2 Tile": 0.5,
}

TILE_ORDER_DICT = {
    "linear": 0,
    "spiral": 1
}

SCALING_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos"
]

MIN_SCALE_FACTOR_THRESHOLD = 1.0

def calculate_overlap(tile_size, overlap_fraction):
    return int(overlap_fraction * tile_size)

def create_tile_coordinates(image_width, image_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order):
    tiles = []
    num_columns = grid_x
    num_rows = grid_y
    matrix = [['' for _ in range(num_columns)] for _ in range(num_rows)]

    # Generate tiles in grid layout
    for row in range(grid_y):
        y = row * (tile_height - overlap_y)
        if row == grid_y - 1:
            y = image_height - tile_height
        for col in range(grid_x):
            x = col * (tile_width - overlap_x)
            if col == grid_x - 1:
                x = image_width - tile_width
            tiles.append((x, y))
    
    if tile_order == 1:  # Spiral order
        # Rearrange tiles in an outward clockwise spiral pattern starting from the center
        spiral_tiles = []
        visited = set()
        x, y = num_columns // 2, num_rows // 2
        dx, dy = 1, 0  # Start moving right
        layer = 1

        while len(spiral_tiles) < len(tiles):
            for _ in range(2):
                for _ in range(layer):
                    if 0 <= x < num_columns and 0 <= y < num_rows and (x, y) not in visited:
                        index = y * num_columns + x
                        if index < len(tiles):
                            spiral_tiles.append(tiles[index])
                            visited.add((x, y))
                    x += dx
                    y += dy
                dx, dy = -dy, dx  # Rotate direction clockwise
            layer += 1

        spiral_tiles.reverse()
        tiles = spiral_tiles

    # Rebuild matrix to match tile order
    for i, (x, y) in enumerate(tiles):
        row, col = y // (tile_height - overlap_y), x // (tile_width - overlap_x)
        matrix[row][col] = f"{i + 1} ({x},{y})"

    return tiles, matrix


class DaC_Algorithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024,}),
                "tile_height": ("INT", {"default": 1024,}),
                "min_overlap": (list(OVERLAP_DICT.keys()), {"default": "1/32 Tile",}),
                "min_scale_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0}),
                "tile_order": (list(TILE_ORDER_DICT.keys()), {"default": "spiral",}),
                "scaling_method": (SCALING_METHODS, {"default": "lanczos"}),  
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),  # Now optional
                "use_upscale_with_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "DAC_DATA", "STRING")
    RETURN_NAMES = ("IMAGE", "dac_data", "ui")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = categoryFolder
    DESCRIPTION = """
Calculate the best dimensions and optionally upscale an image
while maintaining minimum tile overlap and scale factor constraints.
Steudio
"""

    def execute(self, image, scaling_method, tile_width, tile_height, min_overlap, min_scale_factor, tile_order, upscale_model=None, use_upscale_with_model=True):

        overlap = OVERLAP_DICT.get(min_overlap, 0)  # Default to 0 if the key is not found
        tile_order = TILE_ORDER_DICT.get(tile_order, 0)  # Default to 0 if the key is not found

        _, height, width, _ = image.shape

        # Calculate initial overlaps
        overlap_x = calculate_overlap(tile_width, overlap)
        overlap_y = calculate_overlap(tile_height, overlap)

        # Ensure min_scale_factor is at least 1.01 to avoid divide by zero error
        min_scale_factor = max(min_scale_factor, MIN_SCALE_FACTOR_THRESHOLD)

        if width <= height:
            # Calculate initial upscaled width based on min_scale_factor
            multiply_factor = math.ceil(min_scale_factor * width / tile_width)
            while True:
                upscaled_width = tile_width * multiply_factor
                grid_x = math.ceil(upscaled_width / tile_width)
                upscaled_width = (tile_width * grid_x) - (overlap_x * (grid_x - 1))
                upscale_ratio = upscaled_width / width
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            upscaled_height = int(height * upscale_ratio)
            grid_y = math.ceil((upscaled_height - overlap_y) / (tile_height - overlap_y))
            overlap_y = round((tile_height * grid_y - upscaled_height) / (grid_y - 1))
        else:
            multiply_factor = math.ceil(min_scale_factor * height / tile_height)
            while True:
                upscaled_height = tile_height * multiply_factor
                grid_y = math.ceil(upscaled_height / tile_height)
                upscaled_height = (tile_height * grid_y) - (overlap_y * (grid_y - 1))
                upscale_ratio = upscaled_height / height
                if upscale_ratio >= min_scale_factor:
                    break
                multiply_factor += 1
            upscaled_width = int(width * upscale_ratio)
            grid_x = math.ceil((upscaled_width - overlap_x) / (tile_width - overlap_x))
            overlap_x = round((tile_width * grid_x - upscaled_width) / (grid_x - 1))

        effective_upscale = round(upscaled_width / width, 2)
        upscaled_image_size = f"{upscaled_width}x{upscaled_height}"
        original_image = f"{width}x{height}"
        grid_n_xy = f"{grid_x}x{grid_y}"
        Tiles_Q = grid_x * grid_y

        dac_data = {
            'upscaled_width': upscaled_width,
            'upscaled_height': upscaled_height,
            'tile_width': tile_width,
            'tile_height': tile_height,
            'overlap_x': overlap_x,
            'overlap_y': overlap_y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'tile_order': tile_order,
        }

        if use_upscale_with_model and upscale_model:
            # Upscale the image with model
            device = model_management.get_torch_device()
            memory_required = model_management.module_size(upscale_model.model)
            memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
            memory_required += image.nelement() * image.element_size()
            model_management.free_memory(memory_required, device)

            upscale_model.to(device)
            in_img = image.movedim(-1, -3).to(device)

            tile = 512
            overlap_value = 32

            oom = True
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap_value)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap_value, upscale_amount=upscale_model.scale, pbar=pbar)
                    oom = False
                except model_management.OOM_EXCEPTION as e:
                    tile //= 2
                    if tile < 128:
                        raise e

            upscale_model.to("cpu")
            Upscaled_with_Model = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

            samples = Upscaled_with_Model.movedim(-1, 1)
        else:
            samples = image.movedim(-1, 1)  # Use original image

        if upscaled_width == 0:
            upscaled_width = max(1, round(samples.shape[3] * upscaled_height / samples.shape[2]))
        elif upscaled_height == 0:
            upscaled_height = max(1, round(samples.shape[2] * upscaled_width / samples.shape[3]))

        upscaled_image = comfy.utils.common_upscale(samples, upscaled_width, upscaled_height, scaling_method, crop=0).movedim(1, -1)

        algo_ui = f"Divide and Conquer Algorithm:\nOriginal Image Size: {original_image}\nUpscaled Image Size: {upscaled_image_size}\nGrid: {grid_n_xy} ({Tiles_Q} tiles)\nOverlap_x: {overlap_x} pixels\nOverlap_y: {overlap_y} pixels\nEffective_upscale: {effective_upscale}"

        return (upscaled_image, dac_data, algo_ui)


class Divide_Image_Select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
                "tile": ("INT", { "default": 0, "min": 0, "step": 1, }),
            },
        }


    RETURN_TYPES = ("IMAGE", "UI")
    RETURN_NAMES = ("TILE(S)", "ui",)
    OUTPUT_IS_LIST = (True,False)
    FUNCTION = "execute"
    CATEGORY = categoryFolder
    DESCRIPTION = """
tile 0 = All tiles
tile # = Tile #
"""

    def execute(self, image, tile, dac_data,):
        # # Ensure `ui` is not a list
        # if isinstance(ui, list):
        #     ui = ui[0]

        image_height = image.shape[1]
        image_width = image.shape[2]


        tile_width = dac_data['tile_width']
        tile_height = dac_data['tile_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        tile_coordinates, matrix = create_tile_coordinates(
            image_width, image_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        iteration = 1

        image_tiles = []
        for tile_coordinate in tile_coordinates:
            iteration += 1

            image_tile = image[
                :,
                tile_coordinate[1] : tile_coordinate[1] + tile_height,
                tile_coordinate[0] : tile_coordinate[0] + tile_width,
                :,
            ]

            image_tiles.append(image_tile)

        all_tiles = torch.cat(image_tiles, dim=0)
        selected_tile = image_tiles[tile - 1]

        if tile == 0:
            tile_or_tiles = all_tiles
        else:
            tile_or_tiles = selected_tile

        matrix_ui = "Divide and Conquer Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])


        return ([tile_or_tiles[i].unsqueeze(0) for i in range(tile_or_tiles.shape[0])], matrix_ui)


class Combine_Tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "dac_data": ("DAC_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "UI")
    RETURN_NAMES = ("image", "ui",) 
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = categoryFolder

    def execute(self, images, dac_data):

        # Ensure `dac_data` is not a list
        if isinstance(dac_data, list):
            dac_data = dac_data[0]

        # Combine images into a single tensor
        out = []
        for i in range(len(images)):
            img = images[i]
            out.append(img)
        images = torch.stack(out).squeeze(1)
        
        overlap_factor = 4
        # blur_factor = 20

        # Import from dac_data
        upscaled_width = dac_data['upscaled_width']
        upscaled_height = dac_data['upscaled_height']
        overlap_x = dac_data['overlap_x']
        overlap_y = dac_data['overlap_y']
        grid_x = dac_data['grid_x']
        grid_y = dac_data['grid_y']
        tile_order = dac_data['tile_order']

        # Import from Images
        tile_width = images.shape[2]
        tile_height = images.shape[1]

        # Overlap / factor
        f_overlap_x = overlap_x //overlap_factor
        f_overlap_y = overlap_y //overlap_factor

        # Blend factor
        blend_x = math.sqrt(overlap_x)
        blend_y = math.sqrt(overlap_y)


        tile_coordinates, matrix = create_tile_coordinates(
            upscaled_width, upscaled_height, tile_width, tile_height, overlap_x, overlap_y, grid_x, grid_y, tile_order
        )

        original_shape = (1, upscaled_height, upscaled_width, 3)
        output = torch.zeros(original_shape, dtype=images.dtype)

        index = 0
        iteration = 1
        for tile_coordinate in tile_coordinates:
            image_tile = images[index]
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            iteration += 1

            # Create mask for the tile
            mask = Image.new("L", (tile_width, tile_height), 0)
            draw = ImageDraw.Draw(mask)

            # Do not apply gaussian to tile at the edge of the image
            # 1234 Detect corners top/left top/right bottom/left bottom/right and grid >1
            if x == 0 and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width, tile_height - f_overlap_y], fill=255)
            elif x == 0 and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            elif x == upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height], fill=255)
            # 5678 Detect corners 3 edges and grid =1
            elif x == 0 and y == 0 and upscaled_height == tile_height:
                draw.rectangle([x, y, tile_width - f_overlap_x, tile_height], fill=255)
            elif x == upscaled_width - tile_width and y == 0 and upscaled_height == tile_height:
                draw.rectangle([f_overlap_x, y, tile_width, tile_height], fill=255)
            elif x == 0 and y == 0 and upscaled_width == tile_width:
                draw.rectangle([x, y, tile_width, tile_height - f_overlap_y], fill=255)
            elif x == 0 and y == upscaled_height - tile_height and upscaled_width == tile_width:
                draw.rectangle([x, f_overlap_y, tile_width, tile_height], fill=255)
            # 9 12 Detect top or bottom edges
            elif x != 0 and x !=upscaled_width - tile_width and y == 0 and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x != 0 and x !=upscaled_width - tile_width and y == upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height], fill=255)
            # 10 11 Detect left or right edges
            elif x == 0 and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)
            elif x == upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            # 13 Detect top and bottom edges
            elif x != 0 and x !=upscaled_width - tile_width and y == 0 and upscaled_height == tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, y, tile_width - f_overlap_x, tile_height], fill=255)
            # 14 Detect left and right edges
            elif x == 0 and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width == tile_width:
                draw.rectangle([x, f_overlap_y, tile_width, tile_height - f_overlap_y], fill=255)
            # 15 Detect not touching any edges
            elif x != 0 and x !=upscaled_width - tile_width and y !=0 and y != upscaled_height - tile_height and upscaled_height != tile_height and upscaled_width != tile_width:
                draw.rectangle([f_overlap_x, f_overlap_y, tile_width - f_overlap_x, tile_height - f_overlap_y], fill=255)

            # Use a box blur if overlap is getting too narrow
            if overlap_x <= 64 or overlap_y <= 64:
                mask = mask.filter(ImageFilter.BoxBlur(radius=(blend_x, blend_y)))
            else:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=(blend_x, blend_y)))

            mask_np = np.array(mask) / 255.0
            mask_tensor = torch.tensor(mask_np, dtype=images.dtype).unsqueeze(0).unsqueeze(-1)

            output[:, y : y + tile_height, x : x + tile_width, :] *= (1 - mask_tensor)
            output[:, y : y + tile_height, x : x + tile_width, :] += image_tile * mask_tensor

            index += 1

        matrix_ui = "Divide and Conquer Matrix:\n" + '\n'.join([' '.join(row) for row in matrix])


        return output, matrix_ui


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

_any_ = AnyType("*")

RATIO = {
    "1:1 ◻": (1, 1),
    "5:4 ▭": (5, 4),
    "4:3 ▭": (4, 3),
    "3:2 ▭": (3, 2),
    "16:9 ▭": (16, 9),
    "2:1 ▭": (2, 1),
    "21:9 ▭": (21, 9),
    "32:9 ▭": (32, 9),
    "": (1, 1),
    "4:5 ▯": (4, 5),
    "3:4 ▯": (3, 4),
    "2:3 ▯": (2, 3),
    "9:16 ▯": (9, 16),
    "1:2 ▯": (1, 2),
    "9:21 ▯": (9, 21),
    "9:32 ▯": (9, 32),
}

class Ratio_Calculator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = (_any_,)
    RETURN_NAMES = ("ratio",)
    FUNCTION = "calc"
    OUTPUT_NODE = True
    CATEGORY = categoryFolder

    def calc(self, image):      

        # Get dimensions of the image
        _, height, width, _ = image.shape

        # Find the greatest common divisor (GCD)
        gcd = math.gcd(width, height)
        
        # Simplify the dimensions
        simplified_width = width // gcd
        simplified_height = height // gcd

        # Calculate megapixel
        f_megapixel = "{:,} pixels".format(width * height)

        # Find the closest ratio
        closest_ratio = None
        min_difference = float('inf')
        for name, (rw, rh) in RATIO.items():
            difference = abs(simplified_width / simplified_height - rw / rh)
            if difference < min_difference:
                min_difference = difference
                closest_ratio = name

        # return closest_ratio,
        return {"ui": {"text": f"{closest_ratio}\n{f_megapixel}"},"result": (closest_ratio,)}

class Ratio_to_Size:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (list(RATIO.keys()),),
                "Megapixel": ("FLOAT", {"default": 1.05, "min": 0.10, "max": 3.00, "step": 0.01 }),
                "Precision": ("FLOAT", {"default": 0.30, "min": 0.00, "max": 1.00, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "UI",)
    RETURN_NAMES = ("width", "height", "ui",)
    FUNCTION = "calculate_dimensions"
    CATEGORY = categoryFolder

    def calculate_dimensions(self, ratio, Megapixel, Precision):

        # Retrieve aspect width and height from the resolutions dictionary
        aspect_width, aspect_height = RATIO.get(ratio, (1, 1))  # Default to (1, 1) if ratio not found

        # Convert megapixels to total pixels
        total_pixels = int(Megapixel * 1_000_000)

        # Calculate approximate starting dimensions
        width = int((total_pixels * (aspect_width / aspect_height)) ** 0.5)
        height = int(width * (aspect_height / aspect_width))

        # Adjust width and height to multiples of 64 while respecting precision
        while True:
            # Ensure dimensions are multiples of 64
            width = (width // 64) * 64
            height = (height // 64) * 64

            # Check precision
            if abs((width / height) - (aspect_width / aspect_height)) <= Precision:
                break

            # Try reducing dimensions
            if width > 64 and height > 64:
                if (width / height) > (aspect_width / aspect_height):
                    width -= 64
                else:
                    height -= 64
            else:
                break

        f_megapixel = "{:,}".format(width * height) 
        f_precision = round((aspect_width / aspect_height) - (width / height), 2)

        ui = f"Ratio: {ratio}\nWidth: {width}\nHeight: {height}\nMegapixel: {f_megapixel}\nPrecision: {f_precision}\n"


        return int(width), int(height), ui


class Seed_Shifter:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1 }),
                "seed_shifter": ("INT", {"default": 0, "min": 0}),
                "batch": ("INT", {"default": 1, "min": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seeds",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "shift_seeds"
    CATEGORY = categoryFolder
    DESCRIPTION = """
A simple and effective way to generate a “batch” of images with reproducible seed.
Steudio
"""
    def shift_seeds(self, seed_, seed_shifter, batch):
        seeds = [(seed_ + seed_shifter + i) for i in range(batch)]
        return seeds,

class Sequence_Generator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "0...1+0.1"}),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", )
    OUTPUT_IS_LIST = (True,True)
    OUTPUT_NODE = True
    FUNCTION = "Execute"
    CATEGORY = categoryFolder
    DESCRIPTION = """
x...y+z | Generates a sequence of numbers from x to y with a step of z.
x...y#z | Generates z evenly spaced numbers between x and y.
  x,y,z | Generates a list of x, y, z.
    """

    def Execute(self, gen):
        elements = gen.split(',')
        result = []

        def parse_number(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        for element in elements:
            element = element.strip()

            if '...' in element:
                if '#' in element:
                    start, rest = element.split('...')
                    end, num_items = rest.split('#')
                    start = parse_number(start)
                    end = parse_number(end)
                    num_items = int(parse_number(num_items))
                    if num_items == 1:
                        result.append(round(start, 2))
                    else:
                        step = (end - start) / (num_items - 1)
                        for i in range(num_items):
                            result.append(round(start + i * step, 2))
                else:
                    start, rest = element.split('...')
                    end, step = rest.split('+')
                    start = parse_number(start)
                    end = parse_number(end)
                    step = abs(parse_number(step))
                    current = start
                    if start > end:
                        step = -step
                    while (step > 0 and current <= end) or (step < 0 and current >= end):
                        result.append(round(current, 2))
                        current += step
            else:
                result.append(round(parse_number(element), 2))

        seq_int = list(map(int, result))
        seq_float = list(map(float, [f"{num:.2f}"for num in result if isinstance(num, float)]))
        seq_int_float = f"{len(seq_int)} INT: {seq_int}\n{len(seq_float)} FLOAT: {seq_float}"

        return {"ui": {"text": (seq_int_float)}, "result": (seq_int, seq_float)}


class Simple_Config:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):               
        return {
            "required": {
                "steps": ("INT", {"default": 24, "min": 1, "max": 99}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            }
        }
    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("STEPS", "SAMPLER", "SCHEDULER")
    

    FUNCTION = "config"
    CATEGORY = categoryFolder

    def config(self, steps, sampler, scheduler,):           
        return(steps, sampler, scheduler,)
    


# Original display_any code from https://github.com/rgthree/rgthree-comfy
class Display_UI:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name, missing-function-docstring
        return {
            "required": {
                "ui": (_any_, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = categoryFolder

    def main(self, ui=None):
        value = 'None'
        if isinstance(ui, str):
            value = ui
        elif isinstance(ui, (int, float, bool)):
            value = str(ui)

        return {"ui": {"text": (value,)}}


# Original LoadImagesFromFolderKJ code from https://github.com/kijai/ComfyUI-KJNodes
class Load_Images_into_List:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "Load_Images_into_List"
    CATEGORY = categoryFolder

    def Load_Images_into_List(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        # List all files in the directory
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        if not dir_files:
            raise FileNotFoundError(f"No valid image files found in directory '{directory}'.")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        images = []

        for image_path in dir_files:
            try:
                i = Image.open(image_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        if not images:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        # Concatenate images into a single tensor
        images = torch.cat(images, dim=0)
        return ([images[i].unsqueeze(0) for i in range(images.shape[0])],)


NODE_CLASS_MAPPINGS_IMPORTED_STEUDIO_STEUDIO = {
    "Divide and Conquer Algorithm": DaC_Algorithm,
    "Combine Tiles": Combine_Tiles,
    "Divide Image and Select Tile": Divide_Image_Select,
    "Ratio Calculator": Ratio_Calculator,
    "Ratio to Size": Ratio_to_Size,
    "Seed Shifter": Seed_Shifter,
    "Sequence Generator": Sequence_Generator,
    "Load Images into List": Load_Images_into_List,
    "Simple Config": Simple_Config,
    "Display UI": Display_UI,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_STEUDIO_STEUDIO = {
    "Divide and Conquer Algorithm": "Divide and Conquer Algorithm",
    "Combine Tiles": "Combine Tiles",
    "Divide Image and Select Tile": "Divide Image and Select Tile",
    "Ratio Calculator": "Ratio Calculator",
    "Ratio to Size": "Ratio to Size",
    "Seed Shifter": "Seed Shifter",
    "Sequence Generator": "Sequence Generator",
    "Load Images into List": "Load Images into List",
    "Simple Config": "Simple Config",
    "Display UI": "Display UI",
}
