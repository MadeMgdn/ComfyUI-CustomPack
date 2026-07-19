import torch
import numpy as np

categoryFolder = "Custom pack/Mask"

#Analyzes one large mask and produces up to 4 masks in descending order of the space they occupy on the image
class CP_MaskSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3", "mask_4")
    FUNCTION = "separate_regions"
    CATEGORY = categoryFolder

    def separate_regions(self, mask):
        if len(mask.shape) == 4:
            mask = mask[0]
        mask_np = mask.cpu().numpy().squeeze()
        binary = (mask_np > 0).astype(np.uint8)

        height, width = binary.shape

        parent = [0]
        labels = np.zeros_like(binary, dtype=np.int32)
        next_label = 0

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for y in range(height):
            for x in range(width):
                if binary[y, x] == 0:
                    continue

                neighbor_labels = set()
                if y > 0 and x > 0 and labels[y-1, x-1] > 0:
                    neighbor_labels.add(labels[y-1, x-1])
                if y > 0 and labels[y-1, x] > 0:
                    neighbor_labels.add(labels[y-1, x])
                if y > 0 and x < width-1 and labels[y-1, x+1] > 0:
                    neighbor_labels.add(labels[y-1, x+1])
                if x > 0 and labels[y, x-1] > 0:
                    neighbor_labels.add(labels[y, x-1])

                if not neighbor_labels:
                    next_label += 1
                    parent.append(next_label)
                    labels[y, x] = next_label
                else:
                    min_label = min(neighbor_labels)
                    labels[y, x] = min_label
                    for lbl in neighbor_labels:
                        if lbl != min_label:
                            union(min_label, lbl)

        for y in range(height):
            for x in range(width):
                if labels[y, x] > 0:
                    labels[y, x] = find(labels[y, x])

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        areas = {}
        if len(unique_labels) > 0:
            counts = np.bincount(labels.ravel())
            for lbl in unique_labels:
                areas[lbl] = counts[lbl]

        sorted_labels = sorted(unique_labels, key=lambda lbl: areas[lbl], reverse=True)

        output_masks = []
        for i in range(4):
            if i < len(sorted_labels):
                region_mask = (labels == sorted_labels[i]).astype(np.float32)
            elif i == 3 and len(sorted_labels) > 4:
                region_mask = np.zeros_like(binary, dtype=np.float32)
                for lbl in sorted_labels[3:]:
                    region_mask[labels == lbl] = 1.0
            else:
                region_mask = np.zeros_like(binary, dtype=np.float32)

            region_tensor = torch.from_numpy(region_mask).unsqueeze(0).float()
            output_masks.append(region_tensor)

        while len(output_masks) < 4:
            output_masks.append(torch.zeros_like(mask))

        return tuple(output_masks[:4])

#Checks the mask - if it is not empty, then True, otherwise False
class CP_MaskNotEmpty:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "check_mask"
    CATEGORY = categoryFolder

    def check_mask(self, mask):
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
        is_not_empty = np.any(mask_np > 0)
        
        return (is_not_empty,)

#Trims the mask as needed and produces the final mask and its dimensions
class CP_MaskCropWithPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "INT", "INT")
    RETURN_NAMES = ("cropped_mask", "width", "height")
    FUNCTION = "crop_mask"
    CATEGORY = categoryFolder

    def crop_mask(self, mask, padding):
        mask_np = mask.squeeze(0).cpu().numpy()
        
        non_zero = np.where(mask_np > 0)
        
        if len(non_zero[0]) == 0:
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return (empty_mask, 1, 1)
        
        min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
        
        h, w = mask_np.shape
        min_y = max(0, min_y - padding)
        max_y = min(h - 1, max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(w - 1, max_x + padding)
        
        crop_width = max_x - min_x + 1
        crop_height = max_y - min_y + 1
        
        cropped = mask_np[min_y:max_y+1, min_x:max_x+1]
        cropped_tensor = torch.from_numpy(cropped).unsqueeze(0).float()
        
        return (cropped_tensor, int(crop_width), int(crop_height))

#Returns the number of steps based on the mask. If the mask is not empty, the number of steps is based on on_true; otherwise, it is based on on_false.
class CP_StepsByEmptyMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "on_true": ("INT", {"default": 1, "min": 0, "max": 100}),
                "on_false": ("INT", {"default": 0, "min": 0, "max": 150}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "evaluate"
    CATEGORY = categoryFolder

    def evaluate(self, mask, on_true, on_false):
        if torch.any(mask):
            return (on_true,)
        else:
            return (on_false,)

#Darkens the mask at the edges
class CP_MaskBlacker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "display": "number"}),
                "left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "display": "number"}),
                "right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "display": "number"}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "display": "number"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blacken_mask"
    CATEGORY = categoryFolder
    
    def blacken_mask(self, mask, top, left, right, bottom):
        result = mask.clone()
        if len(result.shape) == 4:
            _, _, h, w = result.shape
            batch_dim = True
        elif len(result.shape) == 3:
            _, h, w = result.shape
            batch_dim = False
        else:
            h, w = result.shape
            batch_dim = False
        top = min(top, h)
        bottom = min(bottom, h)
        left = min(left, w)
        right = min(right, w)
        
        if batch_dim:
            if top > 0:
                result[..., :top, :] = 0
            if bottom > 0:
                result[..., -bottom:, :] = 0
            if left > 0:
                result[..., :, :left] = 0
            if right > 0:
                result[..., :, -right:] = 0
        else:
            if len(result.shape) == 3:
                if top > 0:
                    result[:, :top, :] = 0
                if bottom > 0:
                    result[:, -bottom:, :] = 0
                if left > 0:
                    result[:, :, :left] = 0
                if right > 0:
                    result[:, :, -right:] = 0
            else:
                if top > 0:
                    result[:top, :] = 0
                if bottom > 0:
                    result[-bottom:, :] = 0
                if left > 0:
                    result[:, :left] = 0
                if right > 0:
                    result[:, -right:] = 0
        return (result,)

NODE_CLASS_MAPPINGS_MASK = {
    "CP MaskCropWithPadding": CP_MaskCropWithPadding,
    "CP MaskNotEmpty" : CP_MaskNotEmpty,
    "CP MaskRegionSeparator": CP_MaskSplitter,
    "CP StepsByEmptyMask" : CP_StepsByEmptyMask,
    "CP_MaskBlacker": CP_MaskBlacker,
}

NODE_DISPLAY_NAME_MAPPINGS_MASK = {
    "CP MaskCropWithPadding": "Mask crop with padding",
    "CP MaskNotEmpty": "Mask not empty",
    "CP StepsByEmptyMask": "Steps by empty mask",
    "CP MaskRegionSeparator": "Mask region separator",
    "CP_MaskBlacker": "Mask blacker",
}