import logging
import os
from collections import namedtuple

import cv2
import folder_paths
import numpy as np
import torch
from PIL import Image
from nodes import MAX_RESOLUTION
from ultralytics import YOLO


categoryFolder = "Custom pack/Imported/Impact"


LOG = logging.getLogger(__name__)

SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


class NO_BBOX_DETECTOR:
    pass


class NO_SEGM_DETECTOR:
    pass


def _supported_pt_extensions():
    return getattr(folder_paths, "supported_pt_extensions", {".pt", ".pth", ".safetensors"})


def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    for full_folder_path in full_folder_paths:
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    paths, current_extensions = folder_paths.folder_names_and_paths.get(folder_name, ([], set()))
    all_paths = list(paths)
    for full_folder_path in full_folder_paths:
        if full_folder_path not in all_paths:
            all_paths.append(full_folder_path)

    folder_paths.folder_names_and_paths[folder_name] = (all_paths, set(current_extensions) | set(extensions))


def update_model_paths(model_path):
    if not model_path:
        return

    extensions = _supported_pt_extensions()
    add_folder_path_and_extensions(
        "ultralytics_bbox",
        [os.path.join(model_path, "ultralytics", "bbox")],
        extensions,
    )
    add_folder_path_and_extensions(
        "ultralytics_segm",
        [os.path.join(model_path, "ultralytics", "segm")],
        extensions,
    )
    add_folder_path_and_extensions(
        "ultralytics",
        [os.path.join(model_path, "ultralytics")],
        extensions,
    )


update_model_paths(getattr(folder_paths, "models_dir", None))
try:
    if "download_model_base" in folder_paths.folder_names_and_paths:
        update_model_paths(folder_paths.get_folder_paths("download_model_base")[0])
except Exception as error:
    LOG.debug("Skipping Impact ultralytics download_model_base path registration: %s", error)


def tensor2pil(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg, crop_region):
    x1, y1, x2, y2 = crop_region
    return npimg[y1:y2, x1:x2]


def crop_ndarray4(npimg, crop_region):
    x1, y1, x2, y2 = crop_region
    return npimg[:, y1:y2, x1:x2, :]


def crop_image(image, crop_region):
    return crop_ndarray4(image, crop_region)


def _as_numpy_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask)


def make_2d_mask(mask):
    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 4:
            return mask.squeeze(0).squeeze(0)
        if len(mask.shape) == 3:
            return mask.squeeze(0)
        return mask

    mask = np.asarray(mask)
    if len(mask.shape) == 4:
        return np.squeeze(mask, axis=(0, 1))
    if len(mask.shape) == 3:
        return np.squeeze(mask, axis=0)
    return mask


def make_3d_mask(mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(np.asarray(mask))

    if len(mask.shape) == 4:
        return mask.squeeze(0)
    if len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def dilate_mask(mask, dilation_factor, iterations=1):
    mask = _as_numpy_mask(make_2d_mask(mask)).astype(np.float32)
    if dilation_factor == 0:
        return mask

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)
    if dilation_factor > 0:
        return cv2.dilate(mask, kernel, iterations)
    return cv2.erode(mask, kernel, iterations)


def dilate_masks(segmasks, dilation_factor, iterations=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for bbox, mask, confidence in segmasks:
        cv2_mask = np.asarray(mask, dtype=np.float32)
        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iterations)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iterations)
        dilated_masks.append((bbox, dilated_mask, confidence))

    return dilated_masks


def combine_masks(masks):
    if len(masks) == 0:
        return None

    combined = (np.asarray(masks[0][1]) > 0).astype(np.uint8)
    for _, mask, _ in masks[1:]:
        cv2_mask = (np.asarray(mask) > 0).astype(np.uint8)
        if combined.shape == cv2_mask.shape:
            combined = cv2.bitwise_or(combined, cv2_mask)

    return torch.from_numpy(combined.astype(np.float32))


def combine_masks2(masks):
    if len(masks) == 0:
        return None

    combined = (_as_numpy_mask(masks[0]) > 0).astype(np.uint8)
    for mask in masks[1:]:
        cv2_mask = (_as_numpy_mask(mask) > 0).astype(np.uint8)
        if combined.shape == cv2_mask.shape:
            combined = cv2.bitwise_or(combined, cv2_mask)

    return torch.from_numpy(combined.astype(np.float32))


def create_segmasks(results):
    bboxes = results[1]
    segms = results[2]
    confidence = results[3]

    segmasks = []
    for i in range(len(segms)):
        segmasks.append((bboxes[i], np.asarray(segms[i], dtype=np.float32), confidence[i]))
    return segmasks


def load_yolo(model_path):
    return YOLO(model_path)


def inference_bbox(model, image, confidence=0.3, device=""):
    pred = model(image, conf=confidence, device=device)

    boxes = pred[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()
    if len(bboxes) == 0:
        return [[], [], [], []]

    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()
    else:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 1, -1)
        segms.append(cv2_mask.astype(np.float32))

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(boxes[i].conf.cpu().numpy())

    return results


def inference_segm(model, image, confidence=0.3, device=""):
    pred = model(image, conf=confidence, device=device)

    boxes = pred[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()
    if len(bboxes) == 0 or pred[0].masks is None:
        return [[], [], [], []]

    segms = pred[0].masks.data.cpu().numpy()
    h_segms = segms.shape[1]
    w_segms = segms.shape[2]
    h_orig = image.size[1]
    w_orig = image.size[0]
    ratio_segms = h_segms / w_segms
    ratio_orig = h_orig / w_orig

    if ratio_segms == ratio_orig:
        h_gap = 0
        w_gap = 0
    elif ratio_segms > ratio_orig:
        h_gap = int((ratio_segms - ratio_orig) * h_segms)
        w_gap = 0
    else:
        h_gap = 0
        ratio_segms = w_segms / h_segms
        ratio_orig = w_orig / h_orig
        w_gap = int((ratio_segms - ratio_orig) * w_segms)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        mask = mask[h_gap:mask.shape[0] - h_gap, w_gap:mask.shape[1] - w_gap]
        scaled_mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(image.size[1], image.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        results[2].append(scaled_mask.squeeze().squeeze().numpy())
        results[3].append(boxes[i].conf.cpu().numpy())

    return results


class UltraBBoxDetector:
    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation != 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for segmask, label in zip(segmasks, detected_results[0]):
            item_bbox, item_mask, confidence = segmask
            x1, y1, x2, y2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)
                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region).astype(np.float32)
                items.append(SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None))

        segs = (image.shape[1], image.shape[2]), items
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation != 0:
            segmasks = dilate_masks(segmasks, dilation)
        return combine_masks(segmasks)

    def setAux(self, _value):
        pass


class UltraSegmDetector:
    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation != 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for segmask, label in zip(segmasks, detected_results[0]):
            item_bbox, item_mask, confidence = segmask
            x1, y1, x2, y2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)
                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region).astype(np.float32)
                items.append(SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None))

        segs = (image.shape[1], image.shape[2]), items
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation != 0:
            segmasks = dilate_masks(segmasks, dilation)
        return combine_masks(segmasks)

    def setAux(self, _value):
        pass


def segs_to_combined_mask(segs):
    h, w = segs[0]
    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        cropped_mask = np.asarray(seg.cropped_mask)
        crop_region = seg.crop_region
        mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (
            (cropped_mask > 0).astype(np.uint8) * 255
        )

    return torch.from_numpy(mask.astype(np.float32) / 255.0)


def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)
    if mask is None:
        LOG.warning("[Impact] Cannot operate: MASK is empty.")
        return segs[0], []

    mask = (_as_numpy_mask(mask) * 255).astype(np.uint8)
    items = []

    for seg in segs[1]:
        cropped_mask = (np.asarray(seg.cropped_mask) * 255).astype(np.uint8)
        crop_region = seg.crop_region
        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask, cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0
        items.append(SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None))

    return segs[0], items


def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    mask = _as_numpy_mask(make_2d_mask(mask))

    points = []
    plabs = []
    y_step = max(3, int(mask.shape[0] / 20))
    x_step = max(3, int(mask.shape[1] / 20))

    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x + j, y + i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x + j, y + i))
                plabs.append(0)

    return points, plabs


def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []
    y_step = max(3, int(w / 20))
    x_step = max(3, int(h / 20))

    for i in range(10, h - 10, y_step):
        for j in range(10, w - 10, x_step):
            if not (x1 - 10 <= j <= x2 + 10 and y1 - 10 <= i <= y2 + 10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs


def make_sam_mask(
    sam,
    segs,
    image,
    detection_hint,
    dilation,
    threshold,
    bbox_expansion,
    mask_hint_threshold,
    mask_hint_use_negative,
):
    if hasattr(sam, "sam_wrapper"):
        sam_obj = sam.sam_wrapper
    elif all(hasattr(sam, attr) for attr in ("prepare_device", "release_device", "predict")):
        sam_obj = sam
    else:
        raise Exception("[Impact] Invalid SAM model is connected. Use a SAM_MODEL compatible with Impact Pack.")

    sam_obj.prepare_device()
    try:
        image_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        total_masks = []
        use_small_negative = mask_hint_use_negative == "Small"
        seg_items = segs[1]

        if detection_hint == "mask-points":
            points = []
            plabs = []
            for seg in seg_items:
                bbox = seg.bbox
                points.append(center_of_bbox(bbox))
                plabs.append(0 if use_small_negative and bbox[2] - bbox[0] < 10 else 1)
            total_masks += sam_obj.predict(image_np, points, plabs, None, threshold)
        else:
            for seg in seg_items:
                bbox = seg.bbox
                center = center_of_bbox(bbox)
                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image_np.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image_np.shape[0])
                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]
                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.extend([(x1 + gap, center[1]), (x1 + gap * 2, center[1])])
                    plabs = [1, 1]
                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.extend([(center[0], y1 + gap), (center[0], y1 + gap * 2)])
                    plabs = [1, 1]
                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.extend(
                        [
                            (x1 + x_gap, center[1]),
                            (x1 + x_gap * 2, center[1]),
                            (center[0], y1 + y_gap),
                            (center[0], y1 + y_gap * 2),
                        ]
                    )
                    plabs = [1, 1, 1, 1]
                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.extend(
                        [
                            (x1 + x_gap, y1 + y_gap),
                            (x1 + x_gap * 2, y1 + y_gap),
                            (x1 + x_gap, y1 + y_gap * 2),
                            (x1 + x_gap * 2, y1 + y_gap * 2),
                        ]
                    )
                    plabs = [1, 1, 1, 1]
                elif detection_hint == "mask-point-bbox":
                    points.append(center_of_bbox(seg.bbox))
                    plabs = [1]
                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(
                        seg.crop_region[0],
                        seg.crop_region[1],
                        seg.cropped_mask,
                        mask_hint_threshold,
                        use_small_negative,
                    )

                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(
                        image_np.shape[0],
                        image_np.shape[1],
                        seg.crop_region[0],
                        seg.crop_region[1],
                        seg.crop_region[2],
                        seg.crop_region[3],
                    )
                    points += npoints
                    plabs += nplabs

                total_masks += sam_obj.predict(image_np, points, plabs, dilated_bbox, threshold)

        mask = combine_masks2(total_masks)
    finally:
        sam_obj.release_device()

    if mask is not None:
        mask = mask.float()
        mask = torch.from_numpy(dilate_mask(mask.cpu().numpy(), dilation))
    else:
        mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

    return make_3d_mask(mask)


class CP_ImpactUltralyticsDetectorProvider:
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {"required": {"model_name": (bboxs + segms,)}}

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"
    CATEGORY = categoryFolder

    def doit(self, model_name):
        model_path = folder_paths.get_full_path("ultralytics", model_name)

        if model_path is None:
            if model_name.startswith("bbox/"):
                model_path = folder_paths.get_full_path("ultralytics_bbox", model_name[5:])
            elif model_name.startswith("segm/"):
                model_path = folder_paths.get_full_path("ultralytics_segm", model_name[5:])

        if model_path is None:
            cands = []
            cands.extend(folder_paths.get_folder_paths("ultralytics"))
            if model_name.startswith("bbox/"):
                cands.extend(folder_paths.get_folder_paths("ultralytics_bbox"))
            elif model_name.startswith("segm/"):
                cands.extend(folder_paths.get_folder_paths("ultralytics_segm"))

            formatted_cands = "\n\t".join(cands)
            LOG.error("[Impact] model file '%s' is not found in:\n\t%s", model_name, formatted_cands)
            raise ValueError(f"[Impact] model file '{model_name}' is not found.")

        model = load_yolo(model_path)

        if model_name.startswith("bbox/"):
            return (UltraBBoxDetector(model), NO_SEGM_DETECTOR())
        return (UltraBBoxDetector(model), UltraSegmDetector(model))


class CP_ImpactFaceDetectorOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "bbox_detector": ("BBOX_DETECTOR",),
            },
            "optional": {
                "sam_model_opt": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "detailer_hook": ("DETAILER_HOOK",),
                "sam_detection_hint": (
                    [
                        "center-1",
                        "horizontal-2",
                        "vertical-2",
                        "rect-4",
                        "diamond-4",
                        "mask-area",
                        "mask-points",
                        "mask-point-bbox",
                        "none",
                    ],
                ),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask")
    OUTPUT_IS_LIST = (False, True, True, False)
    FUNCTION = "doit"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Detects faces or other objects using Impact-compatible bbox/segm/SAM detectors and "
        "returns the original image with the combined detection mask."
    )

    @staticmethod
    def enhance_face(
        image,
        bbox_threshold,
        bbox_dilation,
        bbox_crop_factor,
        drop_size,
        bbox_detector,
        segm_detector=None,
        sam_model_opt=None,
        detailer_hook=None,
        sam_detection_hint=None,
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative="False",
    ):
        if hasattr(bbox_detector, "setAux"):
            bbox_detector.setAux("face")
        segs = bbox_detector.detect(
            image,
            bbox_threshold,
            bbox_dilation,
            bbox_crop_factor,
            drop_size,
            detailer_hook=detailer_hook,
        )
        if hasattr(bbox_detector, "setAux"):
            bbox_detector.setAux(None)

        if sam_model_opt is not None:
            sam_mask = make_sam_mask(
                sam_model_opt,
                segs,
                image,
                sam_detection_hint,
                sam_dilation,
                sam_threshold,
                sam_bbox_expansion,
                sam_mask_hint_threshold,
                sam_mask_hint_use_negative,
            )
            segs = segs_bitwise_and_mask(segs, sam_mask)
        elif (
            segm_detector is not None
            and not isinstance(segm_detector, NO_SEGM_DETECTOR)
            and hasattr(segm_detector, "detect")
        ):
            segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

            if (
                hasattr(segm_detector, "override_bbox_by_segm")
                and segm_detector.override_bbox_by_segm
                and not (detailer_hook is not None and not hasattr(detailer_hook, "override_bbox_by_segm"))
            ):
                segs = segm_segs
            else:
                segm_mask = segs_to_combined_mask(segm_segs)
                segs = segs_bitwise_and_mask(segs, segm_mask)

        enhanced_img = image
        cropped_enhanced = []
        cropped_enhanced_alpha = []
        mask = segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask

    def doit(
        self,
        image,
        bbox_threshold,
        bbox_dilation,
        bbox_crop_factor,
        drop_size,
        bbox_detector,
        sam_model_opt=None,
        segm_detector_opt=None,
        detailer_hook=None,
        sam_detection_hint=None,
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative="False",
    ):
        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []

        if len(image) > 1:
            LOG.warning(
                "[Impact] FaceDetectorOnly is not designed for video detailing. "
                "For video workflows, use a detector intended for animated batches."
            )

        for single_image in image:
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask = self.enhance_face(
                single_image.unsqueeze(0),
                bbox_threshold,
                bbox_dilation,
                bbox_crop_factor,
                drop_size,
                bbox_detector,
                segm_detector_opt,
                sam_model_opt,
                detailer_hook,
                sam_detection_hint,
                sam_dilation,
                sam_threshold,
                sam_bbox_expansion,
                sam_mask_hint_threshold,
                sam_mask_hint_use_negative,
            )

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)

        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask


NODE_CLASS_MAPPINGS_IMPORTED_IMPACT_ULTRALYTICS = {
    "UltralyticsDetectorProvider": CP_ImpactUltralyticsDetectorProvider,
    "FaceDetectorOnly": CP_ImpactFaceDetectorOnly,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_IMPACT_ULTRALYTICS = {
    "UltralyticsDetectorProvider": "UltralyticsDetectorProvider",
    "FaceDetectorOnly": "Face Detector Only",
}
