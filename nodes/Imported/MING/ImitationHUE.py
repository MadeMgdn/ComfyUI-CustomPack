import numpy as np
import torch
from PIL import Image, ImageFilter

categoryFolder = "Custom pack/Imported/Imitation hue"

def bgr_to_hsv(bgr):
    bgr = bgr.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    h = np.zeros_like(maxc)
    mask = maxc != minc
    diff = maxc - minc
    r_mask = (maxc == r) & mask
    g_mask = (maxc == g) & mask
    b_mask = (maxc == b) & mask
    h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask])) % 360
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360
    s = np.zeros_like(maxc)
    s[maxc != 0] = diff[maxc != 0] / maxc[maxc != 0]
    v = maxc
    h = (h / 2).astype(np.uint8)   
    s = (s * 255).astype(np.uint8) 
    v = (v * 255).astype(np.uint8) 
    return np.stack([h, s, v], axis=-1)


def hsv_to_bgr(hsv):
    h = hsv[..., 0].astype(np.float32) * 2.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    hi = np.floor(h / 60) % 6
    f = h / 60 - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    mask0 = hi == 0
    mask1 = hi == 1
    mask2 = hi == 2
    mask3 = hi == 3
    mask4 = hi == 4
    mask5 = hi == 5
    r[mask0] = v[mask0]; g[mask0] = t[mask0]; b[mask0] = p[mask0]
    r[mask1] = q[mask1]; g[mask1] = v[mask1]; b[mask1] = p[mask1]
    r[mask2] = p[mask2]; g[mask2] = v[mask2]; b[mask2] = t[mask2]
    r[mask3] = p[mask3]; g[mask3] = q[mask3]; b[mask3] = v[mask3]
    r[mask4] = t[mask4]; g[mask4] = p[mask4]; b[mask4] = v[mask4]
    r[mask5] = v[mask5]; g[mask5] = p[mask5]; b[mask5] = q[mask5]
    r = (r * 255).astype(np.uint8)
    g = (g * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def bgr_to_lab(bgr):
    bgr = bgr.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    X = 0.412453 * r + 0.357580 * g + 0.180423 * b
    Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    X /= 0.950456
    Y /= 1.0
    Z /= 1.088754
    eps = 0.008856
    kappa = 903.3
    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116.0)
    fx = f(X)
    fy = f(Y)
    fz = f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)
    L = L * 255.0 / 100.0
    a = a + 128
    b_lab = b_lab + 128
    L = np.clip(L, 0, 255)
    a = np.clip(a, 0, 255)
    b_lab = np.clip(b_lab, 0, 255)
    return np.stack([L, a, b_lab], axis=-1).astype(np.uint8)


def lab_to_bgr(lab):
    lab = lab.astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128
    b_lab = lab[..., 2] - 128
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b_lab / 200
    eps = 0.008856
    def invf(t):
        return np.where(t > 6/29, t**3, (t - 16/116) / 7.787)
    X = invf(fx) * 0.950456
    Y = invf(fy) * 1.0
    Z = invf(fz) * 1.088754
    r = 3.240479 * X - 1.537150 * Y - 0.498535 * Z
    g = -0.969256 * X + 1.875992 * Y + 0.041556 * Z
    b = 0.055648 * X - 0.204043 * Y + 1.057311 * Z
    r = np.clip(r, 0, 1) * 255
    g = np.clip(g, 0, 1) * 255
    b = np.clip(b, 0, 1) * 255
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def bgr_to_gray(bgr):
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def rgb_to_bgr(rgb):
    return rgb[..., ::-1]


def bgr_to_rgb(bgr):
    return bgr[..., ::-1]

def pil_resize(img_np, size, interpolation=Image.BILINEAR):
    img = Image.fromarray(img_np)
    img_resized = img.resize(size, interpolation)
    return np.array(img_resized)


def pil_gaussian_blur(img_np, kernel_size):
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    img = Image.fromarray(img_np)
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(img_blur)

def clahe(img, clip_limit=2.5, grid_size=(8, 8)):
    h, w = img.shape
    nh, nw = grid_size
    tile_h = h // nh
    tile_w = w // nw

    pad_h = tile_h * nh - h
    pad_w = tile_w * nw - w
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        h, w = img.shape

    clip_limit_abs = clip_limit * tile_h * tile_w / 256
    luts = np.zeros((nh, nw, 256), dtype=np.float32)

    for i in range(nh):
        for j in range(nw):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))
            excess = np.sum(np.maximum(hist - clip_limit_abs, 0))
            hist = np.minimum(hist, clip_limit_abs) + excess / 256
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6) * 255
            luts[i, j] = cdf

    result = np.zeros_like(img, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            ti = y // tile_h
            tj = x // tile_w
            yi = y - ti * tile_h
            xi = x - tj * tile_w

            ti = min(ti, nh - 2) if ti >= nh - 1 else ti
            tj = min(tj, nw - 2) if tj >= nw - 1 else tj

            wy = yi / tile_h
            wx = xi / tile_w

            v00 = luts[ti, tj, img[y, x]]
            v01 = luts[ti, tj+1, img[y, x]]
            v10 = luts[ti+1, tj, img[y, x]]
            v11 = luts[ti+1, tj+1, img[y, x]]

            val = (1 - wy) * (1 - wx) * v00 + (1 - wy) * wx * v01 + \
                  wy * (1 - wx) * v10 + wy * wx * v11
            result[y, x] = np.clip(val, 0, 255).astype(np.uint8)

    if pad_h > 0 or pad_w > 0:
        result = result[:h-pad_h, :w-pad_w]

    return result

def image_stats(image):
    return np.mean(image[:, :, 1:], axis=(0, 1)), np.std(image[:, :, 1:], axis=(0, 1))


def is_skin_or_lips(lab_image):
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]
    skin = (l > 20) & (l < 250) & (a > 120) & (a < 180) & (b > 120) & (b < 190)
    lips = (l > 20) & (l < 200) & (a > 150) & (b > 140)
    return (skin | lips).astype(np.float32)


def adjust_brightness(image, factor, mask=None):
    hsv = bgr_to_hsv(image)
    v = hsv[:, :, 2].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        v = np.where(mask > 0, np.clip(v * factor, 0, 255), v)
    else:
        v = np.clip(v * factor, 0, 255)
    hsv[:, :, 2] = v.astype(np.uint8)
    return hsv_to_bgr(hsv)


def adjust_saturation(image, factor, mask=None):
    hsv = bgr_to_hsv(image)
    s = hsv[:, :, 1].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        s = np.where(mask > 0, np.clip(s * factor, 0, 255), s)
    else:
        s = np.clip(s * factor, 0, 255)
    hsv[:, :, 1] = s.astype(np.uint8)
    return hsv_to_bgr(hsv)


def adjust_contrast(image, factor, mask=None):
    mean = np.mean(image)
    adjusted = image.astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        adjusted = np.where(mask > 0, np.clip((adjusted - mean) * factor + mean, 0, 255), adjusted)
    else:
        adjusted = np.clip((adjusted - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_tone(source, target, tone_strength=0.7, mask=None):
    h, w = target.shape[:2]
    source = pil_resize(source, (w, h))
    lab_image = bgr_to_lab(target).astype(np.float32)
    lab_source = bgr_to_lab(source).astype(np.float32)
    l_image = lab_image[:, :, 0]
    l_source = lab_source[:, :, 0]

    if mask is not None:
        mask = pil_resize(mask, (w, h))
        mask = mask.astype(np.float32) / 255.0
        l_adjusted = np.copy(l_image)
        mean_source = np.mean(l_source[mask > 0])
        std_source = np.std(l_source[mask > 0])
        mean_target = np.mean(l_image[mask > 0])
        std_target = np.std(l_image[mask > 0])
        l_adjusted[mask > 0] = (l_image[mask > 0] - mean_target) * (std_source / (std_target + 1e-6)) * 0.7 + mean_source
        l_adjusted[mask > 0] = np.clip(l_adjusted[mask > 0], 0, 255)
        l_adjusted_uint8 = l_adjusted.astype(np.uint8)
        l_enhanced = clahe(l_adjusted_uint8, clip_limit=2.5, grid_size=(8, 8)).astype(np.float32)
        l_final = l_adjusted * 0.7 + l_enhanced * 0.3
        l_final = np.clip(l_final, 0, 255)
        l_contrast = l_final * 1.3 - 20
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image[mask > 0] = l_image[mask > 0] * (1 - tone_strength) + l_contrast[mask > 0] * tone_strength
    else:
        mean_source = np.mean(l_source)
        std_source = np.std(l_source)
        l_mean = np.mean(l_image)
        l_std = np.std(l_image)
        l_adjusted = (l_image - l_mean) * (std_source / (l_std + 1e-6)) * 0.7 + mean_source
        l_adjusted = np.clip(l_adjusted, 0, 255)
        l_adjusted_uint8 = l_adjusted.astype(np.uint8)
        l_enhanced = clahe(l_adjusted_uint8, clip_limit=2.5, grid_size=(8, 8)).astype(np.float32)
        l_final = l_adjusted * 0.7 + l_enhanced * 0.3
        l_final = np.clip(l_final, 0, 255)
        l_contrast = l_final * 1.3 - 20
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image = l_image * (1 - tone_strength) + l_contrast * tone_strength

    lab_image[:, :, 0] = l_image
    return lab_to_bgr(lab_image.astype(np.uint8))


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return rgb_to_bgr(img)


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = bgr_to_rgb(bgr)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def color_transfer(source, target, mask=None, strength=1.0, skin_protection=0.2,
                   auto_brightness=True, brightness_range=0.5,
                   auto_contrast=False, contrast_range=0.5,
                   auto_saturation=False, saturation_range=0.5,
                   auto_tone=False, tone_strength=0.7):
    source_lab = bgr_to_lab(source).astype(np.float32)
    target_lab = bgr_to_lab(target).astype(np.float32)

    src_means, src_stds = image_stats(source_lab)
    tar_means, tar_stds = image_stats(target_lab)

    skin_lips_mask = is_skin_or_lips(target_lab.astype(np.uint8))
    skin_lips_mask_uint8 = (skin_lips_mask * 255).astype(np.uint8)
    skin_lips_mask_blur = pil_gaussian_blur(skin_lips_mask_uint8, 5)
    skin_lips_mask = skin_lips_mask_blur.astype(np.float32) / 255.0

    if mask is not None:
        if mask.ndim != 2:
            if mask.ndim == 3 and (mask.shape[0] == 1 or mask.shape[-1] == 1):
                mask = mask.squeeze()
            else:
                raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        mask = pil_resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0

    result_lab = target_lab.copy()
    for i in range(1, 3):
        adjusted_channel = (target_lab[:, :, i] - tar_means[i - 1]) * (src_stds[i - 1] / (tar_stds[i - 1] + 1e-6)) + src_means[i - 1]
        adjusted_channel = np.clip(adjusted_channel, 0, 255)

        if mask is not None:
            result_lab[:, :, i] = target_lab[:, :, i] * (1 - mask) + \
                                  (target_lab[:, :, i] * skin_lips_mask * skin_protection +
                                   adjusted_channel * skin_lips_mask * (1 - skin_protection) +
                                   adjusted_channel * (1 - skin_lips_mask)) * mask
        else:
            result_lab[:, :, i] = target_lab[:, :, i] * skin_lips_mask * skin_protection + \
                                  adjusted_channel * skin_lips_mask * (1 - skin_protection) + \
                                  adjusted_channel * (1 - skin_lips_mask)

    result_bgr = lab_to_bgr(result_lab.astype(np.uint8))
    final_result = (target * (1 - strength) + result_bgr * strength).astype(np.uint8)

    if mask is not None:
        mask = pil_resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0
        if auto_brightness:
            source_brightness = np.mean(bgr_to_gray(source))
            target_brightness = np.mean(bgr_to_gray(target))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, -brightness_range, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor, mask)
        if auto_contrast:
            source_gray = bgr_to_gray(source)
            target_gray = bgr_to_gray(target)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, -contrast_range, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor, mask)
        if auto_saturation:
            source_hsv = bgr_to_hsv(source)
            target_hsv = bgr_to_hsv(target)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, -saturation_range, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor, mask)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength, mask)
    else:
        if auto_brightness:
            source_brightness = np.mean(bgr_to_gray(source))
            target_brightness = np.mean(bgr_to_gray(target))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, -brightness_range, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor)
        if auto_contrast:
            source_gray = bgr_to_gray(source)
            target_gray = bgr_to_gray(target)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, -contrast_range, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor)
        if auto_saturation:
            source_hsv = bgr_to_hsv(source)
            target_hsv = bgr_to_hsv(target)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, -saturation_range, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength)

    return final_result

class CP_ImitationHueNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imitation_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_protection": ("FLOAT", {"default": 0.2, "min": 0, "max": 1.0, "step": 0.1}),
                "auto_brightness": ("BOOLEAN", {"default": True}),
                "brightness_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "auto_contrast": ("BOOLEAN", {"default": False}),
                "contrast_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "auto_saturation": ("BOOLEAN", {"default": False}),
                "saturation_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "auto_tone": ("BOOLEAN", {"default": False}),
                "tone_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            },
        }

    CATEGORY = categoryFolder

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "imitation_hue"

    def imitation_hue(self, imitation_image, target_image, strength, skin_protection,
                      auto_brightness, brightness_range,
                      auto_contrast, contrast_range,
                      auto_saturation, saturation_range,
                      auto_tone, tone_strength,
                      mask=None):
        img_cv1 = tensor_to_bgr(imitation_image[0])
        img_cv2 = tensor_to_bgr(target_image[0])

        img_cv3 = None
        if mask is not None:
            m = mask.cpu().numpy()
            if m.ndim == 4:
                m = m[0, 0]
            elif m.ndim == 3:
                if m.shape[0] == 1:
                    m = m[0]
                elif m.shape[-1] in (1, 3, 4):
                    m = m[..., 0]
                else:
                    m = m[0]
            elif m.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected mask dimensions: {m.ndim}")

            if m.ndim == 1:
                h, w = mask.shape[-2], mask.shape[-1]
                m = m.reshape(h, w)

            img_cv3 = (m * 255).astype(np.uint8)

        result_img = color_transfer(img_cv1, img_cv2, img_cv3, strength, skin_protection,
                                    auto_brightness, brightness_range,
                                    auto_contrast, contrast_range,
                                    auto_saturation, saturation_range,
                                    auto_tone, tone_strength)

        return (bgr_to_tensor(result_img),)


NODE_CLASS_MAPPINGS_IMPORTED_MING_IMITATIONHUE = {
    "CP ImitationHueNode": CP_ImitationHueNode,
}

NODE_DISPLAY_NAME_MAPPINGS_IMPORTED_MING_IMITATIONHUE = {
    "CP ImitationHueNode": "Imitation Hue",
}
