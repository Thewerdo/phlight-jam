import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json

def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def white_balance(image):
    img_rgb = image.convert("RGB")
    arr_rgb = np.array(img_rgb).reshape(-1, 3)
    r, g, b = arr_rgb[:, 0], arr_rgb[:, 1], arr_rgb[:, 2]
    avg_r = np.mean(r)
    avg_g = np.mean(g)
    avg_b = np.mean(b)
    by_axis = avg_b - avg_r
    k = float(np.clip(by_axis, -128, 128))
    gp_axis = avg_g - (avg_r + avg_b) / 2
    tint = ((-gp_axis + 128) / 256) * 300 - 150
    tint = float(np.clip(tint, -150, 150))
    return k, tint

def lum_histogram(image):
    grayscale = np.array(image.convert("L")).astype(np.float32) / 255.0 * 100
    levels = {
        "black": ((grayscale >= 0) & (grayscale < 10)),
        "shadow": ((grayscale >= 20) & (grayscale < 40)),
        "mid-tone": ((grayscale >= 40) & (grayscale < 60)),
        "highlight": ((grayscale >= 60) & (grayscale < 80)),
        "white": ((grayscale >= 80) & (grayscale <= 100))
    }
    histogram = {level: int(np.sum(mask)) for level, mask in levels.items()}
    return histogram

def lum_sections_avgrgb(image):
    img_rgb = np.array(image.convert("RGB"))
    img_gray = np.array(image.convert("L"))
    arr_gray = img_gray.flatten() / 255.0
    arr_rgb = img_rgb.reshape(-1, 3)
    masks = {
        "shadow": (arr_gray >= 0.0) & (arr_gray < 0.4),
        "mid-tone": (arr_gray >= 0.4) & (arr_gray < 0.6),
        "highlight": (arr_gray >= 0.6) & (arr_gray <= 1.0)
    }
    avg_rgbs = {}
    for section, mask in masks.items():
        if section == "highlight" and np.any(mask):
            highlights = arr_rgb[mask]
            if len(highlights) >= 2:
                sorted_highlights = highlights[np.argsort(np.mean(highlights, axis=1))]
                mid = len(sorted_highlights) // 2
                avg1 = tuple(np.mean(sorted_highlights[:mid], axis=0).astype(int))
                avg_rgbs["highlight_main1"] = avg1
            else:
                avg_rgb = tuple(np.mean(highlights, axis=0).astype(int))
                avg_rgbs["highlight_main1"] = avg_rgb
        elif np.any(mask):
            avg_rgb = tuple(np.mean(arr_rgb[mask], axis=0).astype(int))
            avg_rgbs[section] = avg_rgb
        else:
            avg_rgbs[section] = (0, 0, 0)
    return avg_rgbs

def avg_sat(image):
    img_hsv = image.convert("HSV")
    arr_hsv = np.array(img_hsv)
    sat = arr_hsv[..., 1].astype(np.float32)
    avg_sat_val = np.mean(sat) / 255 * 100
    return avg_sat_val

def rgb_eight(image):
    img_rgb = image.convert("RGB")
    arr_rgb = np.array(img_rgb).reshape(-1, 3)
    arr_hsv = np.array(img_rgb.convert("HSV")).reshape(-1, 3)
    hue = arr_hsv[:, 0].astype(np.float32) * 360.0 / 255.0
    color_ranges = {
        "red": ((hue >= 0) & (hue < 15)) | (hue >= 345),
        "orange": (hue >= 15) & (hue < 45),
        "yellow": (hue >= 45) & (hue < 70),
        "green": (hue >= 70) & (hue < 170),
        "aqua": (hue >= 170) & (hue < 190),
        "blue": (hue >= 190) & (hue < 255),
        "purple": (hue >= 255) & (hue < 300),
        "magenta": (hue >= 300) & (hue < 345)
    }
    avg_rgbs = {}
    for color, mask in color_ranges.items():
        if np.any(mask):
            avg_rgb = tuple(np.mean(arr_rgb[mask], axis=0).astype(int))
        else:
            avg_rgb = (0, 0, 0)
        avg_rgbs[color] = avg_rgb
    return avg_rgbs

def side_lum(image):
    img_gray = image.convert("L")
    arr = np.array(img_gray)
    h, w = arr.shape
    margin = int(min(h, w) * 0.1)
    edge_regions = np.concatenate([
        arr[:margin, :].flatten(),
        arr[-margin:, :].flatten(),
        arr[:, :margin].flatten(),
        arr[:, -margin:].flatten(),
        arr[:margin, :margin].flatten(),
        arr[:margin, -margin:].flatten(),
        arr[-margin:, :margin].flatten(),
        arr[-margin:, -margin:].flatten()
    ])
    return float(np.mean(edge_regions))

def sharpness_level(image):
    img_lum = image.convert('L')
    lum_array = np.asarray(img_lum, dtype=np.int32)
    gy, gx = np.gradient(lum_array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def rgb_to_hsv(rgb):
    arr = np.array([[rgb]], dtype=np.uint8)
    hsv = np.array(Image.fromarray(arr, 'RGB').convert('HSV'))[0,0]
    h = (hsv[0] / 255) * 360
    s = (hsv[1] / 255) * 100
    v = (hsv[2] / 255) * 100
    return (h, s, v)

def process_images(self_base64, ref_base64):
    # Decode base64 strings
    self_bytes = base64.b64decode(self_base64)
    ref_bytes = base64.b64decode(ref_base64)
    
    # Open images with PIL
    image_self = Image.open(BytesIO(self_bytes))
    image_ref = Image.open(BytesIO(ref_bytes))
    
    # Convert to RGB
    image_self = convert_to_rgb(image_self)
    image_ref = convert_to_rgb(image_ref)
    
    # Compute image metrics
    wb_self = white_balance(image_self)
    wb_ref = white_balance(image_ref)
    
    hist_self = lum_histogram(image_self)
    hist_ref = lum_histogram(image_ref)
    
    lum_level_rgb_self = lum_sections_avgrgb(image_self)
    lum_level_rgb_ref = lum_sections_avgrgb(image_ref)
    
    avg_sat_self = avg_sat(image_self)
    avg_sat_ref = avg_sat(image_ref)
    
    rgb_eight_self = rgb_eight(image_self)
    rgb_eight_ref = rgb_eight(image_ref)
    
    side_lum_self = side_lum(image_self)
    side_lum_ref = side_lum(image_ref)
    
    sharpness_self = sharpness_level(image_self)
    sharpness_ref = sharpness_level(image_ref)
    
    # Compare images and compute adjustments
    wb_array = [int(wb_ref[0] - wb_self[0]), int(wb_ref[1] - wb_self[1])]
    
    histogram_array = []
    levels = ['black', 'shadow', 'mid-tone', 'highlight', 'white']
    max_pixels = max(sum(hist_self.values()), sum(hist_ref.values()))
    for level in levels:
        diff = hist_ref[level] - hist_self[level]
        normalized_diff = int(np.clip((diff / max_pixels) * 100, -100, 100))
        histogram_array.append(normalized_diff)
    
    color_grading_array = []
    sections_order = ['shadow', 'mid-tone', 'highlight_main1']
    for section in sections_order:
        if section in lum_level_rgb_ref and section in lum_level_rgb_self:
            rgb_values1, rgb_values2 = list(lum_level_rgb_ref[section]), list(lum_level_rgb_self[section])
            hsv_ref = rgb_to_hsv(rgb_values1)
            hsv_self = rgb_to_hsv(rgb_values2)
            h_diff = int(min(((hsv_ref[0] - hsv_self[0] + 180) % 360 - 180), 100))
            s_diff = int(np.clip(hsv_ref[1] - hsv_self[1], -100, 100))
            v_diff = int(np.clip(hsv_ref[2] - hsv_self[2], -100, 100))
            diff = [h_diff, s_diff, v_diff]
            color_grading_array.extend(diff)
        else:
            color_grading_array.extend([0, 0, 0])
    
    sat_array = [int(avg_sat_ref - avg_sat_self)]
    
    mixer_array = []
    for color in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
        if color in rgb_eight_ref and color in rgb_eight_self:
            hsv_ref = rgb_to_hsv(rgb_eight_ref[color])
            hsv_self = rgb_to_hsv(rgb_eight_self[color])
            hue_diff = (hsv_ref[0] - hsv_self[0] + 180) % 360 - 180
            sat_diff = hsv_ref[1] - hsv_self[1]
            lum_diff = hsv_ref[2] - hsv_self[2]
            mixer_array.extend([
                int(np.clip(hue_diff, -180, 180)),
                int(np.clip(sat_diff, -100, 100)),
                int(np.clip(lum_diff, -100, 100))
            ])
        else:
            mixer_array.extend([0, 0, 0])
    
    vignette_array = [int(side_lum_ref - side_lum_self), 1]
    
    sharp_array = [int(sharpness_ref - sharpness_self), 1]
    
    # Combine all adjustments into a single array
    all_values = []
    all_values.extend(wb_array)           # 2 values: Temperature, Tint
    all_values.extend(histogram_array)    # 5 values: Black, Shadow, Mid-tone, Highlight, White
    all_values.extend(color_grading_array)# 9 values: Shadow H/S/V, Mid-tone H/S/V, Highlight H/S/V
    all_values.extend(sat_array)          # 1 value: Saturation
    all_values.extend(mixer_array)        # 24 values: 8 colors x (Hue, Saturation, Luminance)
    all_values.extend(vignette_array)     # 2 values: Vignette Amount, Vignette Midpoint
    all_values.extend(sharp_array)        # 2 values: Sharpness Amount, Sharpness Radius
    
    return all_values