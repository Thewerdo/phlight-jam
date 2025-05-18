import os
import sys
from PIL import Image
import numpy as np


filename1 = __import__ #put the link to the import images here
filename2 = __import__ #put the link to the import images here

def convert_jpg(infile):
    outfile = os.path.splitext(infile)[0] + ".jpg"
    try:
        with Image.open(infile) as im:
            rgb_im = im.convert("RGB")
            rgb_im.save(outfile, "JPEG")
            print(f"Converted {infile} to {outfile}")
        return outfile  
    except OSError:
        print("Cannot convert", infile)
        return None

def white_balance(image):
    with Image.open(image) as img:
        img_rgb = img.convert("RGB")
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
    with Image.open(image) as img:
        # Convert to numpy array first
        grayscale = np.array(img.convert("L")).astype(np.float32) / 255.0 * 100
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
    try:
        with Image.open(image) as img:
           
            img_rgb = np.array(img.convert("RGB"))
            img_gray = np.array(img.convert("L"))
            
            
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
                        avg2 = tuple(np.mean(sorted_highlights[mid:], axis=0).astype(int))
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
    except Exception as e:
        print(f"Error in lum_sections_avgrgb: {str(e)}")
        return {
            "shadow": (0, 0, 0),
            "mid-tone": (128, 128, 128),
            "highlight_main1": (255, 255, 255),
            "highlight_main2": (255, 255, 255)
        }

def avg_sat(image):
    with Image.open(image) as img:
        img_hsv = img.convert("HSV")
        arr_hsv = np.array(img_hsv)
        
        sat = arr_hsv[..., 1].astype(np.float32)
        avg_sat_val = np.mean(sat) / 255 * 100
        return avg_sat_val

def rgb_eight(image):
    with Image.open(image) as img:
        img_rgb = img.convert("RGB")
        arr_rgb = np.array(img_rgb).reshape(-1, 3)
        arr_hsv = np.array(img_rgb.convert("HSV")).reshape(-1, 3)
        hue = arr_hsv[:, 0].astype(np.float32) * 360.0 / 255.0  

        
        color_ranges = {
            "red":     ((hue >= 0)   & (hue < 15))  | (hue >= 345),
            "orange":  (hue >= 15)   & (hue < 45),
            "yellow":  (hue >= 45)   & (hue < 70),
            "green":   (hue >= 70)   & (hue < 170),
            "aqua":   (hue >= 170)  & (hue < 190),
            "blue":    (hue >= 190)  & (hue < 255),
            "purple":  (hue >= 255) & (hue < 300),
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
    with Image.open(image) as img:
        img_gray = img.convert("L")
        arr = np.array(img_gray)
        h, w = arr.shape
        
        margin = int(min(h, w) * 0.1)
        
        edge_regions = np.concatenate([
            arr[:margin, :].flatten(),          # top
            arr[-margin:, :].flatten(),         # bottom
            arr[:, :margin].flatten(),          # left
            arr[:, -margin:].flatten(),         # right
            arr[:margin, :margin].flatten(),    # top-left
            arr[:margin, -margin:].flatten(),   # top-right
            arr[-margin:, :margin].flatten(),   # bottom-left
            arr[-margin:, -margin:].flatten()   # bottom-right
        ])
        
        return float(np.mean(edge_regions))

def sharpness_level(image): #usually around 2-30
    with Image.open(image) as img:
        img_lum = img.convert('L') # to grayscale
        lum_array = np.asarray(img_lum, dtype=np.int32)

        gy, gx = np.gradient(lum_array)  
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        
        return sharpness

def rgb_to_hsv(rgb):
    """Convert RGB tuple to HSV tuple"""
    arr = np.array([[rgb]], dtype=np.uint8)
    hsv = np.array(Image.fromarray(arr, 'RGB').convert('HSV'))[0,0]
    h = (hsv[0] / 255) * 360
    s = (hsv[1] / 255) * 100
    v = (hsv[2] / 255) * 100
    return (h,s,v)

def image_check(wb_self, wb_ref, lum_level_rgb_self, lum_level_rgb_ref, avg_sat_self, avg_sat_ref, rgb_eight_self, rgb_eight_ref, side_lum_self, side_lum_ref, sharpness_self, sharpness_ref):
    # White balance and tint [temp_change, tint_change]
    wb_array = [
        int(wb_ref[0] - wb_self[0]),
        int(wb_ref[1] - wb_self[1])
    ]

    #Histogram changes
    histogram_array = []
    levels = ['black', 'shadow', 'mid-tone', 'highlight', 'white']
    
    # Get histograms for both images
    hist_self = lum_histogram(jpg_self)
    hist_ref = lum_histogram(jpg_ref)
    
    # Calculate the maximum possible difference for normalization
    max_pixels = max(sum(hist_self.values()), sum(hist_ref.values()))
    
    # Compare each level and normalize to -100 to 100 scale
    for level in levels:
        diff = hist_ref[level] - hist_self[level]
        normalized_diff = int(np.clip((diff / max_pixels) * 100, -100, 100))
        histogram_array.append(normalized_diff)

    # Color grading array - order from dark to bright
    color_grading_array = []
    sections_order = ['shadow', 'mid-tone', 'highlight_main1']  # Correct order: shadow first, then mid-tone, then highlight
    for section in sections_order:
        if section in lum_level_rgb_ref and section in lum_level_rgb_self:
            rgb_values1, rgb_values2 = list(lum_level_rgb_ref[section]), list(lum_level_rgb_self[section])
            hsv_ref = rgb_to_hsv(rgb_values1)
            hsv_self = rgb_to_hsv(rgb_values2)
            
            # Calculate differences with constraints
            h_diff = int(min(((hsv_ref[0] - hsv_self[0] + 180) % 360 - 180), 100))  # Constrain hue diff
            s_diff = int(np.clip(hsv_ref[1] - hsv_self[1], -100, 100))  # Saturation diff
            v_diff = int(np.clip(hsv_ref[2] - hsv_self[2], -100, 100))  # Value diff
            
            diff = [h_diff, s_diff, v_diff]
            color_grading_array.extend(diff)
        else:
            color_grading_array.extend([0, 0, 0])

    # Saturation change
    sat_array = [int(avg_sat_ref - avg_sat_self)]

    # Color mixer array [red_r, red_g, red_b, orange_r, orange_g, orange_b, ...] - comparing ref and self
    mixer_array = []
    for color in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
        if color in rgb_eight_ref and color in rgb_eight_self:
            hsv_ref = rgb_to_hsv(rgb_eight_ref[color])
            hsv_self = rgb_to_hsv(rgb_eight_self[color])

            hue_diff = (hsv_ref[0] - hsv_self[0] + 180) % 360 - 180  # Hue: wrap-around aware
            sat_diff = hsv_ref[1] - hsv_self[1]                      # Saturation
            lum_diff = hsv_ref[2] - hsv_self[2]                      # Luminance (Value in HSV)

            # Optional: clip or scale these if you're displaying/applying them as Lightroom-like sliders
            mixer_array.extend([
                int(np.clip(hue_diff, -180, 180)),
                int(np.clip(sat_diff, -100, 100)),
                int(np.clip(lum_diff, -100, 100))
            ])
        else:
            mixer_array.extend([0, 0, 0])  # Default if color missing


    # Vignette (single value)
    vignette_array = [int(side_lum_ref - side_lum_self), 1]

    # Sharpness change
    sharp_array = [int(sharpness_ref - sharpness_self), 1]

    # Return flat array instead of nested
    all_values = []
    all_values.extend(wb_array)           # 2 values
    all_values.extend(histogram_array)    # 5 values
    all_values.extend(color_grading_array)# 9 values
    all_values.extend(sat_array)          # 1 value 
    all_values.extend(mixer_array)        # 24 values
    all_values.extend(vignette_array)     # 2 values
    all_values.extend(sharp_array)        # 2 values
    return all_values

jpg_self = convert_jpg(filename1)
jpg_ref = convert_jpg(filename2)

white_balance_self = white_balance(jpg_self)
white_balance_ref = white_balance(jpg_ref)

lum_hist_self = lum_histogram(jpg_self)
lum_hist_ref = lum_histogram(jpg_ref)

lum_level_rgb_self = lum_sections_avgrgb(jpg_self)
lum_level_rgb_ref = lum_sections_avgrgb(jpg_ref)

avg_sat_self = avg_sat(jpg_self)
avg_sat_ref = avg_sat(jpg_ref)  

rgb_eight_self = rgb_eight(jpg_self)
rgb_eight_ref = rgb_eight(jpg_ref)

side_lum_self = side_lum(jpg_self)
side_lum_ref = side_lum(jpg_ref)

sharpness_self = sharpness_level(jpg_self)
sharpness_ref = sharpness_level(jpg_ref)


result = image_check(
    white_balance_self, white_balance_ref,
    lum_level_rgb_self, lum_level_rgb_ref,
    avg_sat_self, avg_sat_ref,
    rgb_eight_self, rgb_eight_ref,
    side_lum_self, side_lum_ref,
    sharpness_self, sharpness_ref
)

# Convert to string format if needed
result_string = str(result)
print(result_string)


def generate_xmp(result, output_path="preset.xmp"):
    """Generate XMP file from image adjustment results"""
    xmp_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, 2017/05/06-01:08:21">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
   <crs:Temperature>{result[0]}</crs:Temperature>
   <crs:Tint>{result[1]}</crs:Tint>
   <crs:Exposure2012>0</crs:Exposure2012>
   <crs:Contrast2012>0</crs:Contrast2012>
   <crs:Highlights2012>{result[5]}</crs:Highlights2012>
   <crs:Shadows2012>{result[3]}</crs:Shadows2012>
   <crs:Saturation>{result[16]}</crs:Saturation>
   <crs:Sharpness>{result[43]}</crs:Sharpness>
   <crs:Vibrance>0</crs:Vibrance>
   <crs:ParametricShadows>0</crs:ParametricShadows>
   <crs:ParametricDarks>0</crs:ParametricDarks>
   <crs:ParametricLights>0</crs:ParametricLights>
   <crs:ParametricHighlights>0</crs:ParametricHighlights>
   <crs:ParametricShadowSplit>25</crs:ParametricShadowSplit>
   <crs:ParametricMidtoneSplit>50</crs:ParametricMidtoneSplit>
   <crs:ParametricHighlightSplit>75</crs:ParametricHighlightSplit>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""
    
    with open(output_path, 'w') as f:
        f.write(xmp_template)
    return output_path


def generate_cube(result, output_path="preset.cube"):
    """Generate CUBE LUT file from image adjustment results"""
    cube_header = """#Created with Phlight
LUT_3D_SIZE 32
DOMAIN_MIN 0 0 0
DOMAIN_MAX 1 1 1
"""
    def apply_adjustments(r, g, b):
        # Apply the color adjustments from result using the correct indices
        # Red channel adjustments (using first color in mixer array)
        r = np.clip(r + result[17]/100, 0, 1)  # Red hue
        g = np.clip(g + result[18]/100, 0, 1)  # Red saturation
        b = np.clip(b + result[19]/100, 0, 1)  # Red luminance
        return r, g, b

    with open(output_path, 'w') as f:
        f.write(cube_header)
        size = 32
        for b in np.linspace(0, 1, size):
            for g in np.linspace(0, 1, size):
                for r in np.linspace(0, 1, size):
                    r_adj, g_adj, b_adj = apply_adjustments(r, g, b)
                    f.write(f"{r_adj:.6f} {g_adj:.6f} {b_adj:.6f}\n")
    return output_path

# Add this at the end of your script to export both formats
def export_preset(result, name="preset"):
    """Export the adjustment settings as both XMP and CUBE files"""
    xmp_path = generate_xmp(result, f"{name}.xmp")
    cube_path = generate_cube(result, f"{name}.cube")
    print(f"Exported XMP preset to: {xmp_path}")
    print(f"Exported CUBE LUT to: {cube_path}")

def process_images(self_file, ref_file):
    # Convert browser File objects to PIL Images
    from io import BytesIO
    
    self_image = Image.open(BytesIO(self_file))
    ref_image = Image.open(BytesIO(ref_file))
    
    # Rest of your processing logic
    return result