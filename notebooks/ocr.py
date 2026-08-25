# Generated from: ocr.ipynb
# Converted at: 2026-06-27T20:01:19.101Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Untuk Experiment


import cv2
from matplotlib import pyplot as plt

image_file = "../data/nota_pensmart.jpeg"
image_mat = cv2.imread(image_file)

def display_img(im_data, dpi=80):
    height, width  = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

display_img(image_mat)

# ## Donut Model


import torch
import xmltodict
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, DonutProcessor, VisionEncoderDecoderModel

model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)  

# ### Preprocessing


image = Image.open("../data/rotated2.jpg").convert("RGB")
image

image = image.rotate(90, expand=True)
image

decoder_input_id = processor.tokenizer("<s_cord-v2>", add_special_tokens=False).input_ids
decoder_input_id

decoder_input_id = torch.tensor(decoder_input_id).unsqueeze(0)
decoder_input_id

pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values

# ### Inference


output = model.generate(
            pixel_values,
            decoder_input_ids = decoder_input_id,
            max_length = model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

output

# ### Post-Processing


decoded_sequence = processor.batch_decode(output.sequences)[0]
decoded_sequence

print(decoded_sequence)
dict_ = processor.token2json(decoded_sequence)
dict_

dict_["total"]['total_price']

# ## Florence-2-base


from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base-ft", 
            trust_remote_code=True,
            dtype=torch.float32, 
            device_map="cpu",
            attn_implementation="eager"
            )

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

image = Image.open("../data/rotated2.jpg").convert("RGB")
image

inputs = processor(text="<OCR>", images=image, return_tensors="pt").to("cpu", torch.float32)
inputs

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
    use_cache=False
)

generated_ids

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
generated_text

parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))
parsed_answer

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF_TOKEN belum diset"

import os
import json
import requests

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    "Content-Type": "application/json"
}

def parse_receipt(ocr_text):
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You extract structured JSON from messy receipt OCR. Return JSON only."
            },
            {
                "role": "user",
                "content": f"""
Parse this receipt into JSON.

Schema:
{{
  "merchant": string,
  "items": [
    {{
      "name": string,
      "qty": number | null,
      "price": number
    }}
  ],
  "total": number,
  "payment_method": string | null
}}

Rules:
- Currency IDR
- Numbers only
- If unsure, use null
- DO NOT explain

OCR TEXT:
{ocr_text}
"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }

    r = requests.post(API_URL, headers=headers, json=payload)
    r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"]

    # safety cut
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


parsed = parse_receipt(generated_text)
parsed

parsed['items'][1]['name']

# # Experiment


# ## Warp Perspective


import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def robust_receipt_scanner(img_array):
    # 1. Resize
    orig = img_array.copy()
    ratio = img_array.shape[0] / 500.0
    h = 500
    w = int(img_array.shape[1] / ratio)
    img_small = cv2.resize(img_array, (w, h))

    # Convert ke HSV
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # Kertas putih ciri-ciri: Saturation RENDAH, Value TINGGI.
    # S: 0-255 (Makin kecil makin pudar/putih)
    # V: 0-255 (Makin gede makin terang)
    
    # Saturation (hsv[:,:,1])
    saturation = hsv[:,:,1]
    
    _, binary = cv2.threshold(saturation, 40, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        # Mainkan epsilon. 0.02 standar, kalau hasilnya mletot coba naik/turun dikit
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Cari yang punya 4 sudut
        if len(approx) == 4:
            screenCnt = approx
            break
            
    # --- FALLBACK: Kalau gagal nemu 4 sudut ---
    # Paksa pake Bounding Rect dari kontur terbesar
    if screenCnt is None and len(cnts) > 0:
        print("Gagal nemu 4 sudut presisi, pake bounding box biasa.")
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        screenCnt = np.int0(box)

    if screenCnt is None:
        return orig

    screenCnt = screenCnt.astype("float32") * ratio
    
    
    warped = four_point_transform(orig, screenCnt.reshape(4, 2))
    
    return warped


img = cv2.imread("../data/nota_pensmart.jpeg", cv2.IMREAD_COLOR_RGB)
hasil = robust_receipt_scanner(img)
display_img(hasil)

# %pip install scikit-image

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray

# ======================
# LOAD IMAGE
# ======================
# image = cv2.imread("hasil_warp.png")
image = hasil.copy()
assert image is not None, "gambar gak kebaca"

image = cv2.blur(image, (4, 4))
gray = rgb2gray(image)

# ======================
# EDGE DETECTION
# ======================
edges = canny(gray)

# ======================
# HOUGH TRANSFORM
# ======================
tested_angles = np.deg2rad(np.arange(0.1, 180.0))
h, theta, d = hough_line(edges, theta=tested_angles)

# ======================
# VISUALIZATION
# ======================
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
ax = axes.ravel()

ax[0].imshow(gray, cmap="gray")
ax[0].set_title("Input image")
ax[0].axis("off")

ax[1].imshow(edges, cmap="gray")
ax[1].set_title("Blue = Horizontal | Red = Vertical")
ax[1].axis("off")

origin = np.array((0, edges.shape[1]))

# ======================
# DRAW *ALL* LINES (NO FILTER)
# ======================
accums, angles, dists = hough_line_peaks(h, theta, d)

horizontal_lines = []
vertical_lines = []
diagonal_lines = []

for angle, dist in zip(angles, dists):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    angle_deg = np.rad2deg(angle)

    if 75 <= angle_deg <= 105:
        # horizontal
        color = 'b'
        horizontal_lines.append((angle, angle_deg, dist))

    elif angle_deg <= 15 or angle_deg >= 165:
        # vertical
        color = 'r'
        vertical_lines.append((angle, angle_deg, dist))

    else:
        # diagonal
        color = 'y'
        diagonal_lines.append((angle, angle_deg, dist))

    ax[1].plot(origin, (y0, y1), color, linewidth=1)


ax[1].set_xlim(0, edges.shape[1])
ax[1].set_ylim(edges.shape[0], 0)

plt.tight_layout()
plt.show()

print(f"Total lines      : {len(angles)}")
print(f"Horizontal lines : {len(horizontal_lines)}")
print(f"Vertical lines   : {len(vertical_lines)}")


# display([t[1] for t in horizontal_lines])
# vertical_lines
[t[1] for t in horizontal_lines]

def get_mean_error(data):
    median = np.median(data)
    mean_error = np.mean([abs(t - median) for t in data])

    return median, mean_error

def normalize_vertical_angle(angle_deg):
    if angle_deg > 90:
        return angle_deg - 180
    return angle_deg


v_median, v_mean_error = get_mean_error([normalize_vertical_angle(t[1]) for t in vertical_lines])

print(f'v median : {v_median}')
print(f'v mean error : {v_mean_error}')

h_median, h_mean_error = get_mean_error([t[1] for t in horizontal_lines])

print(f'h median : {h_median}')
print(f'h mean error : {h_mean_error}')

v_rotate = [90, -90]
h_rotate = [0, 180]

img_rotate = []
if (v_mean_error < h_mean_error):
    img_rotate = v_rotate
    print(f"kertasnnya miring, opsi rotasi : {img_rotate}, salah rotasi maka kertas kebalik")
else:
    img_rotate = h_rotate
    print(f"kertasnnya udah potrait, opsi rotasi : {img_rotate}, salah rotasi maka kertas kebalik")

# ### OPSI B


import numpy as np
import cv2

def choose_rotation(img, candidates):
    
    best_rotation = candidates[0]
    best_score = -1
    
    for angle in candidates:
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h_projection = np.sum(binary, axis=1)
        h_variance = np.var(h_projection)
        
        h_mean = np.mean(h_projection)
        h_peaks = np.sum(h_projection > h_mean * 1.5)
        
        score = h_variance * 0.5 + h_peaks * 50
        
        print(f"{angle:4d}°: score={score:.1f}")
        
        if score > best_score:
            best_score = score
            best_rotation = angle
    
    return best_rotation



best_rotation = choose_rotation(hasil, img_rotate)

if best_rotation == 90:
    corrected = cv2.rotate(hasil, cv2.ROTATE_90_CLOCKWISE)
elif best_rotation == -90:
    corrected = cv2.rotate(hasil, cv2.ROTATE_90_COUNTERCLOCKWISE)
elif best_rotation == 180:
    corrected = cv2.rotate(hasil, cv2.ROTATE_180)
else:
    corrected = hasil

# cv2.imwrite("../data/rotated2.jpg", corrected)
h, w = corrected.shape[:2]
corrected_half = cv2.resize(corrected, (w // 4, h // 4))
display_img(corrected_half)