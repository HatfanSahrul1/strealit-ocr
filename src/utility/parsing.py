import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# Pastikan token ada di .env atau environment variable system
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"

def parse_receipt(ocr_text: str):
    """
    Kirim raw OCR text ke Qwen via HF Router API buat dapet JSON bersih.
    """
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN belum diset.")
        return {"items": [], "total": 0}

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

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
- Numbers only (remove Rp, dots as thousand separators if needed)
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

    try:
        r = requests.post(API_URL, headers=headers, json=payload)
        r.raise_for_status()

        text = r.json()["choices"][0]["message"]["content"]

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"items": [], "total": 0}
            
        data = json.loads(text[start:end])
        return data

    except Exception as e:
        print(f"Error Parsing: {e}")
        return {"items": [], "total": 0}