import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from base import AIModel, ItemData, ReceiptData
from ..utility.parsing import parse_receipt

MODEL_NAME = "microsoft/Florence-2-base-ft"

class FlorenceModel(AIModel):
    def __init__(self):
        print(f"Loading Florence: {MODEL_NAME}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            torch_dtype=torch.float32, 
            device_map=self.device,
            attn_implementation="eager"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def run(self, image: Image.Image) -> ReceiptData:
        inputs = self._preprocess(image)
        generated_ids = self._inference(inputs)
        raw_text_result = self._postprocessing(generated_ids, image)
        
        json_data = parse_receipt(raw_text_result)
        
        return self._formatting(json_data)

    def _preprocess(self, image):
        inputs = self.processor(text="<OCR>", images=image, return_tensors="pt")
        return inputs.to(self.device)

    def _inference(self, inputs):
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            use_cache=False 
        )
        return generated_ids

    def _postprocessing(self, generated_ids, image):
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task="<OCR>", 
            image_size=(image.width, image.height)
        )
        return parsed_answer.get('<OCR>', '')

    def _formatting(self, json_data: dict) -> ReceiptData:
        try:
            raw_items = json_data.get("items", [])
            items = []
            
            for item in raw_items:
                name = item.get("name", "Unknown")
                
                qty = item.get("qty")
                if qty is None: qty = 1
                
                price = item.get("price", 0)
                
                items.append(ItemData(
                    name=str(name),
                    count=int(qty),
                    total_price=float(price)
                ))
            
            total_val = float(json_data.get("total", 0))
            
            return ReceiptData(items={it.id: it for it in items}, total=total_val)
            
        except Exception as e:
            print(f"Formatting Error: {e}")
            return ReceiptData(items={}, total=0.0)