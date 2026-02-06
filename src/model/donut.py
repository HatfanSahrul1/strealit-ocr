import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from base import AIModel, ItemData, ReceiptData
from ..utility.parsing import parse_receipt

MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

class DonutModel(AIModel):
    def __init__(self):
        print(f"Loading Donut: {MODEL_NAME}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def run(self, image: Image.Image) -> ReceiptData:
        decoder_input_ids, pixel_values = self._preprocess(image)
        generation_output = self._inference(decoder_input_ids, pixel_values)
        raw_text = self._postprocessing(generation_output)
        json_data = parse_receipt(raw_text)
        return self._formatting(json_data)

    def _preprocess(self, image): 
        decoder_input_ids = self.processor.tokenizer(
            "<s_cord-v2>", add_special_tokens=False
        ).input_ids
        decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze(0).to(self.device)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        return decoder_input_ids, pixel_values

    def _inference(self, decoder_input_ids, pixel_values): 
        generation_output = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        return generation_output

    def _postprocessing(self, generation_output):
        decoded_sequence = self.processor.batch_decode(generation_output.sequences)[0]
        decoded_sequence = decoded_sequence.replace(self.processor.tokenizer.eos_token, "")
        decoded_sequence = decoded_sequence.replace(self.processor.tokenizer.pad_token, "")
        return self.processor.token2json(decoded_sequence)

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