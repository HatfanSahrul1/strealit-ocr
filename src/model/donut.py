import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from base import AIModel, ItemData, ReceiptData

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
        receipt_dict = self._postprocessing(generation_output)
        return self._formatting(receipt_dict)

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
        
        dict_ = self.processor.token2json(decoded_sequence)
        return dict_

    def _formatting(self, receipt_dict: dict) -> ReceiptData:
        try:
            raw_items = receipt_dict.get('menu', [])
            
            if isinstance(raw_items, dict):
                raw_items = [raw_items]

            items = []
            for entry in raw_items:
                if not isinstance(entry, dict):
                    continue

                name_obj = entry.get('nm', 'Unknown')
                cnt_obj = entry.get('cnt', '1')
                price_obj = entry.get('price', '0')

                if isinstance(name_obj, dict):
                    continue 
                else:
                    name = str(name_obj)

                count_str = str(cnt_obj).lower().replace('x', '').replace(' ', '')
                
                if not count_str.isdigit():
                    digits = re.findall(r'\d+', count_str)
                    count = int(digits[0]) if digits else 1
                else:
                    count = int(count_str)

                total_price = self._convert_price_str_to_float(str(price_obj))

                items.append(ItemData(
                    name=name,
                    count=count,
                    total_price=total_price
                ))

            total_obj = receipt_dict.get('total', {})
            
            if isinstance(total_obj, dict):
                total_str = total_obj.get('total_price', '0')
            else:
                total_str = str(total_obj)
                
            total_val = self._convert_price_str_to_float(total_str)

            return ReceiptData(items={it.id: it for it in items}, total=total_val)

        except Exception as e:
            print(f"Format Error di Donut: {e}")
            return ReceiptData(items={}, total=0.0)

    def _convert_price_str_to_float(self, price_str: str) -> float:
        if not price_str: return 0.0
        if re.search(r'[a-zA-Z]', price_str.replace(',', '').replace('.', '')): 
             pass
        
        clean_price = re.sub(r"[^0-9.,]", "", str(price_str))
        
        try:
            clean_digits = re.sub(r"[^0-9]", "", clean_price)
            return float(clean_digits) if clean_digits else 0.0
        except:
            return 0.0