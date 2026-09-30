from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

_processor = None
_model = None

def _load():
    global _processor, _model
    if _processor is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _processor, _model

def generate_caption(image_rgb):
    processor, model = _load()
    image = Image.fromarray(image_rgb)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)
