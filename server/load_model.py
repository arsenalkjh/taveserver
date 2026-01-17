from ultralytics.models.sam import SAM3SemanticPredictor
from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights" / "sam3.pt"

def load_qwen3():
    # Using VL model for text processing to save memory (2B instead of 8B)
    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, processor

def load_varco_ocr():
    from transformers import LlavaOnevisionForConditionalGeneration
    
    model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="sdpa",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

def load_qwen_vl():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    return model, processor

def load_sam3():
    overrides = dict(
    conf=0.90,
    task="segment",
    mode="predict",
    model=str(WEIGHTS_DIR),
    half=True,  # Use FP16 for faster inference
    save=False,
    )
    SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)
    return SAM_MODEL