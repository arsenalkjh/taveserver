from ultralytics.models.sam import SAM3SemanticPredictor
from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights" / "sam3.pt"

def load_qwen3():
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def load_qwen_vl():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    return model, processor

def load_sam3():
    overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str(WEIGHTS_DIR),
    half=True,  # Use FP16 for faster inference
    save=True,
    )
    SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)
    return SAM_MODEL