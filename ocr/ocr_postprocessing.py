import json
import re
from pathlib import Path

def ocr_postprocessing(
        item_list,
        model,
        processor
):
    # Prompt loading
    prompt_path = Path(__file__).parent.parent / "prompts" / "ocr_postprocessing.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    cleaned_ingredients = []

    print(f"Starting OCR Post-processing for {len(item_list)} items with Qwen3-VL...")

    for item in item_list:
        if not item or len(item.strip()) < 2:  # Skip empty or very short strings
            continue
            
        # Format prompt
        prompt = base_prompt.replace("{item}", item)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Use processor for VL model
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = processor([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        
        # Extract new tokens and decode
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        generated_text = processor.decode(output_ids, skip_special_tokens=True).strip()
        
        # Parse JSON output
        try:
            # Try to find JSON block if model chatters
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                is_food = result.get("is_food")
                name = result.get("name")
                
                if is_food and name:
                    cleaned_ingredients.append(name)
                    print(f"Processed: {item} -> Valid: {name}")
                else:
                    print(f"Processed: {item} -> Noise (filtered out)")
                
            else:
                print(f"Failed to parse JSON for item: {item}. Output: {generated_text}")
                
        except json.JSONDecodeError:
            print(f"JSON Decode Error for item: {item}. Output: {generated_text}")
        except Exception as e:
            print(f"Error processing item {item}: {e}")

    # Remove duplicates while preserving order
    final_list = list(dict.fromkeys(cleaned_ingredients))
    
    return final_list
