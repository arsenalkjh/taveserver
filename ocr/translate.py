from pathlib import Path

def translate_ingredients(
    english_list,
    model,
    processor
):
    """
    Translates a list of English ingredient names to Korean one by one.
    """
    if not english_list:
        return []

    korean_list = []
    print(f"Translating {len(english_list)} ingredients one by one...")

    for item in english_list:
        if not item or not item.strip():
            continue
            
        # Specific simple prompt requested by user
        prompt = f"What is {item} in Korean? only answer one word in korean"
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Use processor's tokenizer for text-only input
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = processor.tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32, # Short output expected
        )
        
        # Extract and decode output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        translated_name = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Cleanup: remove punctuation if model adds any
        translated_name = translated_name.replace(".", "").replace(",", "").replace("!", "")
        
        if translated_name:
            print(f"  Translated: {item} -> {translated_name}")
            korean_list.append(translated_name)
    
    return list(dict.fromkeys(korean_list)) # Remove duplicates while preserving order
