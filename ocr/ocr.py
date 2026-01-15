import torch
import numpy as np
import cv2

from PIL import Image
import re

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def run_varco_ocr(
    image_path,
    model,
    processor,
    angle_list=[0, 90, 180, 270]
):
    char_list = []

    # 1️⃣ PIL → cv2
    image_pil = Image.open(image_path).convert("RGB")
    image = pil_to_cv2(image_pil)

    # 2️⃣ 크기 계산 (cv2 기준)
    h, w = image.shape[:2]
    target_size = 2304

    if max(w, h) < target_size:
        scaling_factor = target_size / max(w, h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3️⃣ 각도별 OCR
    for angle in angle_list:
        target = image if angle == 0 else rotate_image(image, angle)

        # 🔁 cv2 → PIL (모델 입력용)
        target_pil = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_pil},
                    {"type": "text", "text": "<ocr>"},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generate_ids = model.generate(**inputs, max_new_tokens=1024)

        generate_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]

        output = processor.decode(
            generate_ids_trimmed[0],
            skip_special_tokens=False
        )

        char_list.extend(re.findall(r"<char>(.*?)</char>", output))

    return char_list



def rotate_image(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate while keeping the full canvas."""
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    rad = np.deg2rad(angle)
    new_w = int(abs(h * np.sin(rad)) + abs(w * np.cos(rad)))
    new_h = int(abs(h * np.cos(rad)) + abs(w * np.sin(rad)))
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    m[0, 2] += new_w / 2 - center[0]
    m[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(
        image_bgr,
        m,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )