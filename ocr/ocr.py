from paddleocr import PaddleOCR
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def run_paddle_ocr(
    image_path,
    ocr_model=None,
    angle_list=[0, 90, 180, 270]
):
    """
    Run PaddleOCR on an image with multiple rotation angles.
    
    Args:
        image_path: Path to the image file
        ocr_model: PaddleOCR instance (if None, will create a new one)
        angle_list: List of rotation angles to try
    
    Returns:
        List of detected text strings
    """
    # Initialize PaddleOCR if not provided
    if ocr_model is None:
        # Use CPU to avoid CUDA conflicts with PyTorch
        ocr_model = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False)
    
    char_list = []
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Resize if needed
    h, w = image.shape[:2]
    target_size = 2304
    
    if max(w, h) < target_size:
        scaling_factor = target_size / max(w, h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Run OCR on different angles
    for angle in angle_list:
        target = image if angle == 0 else rotate_image(image, angle)
        
        # PaddleOCR expects numpy array in BGR format
        result = ocr_model.ocr(target, cls=True)
        
        # Extract text from results
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # line[1] is (text, confidence)
                    if text and len(text.strip()) > 0:
                        char_list.append(text.strip())
    
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
