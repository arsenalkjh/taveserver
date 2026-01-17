from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import sys
import os

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from server.load_model import load_qwen3, load_qwen_vl, load_sam3, load_varco_ocr
from ocr.total_pipeline import run_total_pipeline

app = FastAPI(title="Ingredient Detection API")

# Global model storage
models = {}

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    print("Loading models...")
    
    # Load SAM model
    print("Loading SAM3...")
    models["sam"] = load_sam3()
    
    # Load VLM model
    print("Loading Qwen VL...")
    models["vlm_model"], models["vlm_processor"] = load_qwen_vl()
    
    # Load VARCO OCR
    print("Loading VARCO OCR...")
    models["ocr_model"], models["ocr_processor"] = load_varco_ocr()
    
    # Load LLM for post-processing (using Qwen3-VL-2B to save memory)
    print("Loading Qwen3 LLM...")
    models["llm_model"], models["llm_processor"] = load_qwen3()
    
    print("All models loaded successfully!")

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    """
    POST endpoint to detect ingredients from an uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with list of detected ingredients
    """
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / file.filename
        
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run the pipeline
        ingredients = run_total_pipeline(
            image_path=temp_file_path,
            sam_model=models["sam"],
            vlm_model=models["vlm_model"],
            vlm_processor=models["vlm_processor"],
            ocr_model=models["ocr_model"],
            ocr_processor=models["ocr_processor"],
            llm_model=models["llm_model"],
            llm_processor=models["llm_processor"]
        )
        
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        return JSONResponse(content={
            "success": True,
            "ingredients": ingredients,
            "count": len(ingredients)
        })
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(models) > 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
