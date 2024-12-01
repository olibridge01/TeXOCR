from fastapi import APIRouter, File, UploadFile
from TeXOCR.texocr_app.model import predict_latex

router = APIRouter()

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Get LaTeX prediction for an image."""
    image_bytes = await image.read()
    tokens, latex, str_tokens = predict_latex(image_bytes)
    return {"tokens": tokens, "latex": latex, "strtokens": str_tokens}