import io
from PIL import Image
from typing import Tuple

from TeXOCR.model import TeXOCRWrapper
from TeXOCR.utils import load_config, process_output
from TeXOCR.tokenizer import RegExTokenizer

# Load model (wrapper class handles loading of weights etc.)
config = load_config('config/final_config.yml')
model = TeXOCRWrapper(config)
tokenizer = RegExTokenizer()
tokenizer.load(config['tokenizer_path'])

def predict_latex(image_bytes: bytes) -> Tuple[list, str, list]:
    """Get the LaTeX prediction for an image."""
    # Preprocess image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # Predict LaTeX
    tokens, latex = model(image)

    # Get the LaTeX prediction as a list of decoded tokens
    str_tokens = tokenizer.decode_list(tokens)

    return tokens, latex, str_tokens