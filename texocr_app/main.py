from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from TeXOCR.texocr_app.routers import predict

app = FastAPI()

# Set up static files
app.mount("/static", StaticFiles(directory="texocr_app/static"), name="static")

# Set up template rendering
templates = Jinja2Templates(directory="texocr_app/templates")

# Include routes
app.include_router(predict.router)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return templates.TemplateResponse("index.html", {"request": {}})