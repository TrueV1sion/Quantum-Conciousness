from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.debug("Serving index.html")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.debug(f"Received chat request: {request.message}")
    try:
        # Simple echo response for testing
        response = {
            "response": {
                "text": f"Echo: {request.message}",
                "confidence": 1.0,
                "processing_time": 0.1,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "request_length": len(request.message)
                }
            }
        }
        logger.debug(f"Sending response: {response}")
        return response
    except Exception as e:
        error_msg = f"Error processing chat request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.error(traceback.format_exc())
    return HTMLResponse(
        content=f"Internal Server Error: {str(exc)}",
        status_code=500
    )

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="debug")
