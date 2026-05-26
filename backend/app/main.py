import os
import sys
import time
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.pipelines.inference import ScamShieldInference
from ml.utils.logger import setup_logger

# Setup local logger for FastAPI
logger = setup_logger("fastapi_backend")

# Initialize FastAPI App
app = FastAPI(
    title="ScamShield AI API",
    description="Unified Production-grade AI Platform for Scam SMS, Phishing URLs, and Malicious Text Detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Instance
inference_engine = None

@app.on_event("startup")
def startup_event():
    """
    Load machine learning models on startup.
    """
    global inference_engine
    logger.info("Starting up FastAPI application...")
    inference_engine = ScamShieldInference()
    logger.info("FastAPI application started and models loaded.")

# Request & Response Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw SMS, Email text, or URL string to analyze.", example="Congratulations! You won a free gift card. Claim here: http://scam.com")

class PredictResponse(BaseModel):
    input_type: str = Field(..., description="Type of input detected (TEXT or URL).", example="TEXT")
    prediction: str = Field(..., description="Classification result (SCAM, PHISHING, or SAFE).", example="SCAM")
    label: int = Field(..., description="Numeric label (1 for scam/phish, 0 for legitimate).", example=1)
    confidence: float = Field(..., description="Model confidence score (probability).", example=0.982)
    details: str = Field(..., description="Information about the prediction.", example="Cleaned tokens: 'congratul won free gift'")

class BatchPredictRequest(BaseModel):
    inputs: List[str] = Field(..., description="List of strings (text/URLs) to analyze.", example=["Free prize!", "https://google.com"])

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_processed: int
    scam_count: int

# Middleware for request/response logging & latency tracking
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Extract request information
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Incoming request: {request.method} {request.url.path} from IP: {client_ip}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed request: {request.method} {request.url.path} | Status: {response.status_code} | Latency: {process_time:.4f}s")
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Failed request: {request.method} {request.url.path} | Error: {str(e)} | Latency: {process_time:.4f}s", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred in the server inference pipeline."}
        )

# API Endpoints
@app.get("/", tags=["Metadata"])
def read_root():
    """
    Returns general metadata about the ScamShield AI service.
    """
    return {
        "app_name": "ScamShield AI",
        "description": "Unified cyber-security scam and phishing detection engine.",
        "status": "healthy",
        "endpoints": {
            "health_check": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """
    Checks the status of the server and verifies that the ML models are fully loaded.
    """
    global inference_engine
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
        
    models_loaded = (
        inference_engine.text_model is not None and
        inference_engine.url_model is not None and
        inference_engine.text_vectorizer is not None
    )
    
    if not models_loaded:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "degraded",
                "message": "API is running but some ML models failed to load. Please check logs.",
                "details": {
                    "text_model_loaded": inference_engine.text_model is not None,
                    "url_model_loaded": inference_engine.url_model is not None,
                    "text_vectorizer_loaded": inference_engine.text_vectorizer is not None
                }
            }
        )
        
    return {
        "status": "healthy",
        "message": "All models loaded and inference engine ready."
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(payload: PredictRequest):
    """
    Classify a single input string (SMS, email, or URL).
    Automatically detects input type and routes it to the correct stacking ensemble.
    """
    global inference_engine
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not ready."
        )
        
    try:
        res = inference_engine.predict(payload.text)
        if res.get("prediction") == "ERROR":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=res.get("details", "Prediction failed.")
            )
        return res
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline error: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictRequest):
    """
    Classify a batch of input strings.
    Partitions URLs and Text items, performs inference, and aggregates statistics.
    """
    global inference_engine
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not ready."
        )
        
    results = []
    scam_count = 0
    
    for item in payload.inputs:
        try:
            res = inference_engine.predict(item)
            results.append(res)
            if res.get("label") == 1:
                scam_count += 1
        except Exception as e:
            logger.error(f"Error during batch item processing: {e}")
            results.append({
                "input_type": "UNKNOWN",
                "prediction": "ERROR",
                "label": -1,
                "confidence": 0.0,
                "details": f"Failed: {str(e)}"
            })
            
    return {
        "results": results,
        "total_processed": len(payload.inputs),
        "scam_count": scam_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
