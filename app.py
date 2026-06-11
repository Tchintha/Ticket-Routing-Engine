import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic Schemas
class TicketRequest(BaseModel):
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    text: str = Field(..., description="The content of the support ticket")

class RoutingResponse(BaseModel):
    ticket_id: str
    assigned_category: str
    confidence_score: float
    assigned_priority: str

# Model storage
models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading the serialized model into memory once on startup.
    """
    model_path = "routing_models.joblib"
    try:
        logger.info(f"Loading models from {model_path}...")
        loaded_artifacts = joblib.load(model_path)
        models["category"] = loaded_artifacts["category_model"]
        models["priority"] = loaded_artifacts["priority_model"]
        logger.info("Models loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model file {model_path} not found. Please run train.py first.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    yield
    # Clean up if necessary
    models.clear()

app = FastAPI(
    title="Real-Time B2B Support Ticket Categorization & Priority Routing Engine",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/route-ticket", response_model=RoutingResponse)
async def route_ticket(request: TicketRequest):
    """
    Endpoint to categorize and prioritize support tickets.
    """
    if "category" not in models or "priority" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        text = request.text

        # Category Prediction
        # predict_proba returns [ [prob_cat1, prob_cat2, ...] ]
        category_probs = models["category"].predict_proba([text])[0]
        category_classes = models["category"].classes_
        max_prob_idx = np.argmax(category_probs)

        assigned_category = category_classes[max_prob_idx]
        confidence_score = float(category_probs[max_prob_idx])

        # Priority Prediction
        assigned_priority = models["priority"].predict([text])[0]

        # Fallback/Shadow Logic
        # If confidence_score < 0.70, route to Manual Review
        if confidence_score < 0.70:
            logger.info(f"Low confidence ({confidence_score:.2f}) for ticket {request.ticket_id}. Routing to Manual Review.")
            assigned_category = "Manual Review"

        return RoutingResponse(
            ticket_id=request.ticket_id,
            assigned_category=assigned_category,
            confidence_score=round(confidence_score, 4),
            assigned_priority=assigned_priority
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": "category" in models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
