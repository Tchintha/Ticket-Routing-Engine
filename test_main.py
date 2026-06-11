import pytest
from httpx import AsyncClient, ASGITransport
from app import app, models
import joblib
import os

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure the model is trained and loaded before tests run."""
    # Run training script if model doesn't exist
    if not os.path.exists("routing_models.joblib"):
        import subprocess
        subprocess.run(["python", "train.py"], check=True)
    
    # Normally lifespan would handle this, but for tests we might need to trigger it
    # or manually load for the models dict used in app.py
    loaded_artifacts = joblib.load("routing_models.joblib")
    models["category"] = loaded_artifacts["category_model"]
    models["priority"] = loaded_artifacts["priority_model"]

@pytest.mark.asyncio
async def test_route_ticket_success():
    """Test successful ticket routing for a clear technical issue."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/route-ticket",
            json={"ticket_id": "TKT-100", "text": "Database connection timeout in production"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_id"] == "TKT-100"
    assert "assigned_category" in data
    assert "confidence_score" in data
    assert "assigned_priority" in data
    assert data["assigned_category"] in ["Technical", "Manual Review"]

@pytest.mark.asyncio
async def test_route_ticket_low_confidence_fallback():
    """Test fallback to Manual Review for ambiguous/garbage text."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/route-ticket",
            json={"ticket_id": "TKT-999", "text": "xyz abc random text that should have low confidence"}
        )
    
    assert response.status_code == 200
    data = response.json()
    # For a small synthetic dataset, random text should likely result in low confidence
    if data["confidence_score"] < 0.70:
        assert data["assigned_category"] == "Manual Review"

@pytest.mark.asyncio
async def test_route_ticket_validation_error():
    """Test validation error for missing required fields."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/route-ticket",
            json={"ticket_id": "TKT-101"} # Missing 'text'
        )
    
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "models_loaded": True}
