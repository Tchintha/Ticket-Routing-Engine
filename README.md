# Real-Time B2B Support Ticket Categorization & Priority Routing Engine

This repository contains a production grade, scalable solution for categorizing and prioritizing B2B support tickets using Machine Learning and FastAPI.

## Tech Stack
- **Python 3.11+**
- **FastAPI**: High-performance web framework with async support.
- **Scikit-learn**: TF-IDF Vectorization + Logistic Regression classification.
- **Pydantic v2**: Robust data validation.
- **Joblib**: Model serialization.
- **Pytest**: Comprehensive testing suite.
- **Docker**: Production-optimized containerization.

## Project Structure
- `app.py`: FastAPI application with lifespan model loading and routing logic.
- `train.py`: Training script for generating synthetic data and serializing model pipelines.
- `test_main.py`: Async test suite using Pytest and HTTPX.
- `requirements.txt`: Pinned production dependencies.
- `Dockerfile`: Multi-worker production configuration.

## Setup & Execution

### 1. Local Environment Setup
Create a virtual environment and install dependencies:
```bash
# Setup instructions
pip install -r requirements.txt
```

### 2. Model Training
Generate the synthetic dataset and train the classification pipelines:
```bash
python train.py
```
This will create a `routing_models.joblib` file.

### 3. Running the Server
Start the FastAPI server locally:
```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`.

### 4. Running Tests
Execute the test suite to verify implementation and logic:
```bash
pytest test_main.py
```

### 5. Docker Deployment
You can Build and run the production-optimized container:
```bash
docker build -t ticket-routing-engine .
docker run -p 8000:8000 ticket-routing-engine
```

## API Endpoints

### POST `/api/v1/route-ticket`
Routes a support ticket to a category and assigns a priority.

**Request Body:**
```json
{
  "ticket_id": "TKT-100",
  "text": "The API returns a 500 error when fetching reports"
}
```

**Response:**
```json
{
  "ticket_id": "TKT-100",
  "assigned_category": "Technical",
  "confidence_score": 0.9542,
  "assigned_priority": "High"
}
```

**Fallback Logic:**
If the `confidence_score` is below 0.70, the `assigned_category` is automatically set to `"Manual Review"`.

### GET `/health`
Returns the health status of the application and model loading state.
