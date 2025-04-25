from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from .model import QuestionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question Classifier API",
    description="API for classifying professional questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model with error handling
try:
    classifier = QuestionClassifier()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

class Question(BaseModel):
    text: str

@app.post("/predict", response_model=dict)
async def predict_category(question: Question):
    """
    Predict the category of a question.
    """
    try:
        logger.info(f"Received question: {question.text}")
        
        if not question.text.strip():
            raise HTTPException(status_code=400, detail="Question text cannot be empty")
        
        category = classifier.predict(question.text)
        logger.info(f"Predicted category: {category}")
        
        return {"category": category}
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}