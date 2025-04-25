import joblib
import os
import logging
from sklearn.pipeline import Pipeline
from .preprocessing import preprocess_text

logger = logging.getLogger(__name__)

class QuestionClassifier:
    def __init__(self):
        """
        Initialize the QuestionClassifier by loading the trained model.
        """
        try:
            # Get the absolute path to the model file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'models', 'model.pkl')
            
            logger.info(f"Attempting to load model from: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load the model
            self.model = joblib.load(model_path)
            
            # Verify the model is a pipeline
            if not isinstance(self.model, Pipeline):
                raise ValueError("Loaded model is not a sklearn Pipeline")
            
            # Verify the pipeline has the expected steps
            expected_steps = ['tfidf', 'clf']
            if not all(step in self.model.named_steps for step in expected_steps):
                raise ValueError(f"Model pipeline missing required steps. Expected: {expected_steps}")
            
            # Verify TF-IDF is fitted
            if not hasattr(self.model.named_steps['tfidf'], 'vocabulary_'):
                raise ValueError("TF-IDF vectorizer vocabulary is not fitted")
            if not hasattr(self.model.named_steps['tfidf'], 'idf_'):
                raise ValueError("TF-IDF vectorizer idf vector is not fitted")
            
            # Verify classifier is fitted
            if not hasattr(self.model.named_steps['clf'], 'coef_'):
                raise ValueError("Classifier is not fitted")
            
            # Test prediction
            test_text = "test question"
            self.model.predict([test_text])
            
            logger.info("Model loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, text: str) -> str:
        """
        Predict the category of a given question text.
        """
        try:
            # Input validation
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            if not text.strip():
                raise ValueError("Input text cannot be empty")
            
            # Preprocess the text
            processed_text = preprocess_text(text)
            
            # Make prediction
            prediction = self.model.predict([processed_text])[0]
            logger.info(f"Prediction made successfully: {prediction}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise