import logging
import re

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text similarly to how training data was processed.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
        
    Raises:
        ValueError: If input is invalid
    """
    try:
        # Input validation
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        # Convert to lowercase and strip whitespace
        processed = text.lower().strip()
        
        # Log the preprocessing result
        logger.debug(f"Original text: {text}")
        logger.debug(f"Processed text: {processed}")
        
        return processed
        
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        raise