
import joblib
import sys
sys.path.append('src')
from logger import setup_logger

logger = setup_logger()

def load_model(model_path='models/xgboost_model.pkl'):
    """Model-ஐ Safe-ஆ Load பண்றது"""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise
