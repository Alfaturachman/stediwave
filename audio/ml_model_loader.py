import logging

logger = logging.getLogger(__name__)

GB_MODEL_PATH = None
gbModel = None
classFolders = None

logger.warning(
    'ml_model_loader.py is deprecated. ML models are loaded via audio.services.ml_service'
)
