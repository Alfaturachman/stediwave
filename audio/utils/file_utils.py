"""
File utilities - Handle file operations
"""

import os
import uuid
import logging
import soundfile as sf
import numpy as np
import librosa
from django.conf import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050


def save_uploaded_file_temp(uploaded_file):
    """Save uploaded file to temp directory and return path"""
    try:
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        return temp_path
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise


def save_uploaded_file(uploaded_file, file_type="audio"):
    """Save uploaded file and return filename only"""
    try:
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_filename = f"{file_type}_{uuid.uuid4()}{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        with open(temp_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        return temp_filename
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise


def load_audio_file(file_path):
    """Load audio file with fallback mechanisms"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=30)
        return y, sr
    except Exception as e1:
        logger.warning(f"Librosa load failed: {e1}, trying soundfile...")
        try:
            data, sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE
            return data, sr
        except Exception as e2:
            logger.error(f"Cannot load audio: {e2}")
            raise


def cleanup_temp_file(file_path):
    """Remove temp file"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")
