import logging
import os

import firebase_admin
from django.apps import AppConfig
from firebase_admin import credentials

logger = logging.getLogger(__name__)


class AudioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'audio'

    def ready(self):
        if not firebase_admin._apps:
            try:
                cred_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'serviceAccountKey.json'
                )
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                logger.info('Firebase initialized successfully')
            except Exception as e:
                logger.warning('Firebase initialization warning: %s', e)
