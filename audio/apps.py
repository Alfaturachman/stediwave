from django.apps import AppConfig
import os
import firebase_admin
from firebase_admin import credentials


class AudioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'audio'
    
    def ready(self):
        """
        Initialize Firebase when the app is ready.
        This ensures Firebase is initialized before any views use it.
        """
        # Avoid re-initialization in development server (autoreload)
        if not firebase_admin._apps:
            try:
                # Path to serviceAccountKey.json
                cred_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'serviceAccountKey.json'
                )
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("✓ Firebase initialized successfully")
            except Exception as e:
                print(f"⚠ Firebase initialization warning: {e}")
