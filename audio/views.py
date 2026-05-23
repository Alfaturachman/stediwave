"""
views.py - Refactored thin wrapper
All logic has been moved to modular structure:
- audio/services/auth_views.py - Authentication views
- audio/services/analysis_views.py - Analysis and upload views  
- audio/services/api_views.py - API endpoints
- audio/services/ml_service.py - ML operations
- audio/services/firestore_service.py - Firestore operations
- audio/utils/ - Utility functions
"""

# Import all views for backward compatibility
from audio.services.auth_views import (
    login_view,
    logout_view,
    user_role,
)

from audio.services.analysis_views import (
    analyze_lung_sound_enhanced,
    upload_audio,
    batch_analyze_audio,
)

from audio.services.api_views import (
    save_diagnosis_result,
    get_diagnosis_history,
    delete_riwayat,
    dashboard,
    form_pasien,
    daftar_pasien,
    riwayat_pasien,
)

# Import additional views that are still in this file
from audio.views_base import (
    tambah_pasien,
    daftar_audio,
    diagnosis_detail,
)

# Additional imports that might be needed by other parts of the app
from audio.forms import AudioForm, PasienForm
from audio.models import AudioFile, Pasien

__all__ = [
    # Auth views
    'login_view',
    'logout_view',
    'user_role',
    
    # Analysis views
    'analyze_lung_sound_enhanced',
    'upload_audio',
    'batch_analyze_audio',
    
    # API views
    'save_diagnosis_result',
    'get_diagnosis_history',
    'delete_riwayat',
    'dashboard',
    'form_pasien',
    'daftar_pasien',
    'riwayat_pasien',
    
    # Other views
    'tambah_pasien',
    'daftar_audio',
    'diagnosis_detail',
    
    # Forms and models
    'AudioForm',
    'PasienForm',
    'AudioFile',
    'Pasien',
]
