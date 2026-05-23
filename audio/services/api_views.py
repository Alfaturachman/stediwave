"""
API Views - AJAX endpoints for diagnosis and data operations
"""

import json
import logging
from django.http import JsonResponse, Http404
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from firebase_admin import firestore
from audio.services.firestore_service import firestore_service
from audio.services.ml_service import ml_service

logger = logging.getLogger(__name__)


def save_diagnosis_result(request):
    """Save diagnosis result to Firestore"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            diagnosis_data = {
                'pasien_nama': data.get('pasien_nama'),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'sound_classification': data.get('sound_classification'),
                'disease_detection': data.get('disease_detection'),
                'pre_diagnosis': data.get('pre_diagnosis'),
                'technical_info': data.get('technical_info'),
                'status': 'completed'
            }
            
            doc_id = firestore_service.save_diagnosis_result(diagnosis_data)
            
            # Update pasien status if provided
            if data.get('pasien_nama'):
                pasien_query = firestore_service.db.collection('pasien').where('nama_lengkap', '==', data.get('pasien_nama')).limit(1)
                for doc in pasien_query.stream():
                    doc.reference.update({'status': 'completed'})
                    break
            
            return JsonResponse({
                'status': 'success',
                'doc_id': doc_id,
                'message': 'Hasil diagnosis berhasil disimpan'
            })
        
        except Exception as e:
            logger.error(f"Error saving diagnosis: {e}", exc_info=True)
            return JsonResponse({
                'status': 'error',
                'message': f'Gagal menyimpan diagnosis: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def get_diagnosis_history(request):
    """Get diagnosis history from Firestore"""
    try:
        results = firestore_service.get_diagnosis_history()
        
        return JsonResponse({
            'status': 'success',
            'data': results
        })
    
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


def delete_riwayat(request, exam_id):
    """Delete examination record"""
    if request.method == 'DELETE':
        try:
            firestore_service.delete_riwayat(exam_id)
            
            return JsonResponse({'success': True, 'message': 'Riwayat berhasil dihapus'})
        except Exception as e:
            logger.error(f"Error deleting examination: {e}")
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)


@login_required(login_url='login')
def dashboard(request):
    """Dashboard view with patient counts"""
    pending_count, completed_count = firestore_service.get_pasien_counts()
    
    role = request.session.get('role')
    
    context = {
        "pending_count": pending_count,
        "completed_count": completed_count,
        "role": role,
    }
    
    return render(request, "audio/dashboard.html", context)


def form_pasien(request):
    """Render patient form page"""
    return render(request, 'audio/form.html')


def daftar_pasien(request):
    """View for patient list page - FIXED VERSION"""
    try:
        # Get all patients
        pasien_list = firestore_service.get_pasien_list()
        
        logger.info(f"\n=== DAFTAR PASIEN ===")
        logger.info(f"Total pasien: {len(pasien_list)}")
        
        # Get all analysis grouped by patient
        analisis_grouped = firestore_service.get_all_analisis_grouped()
        
        logger.info(f"Total pasien dengan riwayat: {len(analisis_grouped)}")
        
        # Render template
        return render(request, "audio/daftar_pasien.html", {
            "pasien_json": json.dumps(pasien_list, ensure_ascii=False),
            "analisis_json": json.dumps(analisis_grouped, ensure_ascii=False)
        })
    
    except Exception as e:
        logger.error(f"Error in daftar_pasien: {e}")
        import traceback
        traceback.print_exc()
        
        return render(request, "audio/daftar_pasien.html", {
            "pasien_json": json.dumps([], ensure_ascii=False),
            "analisis_json": json.dumps({}, ensure_ascii=False)
        })


def riwayat_pasien(request, patient_id):
    """View for patient history page"""
    try:
        # Get patient data
        patient = firestore_service.get_pasien_by_id(patient_id)
        
        if not patient:
            raise Http404("Pasien tidak ditemukan")
        
        logger.info(f"\n=== RIWAYAT PASIEN: {patient['namaLengkap']} ===")
        
        # Get examination history
        riwayat_list = firestore_service.get_analisis_for_pasien(patient_id)
        
        logger.info(f"Total riwayat ditemukan: {len(riwayat_list)}")
        
        # Render template
        return render(request, "audio/riwayat_pasien.html", {
            "pasien_json": json.dumps(patient, ensure_ascii=False),
            "riwayat_json": json.dumps(riwayat_list, ensure_ascii=False)
        })
    
    except Http404:
        raise
    except Exception as e:
        logger.error(f"Error in riwayat_pasien: {e}")
        import traceback
        traceback.print_exc()
        raise Http404("Terjadi kesalahan saat mengambil data pasien")
