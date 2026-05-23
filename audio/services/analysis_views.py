"""
Analysis Views - Audio analysis, upload, and batch processing
"""

import os
import logging
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from firebase_admin import firestore
from audio.services.ml_service import ml_service
from audio.services.firestore_service import firestore_service
from audio.utils.file_utils import save_uploaded_file_temp, save_uploaded_file, load_audio_file, cleanup_temp_file

logger = logging.getLogger(__name__)


def analyze_lung_sound_enhanced(request):
    """Main analysis endpoint - analyzes lung sound audio"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if not request.FILES.get("audio_file"):
        return JsonResponse({"error": "No audio file"}, status=400)
    
    temp_file_path = None
    
    try:
        file = request.FILES["audio_file"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎵 Processing: {file.name}")
        logger.info(f"{'='*60}")
        
        temp_file_path = save_uploaded_file_temp(file)
        
        y, sr = load_audio_file(temp_file_path)
        audio_duration = len(y) / sr
        logger.info(f"✓ Loaded: {audio_duration:.2f}s")
        
        y = ml_service.basic_filter(y, sr)
        
        logger.info(f"\n📊 Starting analysis (121 features)...")
        analysis_result = ml_service.sliding_window_analysis(y, sr, win_dur=1.0, hop_dur=0.5)
        
        windows = analysis_result.get('windows', [])
        primary_disease = analysis_result.get('primary_disease', 'Unknown')
        primary_sound = analysis_result.get('primary_sound', 'Unknown')
        primary_confidence = analysis_result.get('primary_confidence', 0.0)
        disease_distribution = analysis_result.get('disease_distribution', {})
        
        logger.info(f"\n🎯 RESULTS:")
        logger.info(f"   Disease: {primary_disease} ({primary_confidence:.1%})")
        logger.info(f"   Sound: {primary_sound}")
        logger.info(f"   Windows: {len(windows)}")
        
        waveform_path = ml_service.generate_waveform(y, sr)
        spectrogram_path = ml_service.generate_spectrogram(y, sr)
        
        severity = 'Ringan'
        if primary_confidence > 0.7:
            severity = 'Berat'
        elif primary_confidence > 0.5:
            severity = 'Sedang'
        
        response_data = {
            "status": "success",
            "waveform_url": os.path.join(settings.MEDIA_URL, "temp", os.path.basename(waveform_path)),
            "spectrogram_url": os.path.join(settings.MEDIA_URL, "temp", os.path.basename(spectrogram_path)),
            
            "sliding_window_results": windows,
            
            "disease_detection": {
                "primary_disease": primary_disease,
                "confidence": f"{primary_confidence * 100:.1f}%",
                "confidence_raw": float(primary_confidence),
                "disease_distribution": disease_distribution
            },
            
            "sound_classification": {
                "detected_sound": primary_sound,
                "confidence": f"{primary_confidence * 100:.1f}%",
                "confidence_raw": float(primary_confidence),
                "mapping_rule": f"{primary_disease} → {primary_sound}"
            },
            
            "pre_diagnosis": {
                'primary_disease': primary_disease,
                'primary_sound': primary_sound,
                'primary_confidence': f"{primary_confidence * 100:.1f}%",
                'severity': severity,
                'urgency': 'Perlu evaluasi medis' if primary_disease != 'NORMAL' else 'Normal',
                'rekomendasi': [
                    'Konsultasi dokter spesialis paru',
                    f'Indikasi suara: {primary_sound}'
                ] if primary_disease != 'NORMAL' else ['Kondisi normal'],
                'color_code': 'danger' if severity == 'Berat' else 'warning' if severity == 'Sedang' else 'success'
            },
            
            "technical_info": {
                'audio_duration': f'{audio_duration:.1f}s',
                'sample_rate': f'{sr} Hz',
                'windows_analyzed': len(windows),
                'model_type': 'XGBoost V6',
                'total_features': 121,
                'accuracy': '100% (on test set)'
            },
            
            "gradient_boosting": {
                "model": "XGBoost V6",
                "disease": primary_disease,
                "sound": primary_sound,
                "confidence": float(primary_confidence),
                "windows": windows
            }
        }
        
        logger.info(f"\n✓ Response ready")
        
        return JsonResponse(response_data)
    
    except Exception as e:
        logger.error(f"\n✗ ERROR: {str(e)}", exc_info=True)
        
        return JsonResponse({
            "error": f"Error: {str(e)}",
            "status": "error",
            "details": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        }, status=500)
    
    finally:
        cleanup_temp_file(temp_file_path)


def upload_audio(request):
    """
    Improved upload handler:
    - robust gets from request.POST / request.FILES
    - ensures no null values are stored in Firestore (use empty string instead)
    - if waveform/spectrogram not uploaded, generate from audio file
    - updates pasien.status -> completed
    """
    if request.method == 'POST':
        try:
            # Reliable reads: try POST first, fallback to form names often used in front-end
            pasien_id = request.POST.get("pasienSelect") or request.POST.get("pasien_id") or request.POST.get("pasien")
            
            # CRITICAL: Get data from hidden inputs (populated by JavaScript)
            jenis_suara = request.POST.get("jenisSuara") or ""
            intensitas = request.POST.get("intensitasSuara") or ""
            pre_diagnosis = request.POST.get("preDiagnosis") or ""
            rujukan = request.POST.get("rujukanFasilitasKesehatan") or ""
            analisis_dokter = request.POST.get("analisisDokter") or ""
            
            # Files (may be None)
            audio_file = request.FILES.get("audio_file") or request.FILES.get("file") or None
            waveform_file = request.FILES.get("waveformImage") or request.FILES.get("waveform_file") or None
            spectrogram_file = request.FILES.get("spectrogramImage") or request.FILES.get("spectrogram_file") or None
            
            # Log received data for debugging
            logger.info(f"Received upload request:")
            logger.info(f"- Pasien ID: {pasien_id}")
            logger.info(f"- Jenis Suara: {jenis_suara}")
            logger.info(f"- Intensitas: {intensitas}")
            logger.info(f"- Pre-diagnosis length: {len(pre_diagnosis)}")
            logger.info(f"- Rujukan: {rujukan}")
            
            if not pasien_id:
                return render(request, "audio/index.html", {"error": "Pasien harus dipilih!"})
            
            if not audio_file:
                return render(request, "audio/index.html", {"error": "File audio wajib diunggah!"})
            
            # Save uploaded audio to temp (and get filename)
            audio_filename = save_uploaded_file(audio_file, "audio")
            temp_audio_path = os.path.join(settings.MEDIA_ROOT, "temp", audio_filename)
            
            # If waveform/spectrogram were uploaded as files, save them; else generate from audio
            if waveform_file:
                waveform_filename = save_uploaded_file(waveform_file, "waveform")
            else:
                # generate waveform image from the audio file we just saved
                try:
                    y, sr = load_audio_file(temp_audio_path)
                    waveform_path = ml_service.generate_waveform(y, sr)
                    waveform_filename = os.path.basename(waveform_path)
                except Exception as e:
                    logger.warning(f"Waveform generation failed: {e}")
                    waveform_filename = ""
            
            if spectrogram_file:
                spectrogram_filename = save_uploaded_file(spectrogram_file, "spectrogram")
            else:
                try:
                    if 'y' not in locals():
                        y, sr = load_audio_file(temp_audio_path)
                    spectrogram_path = ml_service.generate_spectrogram(y, sr)
                    spectrogram_filename = os.path.basename(spectrogram_path)
                except Exception as e:
                    logger.warning(f"Spectrogram generation failed: {e}")
                    spectrogram_filename = ""
            
            # Ensure strings (not None) to avoid storing null in Firestore
            jenis_suara = jenis_suara.strip() if jenis_suara else ""
            intensitas = intensitas.strip() if intensitas else ""
            pre_diagnosis = pre_diagnosis.strip() if pre_diagnosis else ""
            rujukan = rujukan.strip() if rujukan else ""
            
            # Build document payload
            doc_payload = {
                "pasien": pasien_id,
                "audioFile": audio_filename,
                "waveformFile": waveform_filename,
                "spectrogramFile": spectrogram_filename,
                "jenisSuara": jenis_suara,
                "intensitasSuara": intensitas,
                "preDiagnosis": pre_diagnosis,
                "rujukan": rujukan,
                "analisisDokter": analisis_dokter,
                "createdAt": firestore.SERVER_TIMESTAMP
            }
            
            # Log payload before saving
            logger.info(f"Saving to Firebase with payload: {doc_payload}")
            
            # Save to Firestore
            doc_ref = firestore_service.create_analisis(doc_payload)
            
            # mark pasien status completed if exists
            firestore_service.update_pasien_status(pasien_id, "completed")
            
            # Return success page or redirect
            return render(request, "audio/upload_success.html", {
                "file": audio_filename,
                "success_message": "Data berhasil disimpan ke Firebase!"
            })
        
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return render(request, "audio/index.html", {
                "error": f"Terjadi kesalahan: {str(e)}"
            })
    
    # GET: render form with pasien list
    pasien_list = firestore_service.get_pasien_list()
    
    return render(request, 'audio/index.html', {
        'pasien_json': json.dumps(pasien_list, ensure_ascii=False)
    })


def batch_analyze_audio(request):
    """Analyze multiple audio files at once"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    audio_files = request.FILES.getlist("audio_files")
    
    if not audio_files:
        return JsonResponse({"error": "No audio files"}, status=400)
    
    results = []
    temp_paths = []
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH ANALYSIS: {len(audio_files)} files")
        logger.info(f"{'='*60}")
        
        for idx, file in enumerate(audio_files, 1):
            try:
                logger.info(f"\n[{idx}/{len(audio_files)}] Processing: {file.name}")
                
                temp_file_path = save_uploaded_file_temp(file)
                temp_paths.append(temp_file_path)
                
                y, sr = load_audio_file(temp_file_path)
                y = ml_service.basic_filter(y, sr)
                
                pred_disease, pred_confidence, pred_sound, _ = ml_service.predict_disease(y, sr)
                
                results.append({
                    'filename': file.name,
                    'status': 'success',
                    'disease': pred_disease,
                    'confidence': f"{pred_confidence * 100:.1f}%",
                    'sound': pred_sound
                })
                
                logger.info(f"✓ {pred_disease} ({pred_confidence*100:.1f}%) → {pred_sound}")
            
            except Exception as e:
                logger.error(f"✗ Failed: {file.name} - {str(e)}")
                results.append({
                    'filename': file.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        return JsonResponse({
            'status': 'success',
            'total_files': len(audio_files),
            'processed': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"\n✗ BATCH ERROR: {str(e)}", exc_info=True)
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)
    
    finally:
        for path in temp_paths:
            cleanup_temp_file(path)
