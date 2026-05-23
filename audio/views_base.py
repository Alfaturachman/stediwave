import logging

from django.shortcuts import redirect, render
from firebase_admin import firestore

from audio.services.firestore_service import firestore_service

from .forms import AudioForm, PasienForm

logger = logging.getLogger(__name__)
db = firestore.client()


def tambah_pasien(request):
    if request.method == 'POST':
        form = PasienForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            data = firestore_service.decimal_datetime_data(data)
            data["status"] = "pending"
            db.collection("pasien").add(data)
            return redirect('beranda')
        else:
            logger.warning("Form invalid: %s", form.errors)
    else:
        form = PasienForm()
    return render(request, 'audio/form.html', {'form': form})


def daftar_audio(request):
    """List all audio files"""
    from .models import AudioFile
    audios = AudioFile.objects.all()
    return render(request, 'audio/daftar_audio.html', {'audios': audios})


def diagnosis_detail(request):
    """Diagnosis detail page"""
    if request.method == 'GET':
        return render(request, 'audio/diagnosis_detail.html')
