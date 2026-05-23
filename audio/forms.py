from django import forms
from .models import AudioFile, Pasien

class AudioForm(forms.ModelForm):
    class Meta:
        model = AudioFile
        fields = ['judul', 'file']

class PasienForm(forms.ModelForm):
    class Meta:
        model = Pasien
        fields = ['nama_lengkap', 'tempat_lahir', 'tanggal_lahir', 'tanggal_periksa',
                  'jenis_kelamin', 'tinggi_badan', 'berat_badan', 'riwayat_penyakit']