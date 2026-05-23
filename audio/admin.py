from django.contrib import admin
from .models import AudioFile, Pasien

@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = ('judul', 'file')
    search_fields = ('judul',)

@admin.register(Pasien)
class PasienAdmin(admin.ModelAdmin):
    list_display = ('nama_lengkap', 'tanggal_lahir', 'tanggal_periksa', 'jenis_kelamin')
    list_filter = ('jenis_kelamin', 'tanggal_periksa')
    search_fields = ('nama_lengkap',)
