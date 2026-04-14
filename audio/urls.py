"""
URL configuration for audio app.
All audio-related routes are organized here.
"""
from django.urls import path
from audio import views

app_name = 'audio'

urlpatterns = [
    # Dashboard & Authentication
    path('', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Patient Management
    path('tambah-pasien/', views.tambah_pasien, name='tambah_pasien'),
    path('periksa-pasien/', views.upload_audio, name='upload_audio'),
    path('daftar-pasien/', views.daftar_pasien, name='daftar_pasien'),
    path('riwayat-pasien/<str:patient_id>/', views.riwayat_pasien, name='riwayat_pasien'),
    path('riwayat-pasien/delete/<str:exam_id>/', views.delete_riwayat, name='delete_riwayat'),
    
    # Analysis
    path('analyze_lung_sound_enhanced/', views.analyze_lung_sound_enhanced, name='analyze_lung_sound_enhanced'),
]
