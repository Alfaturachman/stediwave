from django.db import models


class AudioFile(models.Model):
    judul = models.CharField(max_length=255)
    file = models.FileField(upload_to='audio/')
    created_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.judul

    class Meta:
        ordering = ['-created_at']


class Pasien(models.Model):
    JENIS_KELAMIN_CHOICES = [
        ('L', 'Laki-laki'),
        ('P', 'Perempuan'),
    ]

    nama_lengkap = models.CharField(max_length=255)
    tempat_lahir = models.CharField(max_length=100)
    tanggal_lahir = models.DateField()
    tanggal_periksa = models.DateField()
    jenis_kelamin = models.CharField(max_length=20, choices=JENIS_KELAMIN_CHOICES)
    tinggi_badan = models.DecimalField(max_digits=5, decimal_places=2)
    berat_badan = models.DecimalField(max_digits=5, decimal_places=2)
    riwayat_penyakit = models.TextField(blank=True)
    created_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.nama_lengkap

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Pasien'
