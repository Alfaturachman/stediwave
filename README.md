# Stetoskop Digital - Lung Sound Analysis System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.4-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Django-based web application** for analyzing respiratory/lung sounds using machine learning. This digital stethoscope system helps healthcare professionals detect respiratory conditions through audio analysis, providing pre-diagnosis with severity levels and medical recommendations.

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Audio Analysis Pipeline](#audio-analysis-pipeline)
- [API Endpoints](#api-endpoints)
- [Firebase Integration](#firebase-integration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Core Functionality

- **Audio Upload & Recording**: Upload lung sound audio files (WAV, MP3, etc.)
- **ML-Powered Analysis**: Automatic detection of lung sound types using XGBoost model
- **Sliding Window Analysis**: Segment audio for detailed time-based analysis
- **Pre-Diagnosis System**: Rule-based classification with severity levels
- **Patient Management**: Complete patient records with medical history
- **Visualization**: Auto-generated waveform and spectrogram images
- **Role-Based Access**: Different permissions for doctors/admins

### Lung Sound Classification

| Sound Type           | Medical Condition        | Severity Levels    |
| -------------------- | ------------------------ | ------------------ |
| **Normal**           | Normal breathing         | Normal             |
| **Crackles**         | Pneumonia, Edema         | Mild to Moderate   |
| **Crepitation**      | Fine crackles            | Mild to Severe     |
| **Mengi (Wheezing)** | Asthma, COPD             | Mild to Severe     |
| **Bronchial**        | Pneumonia, Consolidation | Moderate to Severe |

### Disease Mapping

- ASMA → Mengi
- BRONKITIS → Krepitasi
- COPD → Mengi
- GAGAL JANTUNG → Crackles
- NORMAL → Normal
- PNEUMONIA → Bronchial

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                      │
│                    (HTML/CSS/JS Templates)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      DJANGO BACKEND                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │Authentication│  │Patient Mgmt  │  │Audio Processing  │   │
│  │ (Firebase)   │  │  (Firestore) │  │  (librosa/scipy) │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   MACHINE LEARNING LAYER                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ XGBoost Model│  │Feature Scaler│  │ Label Encoder    │   │
│  │  (121 feats) │  │  (v6)        │  │   (v6)           │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                       DATA STORAGE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   SQLite     │  │  Firestore   │  │   Media Files    │   │
│  │(Django DB)   │  │ (NoSQL)      │  │  (audio/images)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

- **Django 5.2.4** - Web framework
- **Python 3.9+** - Programming language

### Machine Learning & Audio Processing

- **XGBoost** - Primary classification model
- **librosa** - Audio feature extraction
- **scikit-learn** - Preprocessing & scaling
- **scipy** - Signal filtering
- **soundfile** - Audio file I/O
- **matplotlib** - Visualization generation
- **joblib** - Model serialization

### Database & Authentication

- **SQLite** - Local Django database
- **Firebase Firestore** - NoSQL cloud database
- **Firebase Authentication** - User authentication

### Other Libraries

- **numpy** - Numerical computations
- **pillow** - Image processing

---

## 📦 Prerequisites

Before installation, ensure you have:

- **Python 3.9 or higher**
- **pip** (Python package manager)
- **Git**
- **Firebase Project** with Firestore enabled
- **serviceAccountKey.json** from Firebase

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd stetoskop
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install django==5.2.4
pip install firebase-admin
pip install xgboost
pip install librosa
pip install soundfile
pip install scikit-learn
pip install scipy
pip install matplotlib
pip install joblib
pip install numpy
```

**Or create a `requirements.txt` and install all at once:**

```bash
pip install -r requirements.txt
```

### Step 4: Add Firebase Credentials

Place your `serviceAccountKey.json` file in the project root directory:

```
stetoskop/
├── serviceAccountKey.json  ← Add this file
├── manage.py
└── ...
```

---

## ⚙️ Configuration

### Firebase Setup

1. **Create Firebase Project** at [Firebase Console](https://console.firebase.google.com/)
2. **Enable Firestore Database**
3. **Enable Firebase Authentication** (Email/Password)
4. **Download serviceAccountKey.json** from Project Settings → Service Accounts
5. **Create Firestore Collections**:
    - `pasien` - Patient records
    - `analisisDokter` - Analysis results
    - `users` - User roles and permissions

### Environment Variables (Optional)

Create a `.env` file for sensitive data:

```env
DJANGO_SECRET_KEY=your-secret-key-here
FIREBASE_PROJECT_ID=your-project-id
DEBUG=True
```

### Settings Configuration

Edit `stetoskop/settings.py` if needed:

```python
# Allowed hosts
ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'yourdomain.com']

# Database (default: SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

---

## 🗄️ Database Setup

### Step 1: Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 2: Create Superuser (Admin)

```bash
python manage.py createsuperuser
```

Enter username, email, and password when prompted.

### Step 3: Initialize Firestore Collections

Use Firebase Console or a Django management command to create initial collections:

**Firestore Structure:**

```
Firestore Database
├── pasien/
│   ├── {patient_id}/
│   │   ├── nama_lengkap: string
│   │   ├── tempat_lahir: string
│   │   ├── tanggal_lahir: timestamp
│   │   ├── tanggal_periksa: timestamp
│   │   ├── jenis_kelamin: string
│   │   ├── tinggi_badan: number
│   │   ├── berat_badan: number
│   │   ├── riwayat_penyakit: string
│   │   └── status: string (pending/completed)
│
├── analisisDokter/
│   ├── {analysis_id}/
│   │   ├── pasien: reference
│   │   ├── preDiagnosis: string
│   │   ├── rujukan: string
│   │   ├── catatan: string
│   │   ├── jenisSuara: string
│   │   ├── intensitasSuara: string
│   │   ├── audioFile: string (URL)
│   │   ├── spectrogramFile: string (URL)
│   │   ├── waveformFile: string (URL)
│   │   └── createdAt: timestamp
│
└── users/
    ├── {user_uid}/
        ├── email: string
        ├── role: string (doctor/admin)
        └── name: string
```

---

## 🏃 Running the Application

### Development Server

```bash
python manage.py runserver
```

Access the application at: `http://localhost:8000`

### Production Server

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn stetoskop.wsgi:application --bind 0.0.0.0:8000
```

### Access Points

- **Homepage**: `http://localhost:8000/`
- **Admin Panel**: `http://localhost:8000/admin/`
- **Login**: `http://localhost:8000/login/`

---

## 📁 Project Structure

```
stetoskop/
│
├── manage.py                          # Django management script
├── db.sqlite3                         # SQLite database
├── serviceAccountKey.json            # Firebase credentials (gitignore)
├── requirements.txt                   # Python dependencies
│
├── audio/                             # Main Django app
│   ├── __init__.py
│   ├── admin.py                       # Django admin configuration
│   ├── apps.py                        # App configuration
│   ├── forms.py                       # Django forms
│   ├── models.py                      # Database models (AudioFile, Pasien)
│   ├── views.py                       # Request handlers & business logic
│   ├── tests.py                       # Unit tests
│   ├── urls.py                        # URL routing
│   ├── lung_sound_analyzer.py         # Rule-based lung sound analysis
│   └── ml_model_loader.py             # ML model loading utilities
│
├── ml_model/                          # Trained ML models
│   ├── xgb_model_optimized_v6.pkl    # Primary XGBoost model
│   ├── scaler_v6.pkl                  # Feature scaler
│   ├── label_encoder_v6.pkl          # Label encoder
│   ├── trainedGBModel.pkl            # Gradient Boosting model
│   ├── best_model.pth                # PyTorch model
│   └── [other model versions...]
│
├── stetoskop/                         # Project settings
│   ├── __init__.py
│   ├── settings.py                    # Django settings
│   ├── urls.py                        # Root URL configuration
│   ├── wsgi.py                        # WSGI configuration
│   └── asgi.py                        # ASGI configuration
│
├── media/                             # Uploaded files & generated images
│   ├── temp/                          # Temporary files
│   ├── audio/                         # Stored audio files
│   └── [waveforms & spectrograms]
│
└── templates/                         # HTML templates (if any)
    └── audio/
        ├── dashboard.html
        ├── login.html
        ├── daftar_pasien.html
        └── [other templates...]
```

---

## 🤖 Machine Learning Models

### Primary Model: XGBoost (`xgb_model_optimized_v6.pkl`)

**Feature Extraction (121 features):**

| Feature Type       | Count   | Description                             |
| ------------------ | ------- | --------------------------------------- |
| MFCC               | 60      | Mean (20) + Std (20) + Delta (20)       |
| Chroma             | 24      | Mean (12) + Std (12)                    |
| Spectral           | 6       | Centroid, Rolloff, Bandwidth (mean+std) |
| Spectral Contrast  | 14      | Mean (7) + Std (7)                      |
| Tonnetz            | 12      | Mean (6) + Std (6)                      |
| Zero Crossing Rate | 2       | Mean + Std                              |
| RMS Energy         | 2       | Mean + Std                              |
| Tempo              | 1       | Beat tempo                              |
| **Total**          | **121** |                                         |

### Model Classes

```python
['normal', 'crackles', 'crep', 'mengi', 'bron']
```

### Alternative Models

- **Gradient Boosting**: `trainedGBModel.pkl`
- **PyTorch Models**: `best_model.pth`, `multiKernelTripletGB_v4_balanced.pth`
- **Balanced Versions**: `*_BALANCED.pkl` variants

### Model Update Guide

To update the primary model:

1. Train new model and save as `xgb_model_optimized_v7.pkl`
2. Update paths in `views.py`:
    ```python
    MODEL_XGB_PATH = os.path.join(BASE_DIR, 'ml_model', 'xgb_model_optimized_v7.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'ml_model', 'scaler_v7.pkl')
    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'ml_model', 'label_encoder_v7.pkl')
    ```
3. Ensure feature extraction matches training configuration

---

## 🔬 Audio Analysis Pipeline

### 1. Audio Loading

```python
y, sr = librosa.load(file_path, sr=22050, duration=30)
```

- Sample rate: 22050 Hz
- Max duration: 30 seconds
- Fallback to `soundfile` if librosa fails

### 2. Filtering

```python
# Bandpass filter: 80-3000 Hz
y_filtered = basic_filter(y, sr)
```

### 3. Sliding Window Analysis

```python
# Window: 1 second, Hop: 0.5 second
sliding_window_analysis(y, sr, win_dur=1.0, hop_dur=0.5)
```

- Divides audio into overlapping windows
- Predicts disease for each window
- Aggregates results using voting

### 4. Feature Extraction

```python
features = extract_features(y, sr)  # Returns 121 features
features_scaled = scaler.transform(features.reshape(1, -1))
```

### 5. Prediction

```python
pred_proba = xgb_model.predict_proba(features_scaled)[0]
pred_disease = label_encoder.classes_[np.argmax(pred_proba)]
```

### 6. Visualization

```python
waveform_path = generate_waveform(y, sr)
spectrogram_path = generate_spectrogram(y, sr)
```

### 7. Storage & Results

- Save to Firestore with patient reference
- Store audio file, waveform, spectrogram paths
- Record pre-diagnosis, confidence, recommendations

---

## 🔌 API Endpoints

### Authentication

| Method   | URL        | View          | Description                |
| -------- | ---------- | ------------- | -------------------------- |
| GET/POST | `/login/`  | `login_view`  | User login (Firebase Auth) |
| POST     | `/logout/` | `logout_view` | User logout                |

### Main Pages

| Method | URL             | View            | Description               |
| ------ | --------------- | --------------- | ------------------------- |
| GET    | `/`             | `dashboard`     | Dashboard with statistics |
| GET    | `/pasien/`      | `daftar_pasien` | Patient list              |
| GET    | `/form-pasien/` | `form_pasien`   | New patient form          |

### Audio Analysis (Examples)

| Method | URL               | Description                   |
| ------ | ----------------- | ----------------------------- |
| POST   | `/upload-audio/`  | Upload and analyze audio file |
| GET    | `/analysis/{id}/` | View analysis results         |

_Note: Check `audio/urls.py` and `audio/views.py` for complete endpoint list_

---

## 🔥 Firebase Integration

### Firestore Usage

```python
from firebase_admin import firestore
db = firestore.client()

# Save patient
db.collection('pasien').add({
    'nama_lengkap': 'John Doe',
    'tempat_lahir': 'Jakarta',
    'tanggal_lahir': datetime(1990, 1, 1),
    ...
})

# Query patients
patients = db.collection('pasien').where('status', '==', 'pending').stream()
```

### Authentication Flow

1. User logs in with Firebase Auth (email/password)
2. Verify ID token in Django:
    ```python
    decoded = firebase_auth.verify_id_token(token)
    ```
3. Fetch user role from Firestore:
    ```python
    user_doc = db.collection('users').document(uid).get()
    role = user_doc.to_dict().get('role')
    ```
4. Create Django user session:
    ```python
    user, _ = User.objects.get_or_create(username=email)
    login(request, user)
    request.session['role'] = role
    ```

---

## 🚀 Deployment

### Option 1: Traditional Server (VPS/Dedicated)

**Requirements:**

- Ubuntu 20.04+ or similar
- Python 3.9+
- PostgreSQL (optional, for production)
- Nginx + Gunicorn

**Steps:**

```bash
# 1. Clone and setup
git clone <repo-url>
cd stetoskop
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure production settings
# Edit stetoskop/settings.py:
# - DEBUG = False
# - Update ALLOWED_HOSTS
# - Configure PostgreSQL (optional)

# 3. Collect static files
python manage.py collectstatic

# 4. Setup Gunicorn
sudo apt install gunicorn
gunicorn stetoskop.wsgi:application --bind 0.0.0.0:8000

# 5. Setup Nginx as reverse proxy
sudo apt install nginx
# Configure /etc/nginx/sites-available/stetoskop
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name audio.qbyte.web.id;

    location /static/ {
        alias /path/to/stetoskop/static/;
    }

    location /media/ {
        alias /path/to/stetoskop/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Option 2: Cloud Platforms

**Heroku:**

```bash
# Create Procfile
echo "web: gunicorn stetoskop.wsgi:application" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
heroku ps:scale web=1
```

**AWS EC2:**

- Launch EC2 instance
- Install dependencies
- Follow VPS deployment steps
- Configure Security Groups (ports 80, 443)

### SSL/HTTPS

Use Let's Encrypt for free SSL:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d audio.qbyte.web.id
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Model Not Found Error**

```
Error: Model not found: /path/to/xgb_model_optimized_v6.pkl
```

**Solution:**

- Verify model files exist in `ml_model/` folder
- Check file paths in `views.py`
- Ensure models are properly committed to repository

**2. Firebase Initialization Error**

```
firebase_admin.exceptions.InvalidArgumentError: Failed to initialize a certificate credential
```

**Solution:**

- Verify `serviceAccountKey.json` exists in project root
- Check JSON file is valid and not corrupted
- Ensure Firebase project is properly configured

**3. Audio Loading Error**

```
Error: Cannot load audio file
```

**Solution:**

- Install ffmpeg: `sudo apt install ffmpeg`
- Check audio file format is supported (WAV, MP3, FLAC)
- Verify file permissions

**4. Database Migration Error**

```
django.db.utils.OperationalError: no such table
```

**Solution:**

```bash
python manage.py makemigrations
python manage.py migrate
```

**5. Port Already in Use**

```
Error: [Errno 98] Address already in use
```

**Solution:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python manage.py runserver 8001
```

### Debug Mode

Enable detailed logging in `settings.py`:

```python
DEBUG = True

LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}
```

---

## 🎯 Future Enhancements

### Short-term

- [ ] Real-time audio recording from microphone
- [ ] Batch audio file upload
- [ ] Export analysis reports (PDF)
- [ ] Advanced search & filtering for patients
- [ ] Email notifications for new analyses

### Medium-term

- [ ] Mobile app (React Native/Flutter)
- [ ] REST API with Django REST Framework
- [ ] WebSocket support for real-time streaming
- [ ] Multi-language support (English, Indonesian)
- [ ] Advanced visualization (interactive spectrograms)

### Long-term

- [ ] Integration with EMR/EHR systems
- [ ] Telemedicine features
- [ ] Deep learning model improvements
- [ ] Federated learning for model updates
- [ ] Clinical trial data collection
- [ ] FDA/medical device certification

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
    ```bash
    git checkout -b feature/amazing-feature
    ```
3. **Commit your changes:**
    ```bash
    git commit -m 'Add amazing feature'
    ```
4. **Push to the branch:**
    ```bash
    git push origin feature/amazing-feature
    ```
5. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Write docstrings for all functions
- Add tests for new features
- Update documentation

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **Librosa** for audio processing
- **XGBoost** for machine learning
- **Django** for web framework
- **Firebase** for backend services
- Medical professionals for domain expertise

---

**Last Updated**: Desember 2025
