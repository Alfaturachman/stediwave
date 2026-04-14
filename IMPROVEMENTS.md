# 📋 Improvement Checklist - Stetoskop Django Project

> **Status**: In Progress  
> **Created**: 14 April 2026  
> **Priority Order**: 🔴 Critical → 🟡 Medium → 🟢 Low

---

## 🔴 HIGH PRIORITY (Security & Critical Issues)

### 1. Security Hardening

- [ ] **1.1** Create `.gitignore` file
    - Ignore `*.pyc`, `__pycache__/`, `.env`, `venv/`, `*.sqlite3`, `media/`, `ml_model/*.pth`, `.DS_Store`
- [ ] **1.2** **IMMEDIATELY**: Remove `serviceAccountKey.json` from Git
    - Run: `git rm --cached serviceAccountKey.json`
    - Add to `.gitignore`
    - Rotate the credentials (generate new service account key)
- [ ] **1.3** Move `SECRET_KEY` to environment variable
    - Use `python-decouple` or `os.environ.get()`
    - Create `.env.example` template
    - Add `.env` to `.gitignore`
- [ ] **1.4** Fix typo in `settings.py` line 242
    - Missing comma: `'https://www.qbyte.web.id'` should have comma after it
    - Current: `'https://www.qbyte.web.id'  'http://localhost:8000'`
    - Should be: `'https://www.qbyte.web.id', 'http://localhost:8000'`
- [ ] **1.5** Review `@csrf_exempt` usage in `views.py`
    - Line ~1162: `logout_view` has `@csrf_exempt` - potential security risk
    - Implement CSRF token in logout form instead

### 2. Dependencies Management

- [ ] **2.1** Create `requirements.txt`
    ```
    Django==4.2+
    librosa==0.10.x
    numpy==1.26.x
    scikit-learn==1.5.x
    torch==2.4.x
    scapy==1.0.x
    soundfile==0.12.x
    firebase-admin==6.x
    ```
- [ ] **2.2** Create `requirements-dev.txt` for development tools
    ```
    -r requirements.txt
    pytest-django
    black
    flake8
    django-debug-toolbar
    ```
- [ ] **2.3** Add `pip freeze > requirements.txt` command to README

### 3. URL Configuration

- [ ] **3.1** Create `audio/urls.py`
    - Move all audio-related routes from `stetoskop/urls.py` to `audio/urls.py`
    - Use `app_name = 'audio'` for namespacing
- [ ] **3.2** Include app URLs in root `stetoskop/urls.py`
    ```python
    urlpatterns = [
        path('', include('audio.urls')),
        # ... other routes
    ]
    ```
- [ ] **3.3** Remove or clarify `stetoskop/project/urls.py`
    - File currently unused and incomplete
    - Either delete or document its purpose

### 4. Settings.py Cleanup

- [ ] **4.1** Move Firebase initialization out of `settings.py`
    - Current: Firebase initialized directly in settings (lines 268-276)
    - Move to `audio/apps.py` in `AudioConfig.ready()` method
    - Or create `audio/services/firebase.py`
- [ ] **4.2** Add `django-csp` to `INSTALLED_APPS` if using CSP headers
    - Currently has CSP variables but middleware not installed
    - Either install `django-csp` or remove CSP variables
- [ ] **4.3** Configure `TEMPLATES['DIRS']` explicitly
    ```python
    'DIRS': [
        BASE_DIR / 'templates',
        BASE_DIR / 'audio' / 'templates',
    ],
    ```
- [ ] **4.4** Consider splitting `settings.py` for different environments
    - `settings/base.py` - common settings
    - `settings/development.py` - DEBUG=True
    - `settings/production.py` - DEBUG=False

---

## 🟡 MEDIUM PRIORITY (Architecture & Maintainability)

### 5. Views.py Refactoring

- [ ] **5.1** Split `audio/views.py` (currently ~1290 lines)
    - Create `audio/views/auth_views.py` - login, register, logout
    - Create `audio/views/analysis_views.py` - record_audio, analyze_result
    - Create `audio/views/api_views.py` - AJAX endpoints
- [ ] **5.2** Extract ML logic from views
    - Create `audio/services/ml_service.py`
    - Move: `lung_sound_analyzer.py` calls, model loading
    - Views should be thin, delegate to services
- [ ] **5.3** Extract Firebase operations
    - Create `audio/services/firestore_service.py`
    - Centralize Firestore reads/writes
    - Add error handling and retries
- [ ] **5.4** Create utility functions
    - Move `verify_firebase_token` to `audio/utils/auth.py`
    - Move temp file cleanup logic to `audio/utils/files.py`
    - Move response formatting to `audio/utils/responses.py`

### 6. Database Models

- [ ] **6.1** Decide: Django ORM OR Firestore (not both)
    - If using Firestore: Remove `AudioFile` and `Pasien` models
    - If using Django: Migrate Firestore data to Django models
    - Document the decision in README
- [ ] **6.2** If keeping Django models, add audit fields
    ```python
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    ```
- [ ] **6.3** Add proper model relationships
    - Link `AudioFile` to `Pasien` with `ForeignKey`
    - Add `related_name` for reverse queries
- [ ] **6.4** Add model validation and `__str__` methods
- [ ] **6.5** Create Django admin customization
    - Register models with `list_display`, `search_fields`, `list_filter`

### 7. Error Handling & Logging

- [ ] **7.1** Add proper logging configuration
    ```python
    LOGGING = {
        'version': 1,
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'debug.log',
            },
        },
        'loggers': {
            'audio': {
                'handlers': ['file'],
                'level': 'DEBUG',
            },
        },
    }
    ```
- [ ] **7.2** Replace `print()` statements with `logging.debug()`
    - `views.py` has multiple `print()` for debugging
    - Use Python logging module instead
- [ ] **7.3** Add custom error pages
    - `templates/404.html`
    - `templates/500.html`
    - `templates/403.html`
- [ ] **7.4** Implement proper exception handling in views
    - Catch specific exceptions, not bare `except:`
    - Return appropriate HTTP status codes

### 8. Testing

- [ ] **8.1** Set up testing framework
    - Install `pytest-django`
    - Create `conftest.py`
    - Configure `pytest.ini` or `pyproject.toml`
- [ ] **8.2** Write model tests
    - Test `Pasien` model creation
    - Test `AudioFile` model creation
- [ ] **8.3** Write view tests
    - Test authentication flow
    - Test audio upload and analysis
    - Test API endpoints
- [ ] **8.4** Write ML integration tests
    - Test model loading
    - Test prediction pipeline with sample data
- [ ] **8.5** Add test coverage reporting
    - Aim for >70% coverage

---

## 🟢 LOW PRIORITY (Enhancements & Polish)

### 9. Static Files Management

- [ ] **9.1** Create static files structure
    ```
    audio/
    ├── static/
    │   ├── css/
    │   │   ├── base.css
    │   │   ├── analysis.css
    │   │   └── dashboard.css
    │   ├── js/
    │   │   ├── audio_recorder.js
    │   │   ├── analysis.js
    │   │   └── dashboard.js
    │   └── images/
    ```
- [ ] **9.2** Configure static files settings
    ```python
    STATIC_URL = '/static/'
    STATICFILES_DIRS = [BASE_DIR / 'static']
    STATIC_ROOT = BASE_DIR / 'staticfiles'  # For production
    ```
- [ ] **9.3** Add manifest file for static assets
- [ ] **9.4** Consider CDN for heavy assets (Bootstrap, etc.)

### 10. ML Model Management

- [ ] **10.1** Implement model versioning
    - Add version tracking in `ml_model/`
    - Create `ml_model/v1/`, `ml_model/v2/`, etc.
    - Store version in Firestore for analysis records
- [ ] **10.2** Add model validation on load
    - Check model file integrity
    - Validate expected input shape
    - Log model version on initialization
- [ ] **10.3** Create model update script
    - Management command to download/update models
    - `python manage.py update_ml_model`
- [ ] **10.4** Add fallback mechanism
    - If model fails to load, show graceful error
    - Don't crash the entire application

### 11. Management Commands

- [ ] **11.1** Create cleanup command

    ```bash
    python manage.py cleanup_temp_files
    ```

    - Remove old temp audio files
    - Clear expired sessions

- [ ] **11.2** Create backup command

    ```bash
    python manage.py backup_firestore
    ```

    - Export Firestore data to JSON
    - Schedule with cron job

- [ ] **11.3** Create health check command

    ```bash
    python manage.py check_health
    ```

    - Verify ML model loaded
    - Test Firebase connection
    - Check disk space

### 12. API Documentation

- [ ] **12.1** Add API endpoint documentation
    - Use Postman collection or OpenAPI/Swagger
    - Document all `/api/` endpoints
    - Include request/response examples
- [ ] **12.2** Add code examples
    - How to call API from frontend
    - Authentication flow
    - Audio recording implementation
- [ ] **12.3** Generate API docs automatically
    - Consider `drf-spectacular` if migrating to DRF

### 13. Performance Optimization

- [ ] **13.1** Implement caching
    - Cache ML model predictions for identical inputs
    - Cache template fragments
    - Use Redis for session storage in production
- [ ] **13.2** Optimize audio file handling
    - Stream large files instead of loading into memory
    - Implement chunked uploads for large audio files
- [ ] **13.3** Add database query optimization
    - Use `select_related()` and `prefetch_related()`
    - Add database indexes on frequently queried fields
- [ ] **13.4** Profile and optimize view functions
    - Use Django Debug Toolbar
    - Identify N+1 queries

### 14. Deployment Preparation

- [ ] **14.1** Create `Dockerfile`
    ```dockerfile
    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["gunicorn", "stetoskop.wsgi:application"]
    ```
- [ ] **14.2** Create `docker-compose.yml`
    - Django app service
    - PostgreSQL service (optional)
    - Redis service (optional)
- [ ] **14.3** Set up production WSGI server
    - Install `gunicorn` (Linux/Mac) or `waitress` (Windows)
    - Don't use `runserver` in production
- [ ] **14.4** Configure production settings
    - `DEBUG = False`
    - Proper `ALLOWED_HOSTS`
    - HTTPS enforcement
    - Database connection pooling
- [ ] **14.5** Create CI/CD pipeline
    - GitHub Actions or GitLab CI
    - Run tests on push
    - Auto-deploy to staging

### 15. Code Quality

- [ ] **15.1** Set up code formatter
    ```bash
    black audio/ stetoskop/
    isort audio/ stetoskop/
    ```
- [ ] **15.2** Add linter configuration
    - Create `.flake8` or `.pylintrc`
    - Configure in pre-commit hook
- [ ] **15.3** Add type hints
    - Start with new code
    - Gradually add to existing functions
- [ ] **15.4** Set up pre-commit hooks
    ```yaml
    repos:
        - hooks:
              - id: black
              - id: flake8
    ```
- [ ] **15.5** Add code comments and docstrings
    - Document complex ML logic
    - Add function docstrings with Args/Returns

### 16. Documentation

- [ ] **16.1** Add architecture decision records (ADRs)
    - Why Firestore over Django ORM?
    - Why real-time WebSocket analysis?
    - Why Firebase Auth?
- [ ] **16.2** Create developer onboarding guide
    - Step-by-step setup instructions
    - Common issues and solutions
    - Code walkthrough
- [ ] **16.3** Add troubleshooting FAQ
    - ML model not loading
    - Firebase connection issues
    - Audio recording problems
- [ ] **16.4** Document API changes
    - Maintain CHANGELOG.md
    - Version API endpoints

---

## 📊 Progress Tracker

| Category        | Total Tasks | Completed | Percentage |
| --------------- | ----------- | --------- | ---------- |
| High Priority   | 15          | 0         | 0%         |
| Medium Priority | 20          | 0         | 0%         |
| Low Priority    | 35          | 0         | 0%         |
| **TOTAL**       | **70**      | **0**     | **0%**     |

---

## 🎯 Quick Start: First 5 Steps

If you're overwhelmed, start with these critical fixes:

1. **Create `.gitignore`** and remove `serviceAccountKey.json` from Git

    ```bash
    echo "serviceAccountKey.json" >> .gitignore
    git rm --cached serviceAccountKey.json
    git commit -m "Remove service account key from Git"
    ```

2. **Generate new Firebase service account key** (since old one is exposed)

3. **Fix CSRF_TRUSTED_ORIGINS typo** in `settings.py` line 242

4. **Create `requirements.txt`**:

    ```bash
    pip freeze > requirements.txt
    ```

5. **Move SECRET_KEY to environment variable**:
    ```python
    # settings.py
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fallback-for-dev-only')
    ```

---

## 📝 Notes

- **Estimated total effort**: ~40-60 hours for all items
- **Recommended timeline**: 2-3 weeks (focus on High Priority first)
- **Risk level**: High Priority items should be addressed ASAP (especially security)
- **Breaking changes**: Items 5 (Views refactoring) and 6 (Database models) may require testing and migration

---

## 🤝 Contributing

When working on improvements:

1. Create a branch: `git checkout -b fix/issue-description`
2. Make changes following code style guidelines
3. Write/update tests
4. Submit pull request with description
5. Update this checklist

---

_Last updated: 14 April 2026_
