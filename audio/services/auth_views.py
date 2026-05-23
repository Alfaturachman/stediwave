"""
Authentication Views - Login, logout, and user management
"""

import json
import logging
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import redirect, render
from django.http import JsonResponse
from firebase_admin import auth as firebase_auth
from audio.services.firestore_service import firestore_service
from audio.utils.auth import anonymous_required

logger = logging.getLogger(__name__)


@anonymous_required
def login_view(request):
    """Handle Firebase authentication and Django login"""
    if request.method != 'POST':
        return render(request, 'audio/login.html')
    
    try:
        data = json.loads(request.body)
        token = data.get('token')
        
        decoded = firebase_auth.verify_id_token(token)
        email = decoded['email']
        uid = decoded['uid']
        
        # Get user data from Firestore
        user_doc = firestore_service.db.collection('users').document(uid).get()
        role = user_doc.to_dict().get('role')
        
        # Create or get Django user
        user, _ = User.objects.get_or_create(
            username=email,
            defaults={'email': email}
        )
        
        login(request, user)
        
        # Save role to session
        request.session['role'] = role
        
        return JsonResponse({'success': True})
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JsonResponse({'error': str(e)}, status=400)


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        request.session.flush()
        return JsonResponse({'success': True})

    return JsonResponse({'success': False}, status=400)


def user_role(request):
    """
    Context processor to get user role from Firestore
    """
    role = None
    
    if request.user.is_authenticated:
        try:
            email = request.user.email
            
            # Query Firestore by email
            role = firestore_service.get_user_role(email)
        except Exception as e:
            logger.error(f"Error getting role: {e}")
            role = None
    
    return {'user_role': role}
