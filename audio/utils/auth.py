from functools import wraps
from django.shortcuts import redirect


def anonymous_required(view_func):
    """Decorator to ensure user is not authenticated"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('beranda')
        return view_func(request, *args, **kwargs)
    return wrapper
