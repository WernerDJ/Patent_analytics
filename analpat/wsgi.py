"""
WSGI config for analpat project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "analpat.settings")

application = get_wsgi_application()
project_root = Path(__file__).resolve().parent.parent
media_root = str(project_root / 'media')
static_root = str(project_root / 'staticfiles')

application = WhiteNoise(application)
application.add_files(static_root, prefix='static')
application.add_files(media_root, prefix='media')
