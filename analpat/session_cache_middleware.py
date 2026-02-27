from django.utils.deprecation import MiddlewareMixin
from django.utils.cache import patch_cache_control

class SessionCacheControlMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        patch_cache_control(response, public=True, max_age=0, must_revalidate=True)
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
