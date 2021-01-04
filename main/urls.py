"""textViz URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'main'

# static allows app. to serve from media root
urlpatterns = [
    path('', views.homepage, name="homepage"),
    path('info/', views.info, name="info"),
    path('logout/', views.logoutRequest, name="logout"),
    path('login/', views.loginRequest, name="login"),
    path('account/<userName>/', views.account, name="account"),
    path('register/', views.register, name="register"),
    path('display/<userName>/<int:fileKey>/', views.display, name="display"),
    # path(r'ajax/', views.updateContactDropdown, name='ajax'),
    path('experimental/', views.experimental, name="experimental"),
    path('visuals/<userName>/<int:fileKey>/<contactName>/', views.visuals, name='visuals'),
    path('analysis/<userName>/<int:fileKey>/<contactName>/', views.analysis, name='analysis')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
