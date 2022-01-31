from django.urls import path

from . import views

app_name = 'masking'

urlpatterns = [
    path('', views.main, name='main'),

    path('video', views.video, name='video'),

    path('home', views.home, name='home'),
]