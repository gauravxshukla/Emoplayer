from django.urls import path, include
from streamapp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('Emo_feed', views.Emo_feed, name='Emo_feed'),
	path('rend', views.rend, name='rend'),
    ]
