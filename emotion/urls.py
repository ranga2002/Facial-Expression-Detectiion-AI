from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),         # Homepage
    path('analyze/', views.index, name='analyze'),  # Emotion analysis page
    path('about/', views.about, name='about'),  # About page
    path('chat-reply/', views.chat_reply, name='chat_reply'),  # Chat reply endpoint
]