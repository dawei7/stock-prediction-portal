from django.shortcuts import render
from .serializers import UserSerializer
from rest_framework import generics
from django.contrib.auth.models import User
from rest_framework import permissions

# Create your views here.

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]
