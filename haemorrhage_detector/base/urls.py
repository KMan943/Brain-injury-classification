from django.urls import path
from . import views

app_name = 'base'

urlpatterns = [
    path('' , views.predict , name="predict"),
    path('results/' , views.results , name="results")
]