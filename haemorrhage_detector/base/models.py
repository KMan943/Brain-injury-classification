from django.db import models

# Create your models here.
class Patient(models.Model):
    Patient_ID = models.CharField(max_length=64)
    brain_image = models.ImageField(upload_to='patients/brain/')
    bone_image = models.ImageField(upload_to='patients/bone/')


