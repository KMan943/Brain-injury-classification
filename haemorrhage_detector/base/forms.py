from django import forms
from base.models import Patient

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['Patient_ID' , 'brain_image' , 'bone_image']    