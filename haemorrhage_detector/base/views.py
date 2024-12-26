from django.shortcuts import render
from base.forms import PatientForm
import tensorflow as tf
import cv2
import numpy as np


brain_model_download_link = 'https://drive.google.com/file/d/1-5HFTklEXsAiF5ovF1hk7xNGtkTAqY3s/view?usp=sharing'
bone_model_download_link = 'https://drive.google.com/file/d/1-3QoEf5YWJhnbqvHvqpEdk0dcuwHwPz_/view?usp=sharing'

brain_model_path = r'C:\Users\kman0\Downloads\college projects\siddhi 2.0\models\Res101-brainModel.keras'
bone_model_path = r'C:\Users\kman0\Downloads\college projects\siddhi 2.0\models\Res101-boneModel.keras'
# saved_brain_model_layer = tf.keras.layers.TFSMLayer(brain_model_path, call_endpoint="serving_default")
# saved_bone_model_layer = tf.keras.layers.TFSMLayer(bone_model_path, call_endpoint="serving_default")

# brain_model = tf.keras.Sequential([saved_brain_model_layer])
# bone_model = tf.keras.Sequential([saved_bone_model_layer])

brain_model = tf.keras.models.load_model(brain_model_path)
bone_model = tf.keras.models.load_model(bone_model_path)

def make_brain_prediction(image_path):
    img = cv2.imread(image_path)
    img_array = np.array([img])
    predictions = brain_model.predict(img_array)
     
    predicted_class = np.argmax(predictions[0])  
    if predicted_class==0 :
        haemorrhage = 'Intraventricular' 
    elif predicted_class==1 :
        haemorrhage = 'Intraparenchymal'
    elif predicted_class==2 :
        haemorrhage = 'Subarachnoid'
    elif predicted_class==3 :
        haemorrhage = 'Epidural'
    elif predicted_class==4 :
        haemorrhage = 'Subdural'
    elif predicted_class==5 :
        haemorrhage = 'No'

    predicted_probability = predictions[0][predicted_class]  

    return haemorrhage, predicted_probability

def make_bone_prediction(image_path):
    img = cv2.imread(image_path)
    img_array = np.array([img])
    predictions = bone_model.predict(img_array)
    predicted_probability = predictions[0]
    prob = predicted_probability
    if predicted_probability < 0.5:
        bone_pred = "No Fracture"
        prob = 1-predicted_probability

    else:
        bone_pred = "Fracture"

    return bone_pred, prob

def predict(request):
    if request.method == "POST":
        form = PatientForm(request.POST , request.FILES)
        if form.is_valid():
            img = form.save()
            brain_image_path = img.brain_image.path
            bone_image_path = img.bone_image.path

            brain_pred_class , brain_pred_prob = make_brain_prediction(brain_image_path)
            bone_pred , bone_pred_prob = make_bone_prediction(bone_image_path)

            # print(bone_image_path)
            # print(brain_image_path)

            context = {'brain_img' : brain_image_path , 'bone_img' : bone_image_path , 'brain_pred_class' : brain_pred_class  , 'brain_pred_prob' : brain_pred_prob , 'bone_pred' : bone_pred , 'bone_pred_prob' : bone_pred_prob}
            return render(request , 'base/results.html' , context=context)

        else:
            context = {'form' : form}
            return render(request , 'base/predict.html' , context=context)

    context = {'form' : PatientForm()}
    return render(request , 'base/predict.html' , context=context)

def results(request):
    return render(request , 'base/results.html')


# Create your views here.
