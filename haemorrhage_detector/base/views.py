from django.shortcuts import render
from base.forms import PatientForm
import tensorflow as tf
import cv2
import numpy as np
from xgboost import XGBClassifier

models_download_link = 'https://drive.google.com/drive/folders/1OFaAADrsEOCJl-WGB8BzeXlsnME-2dFG?usp=sharing'

feature_model_path = r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\feature_gen.keras"
bone_model_path = r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\models\Res101-boneModel.keras"
# saved_brain_model_layer = tf.keras.layers.TFSMLayer(brain_model_path, call_endpoint="serving_default")
# saved_bone_model_layer = tf.keras.layers.TFSMLayer(bone_model_path, call_endpoint="serving_default")

# brain_model = tf.keras.Sequential([saved_brain_model_layer])
# bone_model = tf.keras.Sequential([saved_bone_model_layer])

feature_model = tf.keras.models.load_model(feature_model_path)
bone_model = tf.keras.models.load_model(bone_model_path)

def make_brain_prediction(image_path):
    img = cv2.imread(image_path)
    img_array = np.array([img])
    features = feature_model.predict(img_array)
     
    
    xgb_model_iv = XGBClassifier()
    xgb_model_iv.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_iv.bin")
    iv = xgb_model_iv.predict(features)[0]

    xgb_model_ip = XGBClassifier()
    xgb_model_ip.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_ip.bin")
    ip = xgb_model_ip.predict(features)[0]

    xgb_model_sa = XGBClassifier()
    xgb_model_sa.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_sa.bin")
    sa = xgb_model_sa.predict(features)[0]

    xgb_model_ep = XGBClassifier()
    xgb_model_ep.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_ep.bin")
    ep = xgb_model_ep.predict(features)[0]

    xgb_model_sd = XGBClassifier()
    xgb_model_sd.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_sd.bin")
    sd = xgb_model_sd.predict(features)[0]

    xgb_model_no = XGBClassifier()
    xgb_model_no.load_model(r"C:\Users\kman0\Downloads\college projects\siddhi 2.0\XGB_models\xg_model_no.bin")
    no = xgb_model_no.predict(features)[0]


    haemorrhage = []
    if iv==1 :
        haemorrhage.append('Intraventricular Haemorrhage') 
    if ip==1 :
        haemorrhage.append('Intraparenchymal Haemorrhage')
    if sa==1 :
        haemorrhage.append('Subarachnoid Haemorrhage')
    if ep==1 :
        haemorrhage.append('Epidural Haemorrhage')
    if sd==1 :
        haemorrhage.append('Subdural Haemorrhage')
    if no==1 :
        haemorrhage = ['No Haemorrhage']


    return haemorrhage

def make_bone_prediction(image_path):
    img = cv2.imread(image_path)
    img_array = np.array([img])
    predictions = bone_model.predict(img_array)
    predicted_probability = predictions[0][0]
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

            haemorrhage = make_brain_prediction(brain_image_path)
            bone_pred , bone_pred_prob = make_bone_prediction(bone_image_path)

            # print(bone_image_path)
            # print(brain_image_path)

            context = {'brain_img' : brain_image_path , 'bone_img' : bone_image_path , 'haemorrhage' : haemorrhage  , 'bone_pred' : bone_pred , 'bone_pred_prob' : bone_pred_prob}
            return render(request , 'base/results.html' , context=context)

        else:
            context = {'form' : form}
            return render(request , 'base/predict.html' , context=context)

    context = {'form' : PatientForm()}
    return render(request , 'base/predict.html' , context=context)

def results(request):
    return render(request , 'base/results.html')


# Create your views here.
