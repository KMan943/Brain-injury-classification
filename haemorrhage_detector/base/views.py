from django.shortcuts import render
from base.forms import PatientForm
import tensorflow as tf


brain_model_path = 'https://drive.google.com/drive/folders/1KRTpq-pqhy46FECyGMrzlFnbKiX56HMH?usp=drive_link'
bone_model_path = 'https://drive.google.com/drive/folders/1RJc2M3FKlI84TzXhKHnIuuhg19tyx_ZB?usp=drive_link'
saved_brain_model_layer = tf.keras.layers.TFSMLayer(brain_model_path, call_endpoint="serving_default")
saved_bone_model_layer = tf.keras.layers.TFSMLayer(bone_model_path, call_endpoint="serving_default")

brain_model = tf.keras.Sequential([saved_brain_model_layer])
bone_model = tf.keras.Sequential([saved_bone_model_layer])

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    img_array = img_array / 255.0 

    return img_array

def make_brain_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = brain_model(img_array)
    if isinstance(predictions, dict):
        
        prediction_tensor = predictions['dense_1']
        
        if hasattr(prediction_tensor, 'numpy'):
            prediction_array = prediction_tensor.numpy()
        else:
            prediction_array = prediction_tensor
    else:
        prediction_array = predictions.numpy()  


    
    predicted_class = prediction_array[0].argmax(axis=-1)  
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

    predicted_probability = prediction_array[0][predicted_class]  

    return haemorrhage, predicted_probability

def make_bone_prediction(image_path):
    img_array = preprocess_image(image_path)
    predictions = bone_model(img_array)
    if isinstance(predictions, dict):
       
        prediction_tensor = predictions['dense_1']
        
        if hasattr(prediction_tensor, 'numpy'):
            prediction_array = prediction_tensor.numpy()
        else:
            prediction_array = prediction_tensor
    else:
        prediction_array = predictions.numpy()  
    predicted_probability = prediction_array[0]

    if predicted_probability < 0.5:
        bone_pred = "No Fracture"

    else:
        bone_pred = "Fracture"

    return bone_pred

def predict(request):
    if request.method == "POST":
        form = PatientForm(request.POST , request.FILES)
        if form.is_valid():
            img = form.save()
            brain_image_path = img.brain_image.path
            bone_image_path = img.bone_image.path

            brain_pred_class , brain_pred_prob = make_brain_prediction(brain_image_path)
            bone_pred = make_bone_prediction(bone_image_path)

            # print(bone_image_path)
            # print(brain_image_path)

            context = {'brain_img' : brain_image_path , 'bone_img' : bone_image_path , 'brain_pred_class' : brain_pred_class  , 'brain_pred_prob' : brain_pred_prob , 'bone_pred' : bone_pred}
            return render(request , 'base/results.html' , context=context)

        else:
            context = {'form' : form}
            return render(request , 'base/predict.html' , context=context)

    context = {'form' : PatientForm()}
    return render(request , 'base/predict.html' , context=context)

def results(request):
    return render(request , 'base/results.html')


# Create your views here.
