import tensorflow as tf
import numpy as np
from PIL import Image




CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def load_model():
    model = tf.keras.models.load_model("model/plant_disease_model.h5")
    


def predict(path):
    
    img = Image.open(path)
    img = img.resize((224,224))
    
    if img.mode != "RGB":
        img = img.convert('RGB')
    
    
    
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    
    
    prediction = model.predict(img_array)
    
    prediction_index = np.argmax(prediction[0])
    
    raw_prediction = CLASS_NAMES[prediction_index]
    
    print(f"\n🌱 AI Diagnosis: {raw_prediction}")






if __name__ == "__main__":
    path =  "G:\AI\Plant_Disease_Detection_System\Test_Sample\l1.jpg"
    predict(path)





