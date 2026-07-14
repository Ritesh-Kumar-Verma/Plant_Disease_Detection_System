import tensorflow as tf
import numpy as np
from PIL import Image
import sys




CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']




def main():
    
    model = tf.keras.models.load_model("model/plant_disease_model.h5")
    
    print("Hello from plant-disease-detection-system!")
    img = Image.open("Test_Sample/image.png")
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
    main()







# import numpy as np
# from tensorflow.keras.preprocessing import image

# def predict_plant_disease(img_path, model, train_generator):
#     # Load and preprocess image
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Get predictions
#     predictions = model.predict(img_array)[0]
    
#     # Map index to class names
#     class_labels = list(train_generator.class_indices.keys())
    
#     # Sort and print all confidence scores higher than 1%
#     print("\n📊 --- Confidence Distribution ---")
#     sorted_indices = np.argsort(predictions)[::-1]
#     for idx in sorted_indices:
#         confidence = predictions[idx] * 100
#         if confidence > 1.0: # Only print relevant classes
#             print(f"{class_labels[idx]}: {confidence:.2f}%")

# # Usage:
# # predict_plant_disease('path_to_your_google_downloaded_image.jpg', model, train_generator)