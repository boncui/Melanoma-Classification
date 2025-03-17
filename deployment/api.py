import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io


#initialize FastAPI
app = FastAPI()

#load the trained model
model_path = "MODEL_PATH_HERE" 
mdoel = tf.keras.models.loadl_model(model_path)

#Class labels
class_labels = ["Melanoma", "Nevus", "Seborrheic Keratosis"]

#Image preprocessing Function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((180,180)) #resized to model's ex[ected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

#API Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        #model predictions
        predictions = model.predict(img_array)
        confidence_scores = predictions[0]
        predicted_class = class_labels[np.argmax(confidence_scores)]
        confidence = np.max(confidence_scores) * 100 #return percentage
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    
    except Exception as e:
        return {"error": str(e)}
    
#Run the API
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)


