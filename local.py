from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)


model = load_model("model_120.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("C:/Users/KIIT/Desktop/internship-tasks/InfraredSolarModules/images/7394.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

data[0] = normalized_image_array

prediction = model.predict(data)
prediction_percentages = prediction[0]

sorted_indices = np.argsort(prediction_percentages)[::-1]

for i in sorted_indices:
    class_name = class_names[i]
    confidence_score = prediction_percentages[i]
    confidence_percentage = round(confidence_score * 100, 2)
    print("Class:", class_name[2:], "Confidence Score:", confidence_percentage)
