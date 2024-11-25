import keras
import numpy as np
from PIL import Image
def prediction(path):
    # Load and preprocess the image
    class_name=""
    img = Image.open(path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Load the model
    model = keras.models.load_model("model.h5")
    # Predict
    result = model.predict(img_array)
    if result [0][0]>0.5:
      print("Dog")
      class_name="Dog"
    else:
      print("Cat")
      class_name="Cat"
    return class_name
  # Test the function
pre=prediction("pngtree-cat-cute-png-png-image_10151876.png")
print(pre)


 