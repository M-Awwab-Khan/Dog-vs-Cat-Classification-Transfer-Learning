import numpy as np
import tf_keras as keras
import tensorflow_hub as hub
from PIL import Image
import io

model = keras.models.load_model(
       ('catdog.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

# Define the target image size expected by the model
target_size = (224, 224)

# Load the trained model
# model = tf.keras.models.load_model("path/to/saved/model")

def preprocess_image(contents):
    image = Image.open(io.BytesIO(contents))
    # Resize the image to match the target size expected by the model
    image = image.resize(target_size)
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    # Normalize the pixel values to [0, 1]
    image_array = image_array / 255.0
    # Expand the dimensions of the image array to match the expected input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def classify_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make predictions using the trained model
    predictions = model.predict(preprocessed_image)
    # Map prediction probabilities to class labels
    class_labels = ["cat", "dog"]
    probabilities = softmax(predictions[0])
    predicted_class = class_labels[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100
    return predicted_class, confidence

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
