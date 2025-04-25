import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
import keras._tf_keras.keras.backend as K
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import load_img, img_to_array
from scipy.optimize import differential_evolution

# Define dataset paths
data_dir = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
test_dir = os.path.join(data_dir, "chest_xray/val/NORMAL")
model = load_model("./saved-models/model-balanced.h5")

def preprocess(image):
    image = tf.image.resize(image, (150, 150))
    return np.expand_dims(image / 255.0, axis=0)

def perturb_image(x, image):
    x_pos, y_pos, r, g, b = map(int, x)
    perturbed = np.copy(image)
    perturbed[x_pos, y_pos] = [r, g, b]
    return perturbed

def predict_class(x, image, model, true_label):
    perturbed = perturb_image(x, image)
    perturbed_input = np.expand_dims(perturbed / 255.0, axis=0)
    prediction = model.predict(perturbed_input, verbose=0)[0][0]
    if true_label == 1:
        return prediction
    else:
        return 1 - prediction

bounds = [
    (0, 149),  # x
    (0, 149),  # y
    (0, 255),  # R
    (0, 255),  # G
    (0, 255),  # B
]

image_dir = test_dir
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

for idx, img_path in enumerate(image_paths):
    original_img = load_img(img_path, target_size=(150, 150))
    original_img_array = img_to_array(original_img)
    image_input = preprocess(original_img_array)
    
    original_pred = model.predict(image_input, verbose=0)[0][0]
    true_label = 0 if original_pred < 0.5 else 1
    print(f"\n[{idx+1}] Original prediction: {original_pred:.4f} (Label {true_label})")

    result = differential_evolution(
        predict_class,
        bounds,
        args=(original_img_array, model, true_label),
        maxiter=75,
        popsize=20,
        recombination=0.7,
        polish=True
    )

    perturbed_image = perturb_image(result.x, original_img_array)
    perturbed_pred = model.predict(np.expand_dims(perturbed_image / 255.0, axis=0), verbose=0)[0][0]
    print("After one-pixel attack:", perturbed_pred)