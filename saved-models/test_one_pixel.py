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

model = load_model("./saved-models/lenet-balanced.h5")

def preprocess(image):
    image = tf.image.resize(image, (150, 150))
    return np.expand_dims(image / 255.0, axis=0)

# Attack function
def one_pixel_attack(x, model, original_image, true_label):
    try:
        adv_img = original_image.copy()
        x = np.round(x).astype(int)
        px, py, r, g, b = x
        px = np.clip(px, 0, 149)
        py = np.clip(py, 0, 149)
        adv_img[py, px] = [r, g, b]
        adv_input = preprocess(adv_img)
        pred = model.predict(adv_input, verbose=0)[0][0]
        return pred if true_label == 0 else 1 - pred
    except Exception as e:
        print(f"Error in fitness fn: {e}")
        return 1e6  # Fail-safe

# Bounds: 1 pixel (x, y, R, G, B)
#bounds = [(0, 149), (0, 149), (0, 255), (0, 255), (0, 255)]
bounds = [(50, 100), (50, 100), (0, 255), (0, 255), (0, 255)]

# Load all images from directory
image_dir = test_dir
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.jpeg', '.png'))]

for idx, img_path in enumerate(image_paths):
    # Load and preprocess
    original_img = load_img(img_path, target_size=(150, 150))
    original_img_array = img_to_array(original_img)
    image_input = preprocess(original_img_array)
    
    # Predict
    original_pred = model.predict(image_input, verbose=0)[0][0]
    true_label = 0 if original_pred < 0.5 else 1
    print(f"\n[{idx+1}] Original prediction: {original_pred:.4f} (Label {true_label})")

    # Run DE
    result = differential_evolution(
        one_pixel_attack,
        bounds,
        args=(model, original_img_array, true_label),
        maxiter=30,
        popsize=5,
        recombination=1,
        polish=False,
        disp=False
    )

    # Apply perturbation
    best_x = np.round(result.x).astype(int)
    px, py, r, g, b = best_x
    adv_img = original_img_array.copy()
    adv_img[py, px] = [r, g, b]
    adv_input = preprocess(adv_img)
    adv_pred = model.predict(adv_input, verbose=0)[0][0]

    print(f"    Adversarial prediction: {adv_pred:.4f}")