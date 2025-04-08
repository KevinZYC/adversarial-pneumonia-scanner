import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
import keras._tf_keras.keras.backend as K
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import load_model

# Clear previous data
K.clear_session()
keras.backend.clear_session()

# Define dataset paths
data_dir = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
test_dir = os.path.join(data_dir, "chest_xray/test")

test_n = glob.glob(test_dir + '/NORMAL/*.jpeg')
df_test_n = pd.DataFrame({'path': test_n, 'label': 'NORMAL'}).head(230)
test_p = glob.glob(test_dir + '/PNEUMONIA/*.jpeg')
df_test_p = pd.DataFrame({'path': test_p, 'label': 'PNEUMONIA'}).head(230)

df_test = pd.concat([df_test_n, df_test_p], ignore_index=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

test, valid = train_test_split(df_test, test_size=0.67, random_state=49)
print("valid set:", valid['label'].value_counts())

val_test_datagen = ImageDataGenerator(rescale=1./255)

# adversarial testing
adv_light_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

adv_med_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

adv_heavy_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=True
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='path', 
    y_col='label',  
    target_size=(150, 150),  
    color_mode='rgb',
    batch_size=64,  
    class_mode='binary',
    shuffle=False  
)

adv_test_generator = adv_med_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='path', 
    y_col='label',  
    target_size=(150, 150),  
    color_mode='rgb',
    batch_size=64,  
    class_mode='binary',
    shuffle=False  
)

model = load_model("./saved-models/xception-balanced.h5")
test_loss, test_acc = model.evaluate(test_generator)
print(f'Unaltered Test Accuracy: {test_acc:.4f}')

adv_test_loss, adv_test_acc = model.evaluate(adv_test_generator)
print(f'Adversarial Test Accuracy: {adv_test_acc:.4f}')