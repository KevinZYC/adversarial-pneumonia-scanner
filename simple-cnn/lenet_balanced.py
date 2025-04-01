import tensorflow as tf
import keras
from keras import layers, models, regularizers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
import keras._tf_keras.keras.backend as K
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Clear previous data
K.clear_session()
keras.backend.clear_session()

# Define dataset paths
data_dir = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
train_dir = os.path.join(data_dir, "chest_xray/train")
val_dir = os.path.join(data_dir, "chest_xray/val")
test_dir = os.path.join(data_dir, "chest_xray/test")

# Get class weights
count_train_n = len(os.listdir(train_dir+'/NORMAL'))
count_train_p = len(os.listdir(train_dir+'/PNEUMONIA'))
count_train_t = count_train_n + count_train_p
print("training set: normal =", count_train_n, "pneumonia =", count_train_p)
class_weights = {1: count_train_t/(2 * count_train_p), 0: count_train_t/(2 * count_train_n)}
print("calculated class weights", class_weights)

test_n = glob.glob(test_dir + '/NORMAL/*.jpeg')
df_test_n = pd.DataFrame({'path': test_n, 'label': 'NORMAL'}).head(230)
test_p = glob.glob(test_dir + '/PNEUMONIA/*.jpeg')
df_test_p = pd.DataFrame({'path': test_p, 'label': 'PNEUMONIA'}).head(230)

df_test = pd.concat([df_test_n, df_test_p], ignore_index=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

test, valid = train_test_split(df_test, test_size=0.33, random_state=49)
print("valid set:", valid['label'].value_counts())

# Image data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

"""val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)"""

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=valid,
    x_col='path', 
    y_col='label',  
    target_size=(150, 150),  
    color_mode='rgb',
    batch_size=64,  
    class_mode='binary',
    shuffle=False  
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

# CNN Model
def build_model():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(150, 150, 3), kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(84, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = build_model()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.4f}')


'''
Test Data:
Epoch 1/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 88s 530ms/step - accuracy: 0.5551 - loss: 0.7949 - val_accuracy: 0.7368 - val_loss: 0.6705 - learning_rate: 1.0000e-04
Epoch 2/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 89s 544ms/step - accuracy: 0.7370 - loss: 0.6119 - val_accuracy: 0.8289 - val_loss: 0.5156 - learning_rate: 1.0000e-04
Epoch 3/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 87s 531ms/step - accuracy: 0.8095 - loss: 0.4911 - val_accuracy: 0.8618 - val_loss: 0.4668 - learning_rate: 1.0000e-04
Epoch 4/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 98s 604ms/step - accuracy: 0.8266 - loss: 0.4463 - val_accuracy: 0.8618 - val_loss: 0.4533 - learning_rate: 1.0000e-04
Epoch 5/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 107s 655ms/step - accuracy: 0.8371 - loss: 0.4250 - val_accuracy: 0.7961 - val_loss: 0.5526 - learning_rate: 1.0000e-04
Epoch 6/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 110s 675ms/step - accuracy: 0.8711 - loss: 0.3805 - val_accuracy: 0.8618 - val_loss: 0.4400 - learning_rate: 1.0000e-04
Epoch 7/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 110s 674ms/step - accuracy: 0.8645 - loss: 0.3819 - val_accuracy: 0.8487 - val_loss: 0.4267 - learning_rate: 1.0000e-04
Epoch 8/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 99s 605ms/step - accuracy: 0.8785 - loss: 0.3557 - val_accuracy: 0.8421 - val_loss: 0.4534 - learning_rate: 1.0000e-04
Epoch 9/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 108s 660ms/step - accuracy: 0.8685 - loss: 0.3696 - val_accuracy: 0.8487 - val_loss: 0.4300 - learning_rate: 1.0000e-04
Epoch 10/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 115s 702ms/step - accuracy: 0.8523 - loss: 0.3872 - val_accuracy: 0.8421 - val_loss: 0.4330 - learning_rate: 1.0000e-04
Epoch 11/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 115s 703ms/step - accuracy: 0.8796 - loss: 0.3445 - val_accuracy: 0.8684 - val_loss: 0.4546 - learning_rate: 1.0000e-04
Epoch 12/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 0s 668ms/step - accuracy: 0.8850 - loss: 0.3274  
Epoch 12: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
163/163 ━━━━━━━━━━━━━━━━━━━━ 112s 683ms/step - accuracy: 0.8850 - loss: 0.3274 - val_accuracy: 0.8421 - val_loss: 0.5012 - learning_rate: 1.0000e-04
5/5 ━━━━━━━━━━━━━━━━━━━━ 5s 929ms/step - accuracy: 0.8827 - loss: 0.3511
'''