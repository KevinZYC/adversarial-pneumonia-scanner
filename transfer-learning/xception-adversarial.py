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
from keras._tf_keras.keras.applications import Xception

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

train_n = glob.glob(train_dir + '/NORMAL/*.jpeg')
df_train_n = pd.DataFrame({'path': train_n, 'label': 'NORMAL'})
train_p = glob.glob(train_dir + '/PNEUMONIA/*.jpeg')
df_train_p = pd.DataFrame({'path': train_p, 'label': 'PNEUMONIA'})

df_train = pd.concat([df_train_n, df_train_p], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# double every image in train
train_2 = df_train.copy()
train_large = pd.concat([train_2, df_train], ignore_index=True)

test_n = glob.glob(test_dir + '/NORMAL/*.jpeg')
df_test_n = pd.DataFrame({'path': test_n, 'label': 'NORMAL'}).head(230)
test_p = glob.glob(test_dir + '/PNEUMONIA/*.jpeg')
df_test_p = pd.DataFrame({'path': test_p, 'label': 'PNEUMONIA'}).head(230)

df_test = pd.concat([df_test_n, df_test_p], ignore_index=True)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

test, valid = train_test_split(df_test, test_size=0.67, random_state=49)
print("valid set:", valid['label'].value_counts())

# Image data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

adv_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# adversarial testing
adv_test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=75,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

# Load images from directory
train_generator = adv_datagen.flow_from_dataframe(
    dataframe=train_large,
    x_col='path', 
    y_col='label',  
    target_size=(150, 150),  
    color_mode='rgb',
    batch_size=64,  
    class_mode='binary',
    shuffle=False  
)

"""val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)"""
val_generator = adv_datagen.flow_from_dataframe(
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

adv_test_generator = adv_test_datagen.flow_from_dataframe(
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
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(150,150,3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(220, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(60, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = build_model()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)


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
print(f'Unaltered Test Accuracy: {test_acc:.4f}')

adv_test_loss, adv_test_acc = model.evaluate(adv_test_generator)
print(f'Adversarial Test Accuracy: {adv_test_acc:.4f}')

model.save("xception-balanced.h5")

'''
training data without L2:
Epoch 1/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 149s 888ms/step - accuracy: 0.8381 - loss: 0.3608 - val_accuracy: 0.8289 - val_loss: 0.3194 - learning_rate: 0.0010
Epoch 2/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 143s 876ms/step - accuracy: 0.9114 - loss: 0.2174 - val_accuracy: 0.8224 - val_loss: 0.3273 - learning_rate: 0.0010
Epoch 3/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 150s 919ms/step - accuracy: 0.9043 - loss: 0.2341 - val_accuracy: 0.8816 - val_loss: 0.2825 - learning_rate: 0.0010
Epoch 4/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 151s 926ms/step - accuracy: 0.9132 - loss: 0.2139 - val_accuracy: 0.8816 - val_loss: 0.2464 - learning_rate: 0.0010
Epoch 5/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 154s 942ms/step - accuracy: 0.9081 - loss: 0.2074 - val_accuracy: 0.9079 - val_loss: 0.2363 - learning_rate: 0.0010
Epoch 6/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 148s 905ms/step - accuracy: 0.9117 - loss: 0.2052 - val_accuracy: 0.8816 - val_loss: 0.2538 - learning_rate: 0.0010
Epoch 7/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 146s 890ms/step - accuracy: 0.9218 - loss: 0.1862 - val_accuracy: 0.8816 - val_loss: 0.2521 - learning_rate: 0.0010
Epoch 8/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 146s 896ms/step - accuracy: 0.9206 - loss: 0.1818 - val_accuracy: 0.8816 - val_loss: 0.2574 - learning_rate: 0.0010
Epoch 9/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 149s 913ms/step - accuracy: 0.9318 - loss: 0.1753 - val_accuracy: 0.8882 - val_loss: 0.2710 - learning_rate: 0.0010
Epoch 10/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 0s 878ms/step - accuracy: 0.9238 - loss: 0.1800  
Epoch 10: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
163/163 ━━━━━━━━━━━━━━━━━━━━ 148s 903ms/step - accuracy: 0.9238 - loss: 0.1801 - val_accuracy: 0.8947 - val_loss: 0.2570 - learning_rate: 0.0010
5/5 ━━━━━━━━━━━━━━━━━━━━ 10s 2s/step - accuracy: 0.8866 - loss: 0.3076
Test Accuracy: 0.8896

training data with L2:
Epoch 1/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 138s 820ms/step - accuracy: 0.8401 - loss: 1.0518 - val_accuracy: 0.8487 - val_loss: 0.9361 - learning_rate: 0.0010
Epoch 2/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 139s 849ms/step - accuracy: 0.9047 - loss: 0.8417 - val_accuracy: 0.8750 - val_loss: 0.8407 - learning_rate: 0.0010
Epoch 3/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 136s 832ms/step - accuracy: 0.8998 - loss: 0.7180 - val_accuracy: 0.8618 - val_loss: 0.7283 - learning_rate: 0.0010
Epoch 4/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 139s 848ms/step - accuracy: 0.9121 - loss: 0.6365 - val_accuracy: 0.8618 - val_loss: 0.6704 - learning_rate: 0.0010
Epoch 5/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 139s 853ms/step - accuracy: 0.9084 - loss: 0.5639 - val_accuracy: 0.8816 - val_loss: 0.5749 - learning_rate: 0.0010
Epoch 6/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 140s 859ms/step - accuracy: 0.9222 - loss: 0.4908 - val_accuracy: 0.9013 - val_loss: 0.5367 - learning_rate: 0.0010
Epoch 7/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 143s 875ms/step - accuracy: 0.9155 - loss: 0.4541 - val_accuracy: 0.8684 - val_loss: 0.4946 - learning_rate: 0.0010
Epoch 8/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 171s 1s/step - accuracy: 0.9252 - loss: 0.3851 - val_accuracy: 0.8816 - val_loss: 0.4353 - learning_rate: 0.0010
Epoch 9/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 141s 863ms/step - accuracy: 0.9145 - loss: 0.3918 - val_accuracy: 0.8553 - val_loss: 0.4557 - learning_rate: 0.0010
Epoch 10/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 141s 863ms/step - accuracy: 0.9108 - loss: 0.3586 - val_accuracy: 0.8816 - val_loss: 0.4160 - learning_rate: 0.0010
Epoch 11/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 141s 865ms/step - accuracy: 0.9152 - loss: 0.3573 - val_accuracy: 0.8947 - val_loss: 0.3909 - learning_rate: 0.0010
Epoch 12/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 144s 884ms/step - accuracy: 0.9022 - loss: 0.3564 - val_accuracy: 0.8750 - val_loss: 0.3710 - learning_rate: 0.0010
Epoch 13/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 153s 934ms/step - accuracy: 0.9294 - loss: 0.3013 - val_accuracy: 0.8750 - val_loss: 0.4001 - learning_rate: 0.0010
Epoch 14/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 144s 883ms/step - accuracy: 0.9242 - loss: 0.3015 - val_accuracy: 0.9013 - val_loss: 0.3119 - learning_rate: 0.0010
Epoch 15/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 144s 880ms/step - accuracy: 0.9268 - loss: 0.2750 - val_accuracy: 0.8684 - val_loss: 0.3948 - learning_rate: 0.0010
Epoch 16/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 147s 900ms/step - accuracy: 0.9147 - loss: 0.3056 - val_accuracy: 0.8487 - val_loss: 0.3852 - learning_rate: 0.0010
Epoch 17/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 147s 896ms/step - accuracy: 0.9202 - loss: 0.2828 - val_accuracy: 0.8947 - val_loss: 0.3395 - learning_rate: 0.0010
Epoch 18/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 0s 876ms/step - accuracy: 0.9163 - loss: 0.2938  
Epoch 18: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
163/163 ━━━━━━━━━━━━━━━━━━━━ 147s 900ms/step - accuracy: 0.9163 - loss: 0.2938 - val_accuracy: 0.8947 - val_loss: 0.3591 - learning_rate: 0.0010
Epoch 19/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 145s 887ms/step - accuracy: 0.9204 - loss: 0.2645 - val_accuracy: 0.8882 - val_loss: 0.3387 - learning_rate: 5.0000e-04
Epoch 20/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 143s 878ms/step - accuracy: 0.9293 - loss: 0.2497 - val_accuracy: 0.8487 - val_loss: 0.3616 - learning_rate: 5.0000e-04
5/5 ━━━━━━━━━━━━━━━━━━━━ 7s 1s/step - accuracy: 0.8925 - loss: 0.3931
Test Accuracy: 0.8929
'''