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
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
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
training data:
Epoch 1/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 123s 743ms/step - accuracy: 0.4822 - loss: 1.8604 - val_accuracy: 0.7697 - val_loss: 1.3949 - learning_rate: 1.0000e-04
Epoch 2/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 127s 776ms/step - accuracy: 0.7246 - loss: 1.2741 - val_accuracy: 0.7763 - val_loss: 1.1094 - learning_rate: 1.0000e-04
Epoch 3/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 127s 779ms/step - accuracy: 0.8113 - loss: 0.9840 - val_accuracy: 0.8289 - val_loss: 0.9191 - learning_rate: 1.0000e-04
Epoch 4/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 129s 791ms/step - accuracy: 0.8382 - loss: 0.8565 - val_accuracy: 0.8224 - val_loss: 0.9020 - learning_rate: 1.0000e-04
Epoch 5/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 129s 787ms/step - accuracy: 0.8332 - loss: 0.8160 - val_accuracy: 0.8289 - val_loss: 0.8141 - learning_rate: 1.0000e-04
Epoch 6/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 135s 828ms/step - accuracy: 0.8607 - loss: 0.7262 - val_accuracy: 0.8289 - val_loss: 0.7863 - learning_rate: 1.0000e-04
Epoch 7/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 124s 761ms/step - accuracy: 0.8590 - loss: 0.7053 - val_accuracy: 0.8487 - val_loss: 0.7710 - learning_rate: 1.0000e-04
Epoch 8/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 125s 768ms/step - accuracy: 0.8559 - loss: 0.7023 - val_accuracy: 0.8553 - val_loss: 0.8002 - learning_rate: 1.0000e-04
Epoch 9/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 119s 730ms/step - accuracy: 0.8822 - loss: 0.6231 - val_accuracy: 0.8355 - val_loss: 0.7708 - learning_rate: 1.0000e-04
Epoch 10/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 126s 775ms/step - accuracy: 0.8544 - loss: 0.6566 - val_accuracy: 0.8618 - val_loss: 0.7581 - learning_rate: 1.0000e-04
Epoch 11/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 130s 797ms/step - accuracy: 0.8487 - loss: 0.6744 - val_accuracy: 0.8487 - val_loss: 0.7053 - learning_rate: 1.0000e-04
Epoch 12/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 121s 741ms/step - accuracy: 0.8795 - loss: 0.6058 - val_accuracy: 0.8487 - val_loss: 0.6897 - learning_rate: 1.0000e-04
Epoch 13/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 124s 758ms/step - accuracy: 0.8832 - loss: 0.5951 - val_accuracy: 0.8487 - val_loss: 0.6862 - learning_rate: 1.0000e-04
Epoch 14/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 126s 774ms/step - accuracy: 0.8780 - loss: 0.5964 - val_accuracy: 0.8553 - val_loss: 0.6954 - learning_rate: 1.0000e-04
Epoch 15/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 124s 759ms/step - accuracy: 0.8880 - loss: 0.5736 - val_accuracy: 0.8816 - val_loss: 0.6667 - learning_rate: 1.0000e-04
Epoch 16/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 127s 777ms/step - accuracy: 0.8773 - loss: 0.5617 - val_accuracy: 0.8421 - val_loss: 0.7648 - learning_rate: 1.0000e-04
Epoch 17/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 123s 755ms/step - accuracy: 0.8907 - loss: 0.5304 - val_accuracy: 0.8618 - val_loss: 0.6340 - learning_rate: 1.0000e-04
Epoch 18/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 124s 760ms/step - accuracy: 0.8943 - loss: 0.5313 - val_accuracy: 0.8618 - val_loss: 0.6116 - learning_rate: 1.0000e-04
Epoch 19/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 120s 735ms/step - accuracy: 0.8962 - loss: 0.5209 - val_accuracy: 0.8487 - val_loss: 0.7200 - learning_rate: 1.0000e-04
Epoch 20/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 121s 740ms/step - accuracy: 0.9020 - loss: 0.4924 - val_accuracy: 0.8684 - val_loss: 0.6699 - learning_rate: 1.0000e-04
Epoch 21/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 120s 737ms/step - accuracy: 0.8992 - loss: 0.5045 - val_accuracy: 0.8618 - val_loss: 0.6430 - learning_rate: 1.0000e-04
Epoch 22/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 1092s 7s/step - accuracy: 0.8983 - loss: 0.4917 - val_accuracy: 0.8553 - val_loss: 0.6747 - learning_rate: 1.0000e-04
Epoch 23/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 0s 835ms/step - accuracy: 0.8983 - loss: 0.4838   
Epoch 23: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
163/163 ━━━━━━━━━━━━━━━━━━━━ 163s 851ms/step - accuracy: 0.8983 - loss: 0.4838 - val_accuracy: 0.8026 - val_loss: 0.9135 - learning_rate: 1.0000e-04
5/5 ━━━━━━━━━━━━━━━━━━━━ 5s 919ms/step - accuracy: 0.8899 - loss: 0.5368


Epoch 1/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 126s 758ms/step - accuracy: 0.4658 - loss: 7.2140 - val_accuracy: 0.5395 - val_loss: 3.6941 - learning_rate: 1.0000e-04
Epoch 2/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 136s 831ms/step - accuracy: 0.5836 - loss: 3.3460 - val_accuracy: 0.5724 - val_loss: 2.6715 - learning_rate: 1.0000e-04
Epoch 3/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 141s 861ms/step - accuracy: 0.7282 - loss: 2.3835 - val_accuracy: 0.8224 - val_loss: 1.9420 - learning_rate: 1.0000e-04
Epoch 4/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 135s 828ms/step - accuracy: 0.7953 - loss: 1.8496 - val_accuracy: 0.6250 - val_loss: 1.9692 - learning_rate: 1.0000e-04
Epoch 5/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 136s 835ms/step - accuracy: 0.7992 - loss: 1.5375 - val_accuracy: 0.8224 - val_loss: 1.3640 - learning_rate: 1.0000e-04
Epoch 6/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 131s 800ms/step - accuracy: 0.8412 - loss: 1.2802 - val_accuracy: 0.8224 - val_loss: 1.1974 - learning_rate: 1.0000e-04
Epoch 7/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 134s 822ms/step - accuracy: 0.8422 - loss: 1.1195 - val_accuracy: 0.7368 - val_loss: 1.2433 - learning_rate: 1.0000e-04
Epoch 8/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 137s 842ms/step - accuracy: 0.8351 - loss: 1.0312 - val_accuracy: 0.8289 - val_loss: 1.0001 - learning_rate: 1.0000e-04
Epoch 9/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 132s 808ms/step - accuracy: 0.8518 - loss: 0.9305 - val_accuracy: 0.8092 - val_loss: 0.9875 - learning_rate: 1.0000e-04
Epoch 10/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 139s 850ms/step - accuracy: 0.8528 - loss: 0.8615 - val_accuracy: 0.8289 - val_loss: 0.9016 - learning_rate: 1.0000e-04
Epoch 11/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 135s 827ms/step - accuracy: 0.8526 - loss: 0.8197 - val_accuracy: 0.8158 - val_loss: 0.8576 - learning_rate: 1.0000e-04
Epoch 12/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 131s 801ms/step - accuracy: 0.8722 - loss: 0.7550 - val_accuracy: 0.8355 - val_loss: 0.8381 - learning_rate: 1.0000e-04
Epoch 13/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 137s 839ms/step - accuracy: 0.8304 - loss: 0.7869 - val_accuracy: 0.8355 - val_loss: 0.7930 - learning_rate: 1.0000e-04
Epoch 14/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 135s 829ms/step - accuracy: 0.8523 - loss: 0.7353 - val_accuracy: 0.8421 - val_loss: 0.7871 - learning_rate: 1.0000e-04
Epoch 15/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 137s 838ms/step - accuracy: 0.8625 - loss: 0.6866 - val_accuracy: 0.8289 - val_loss: 0.7614 - learning_rate: 1.0000e-04
Epoch 16/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 136s 831ms/step - accuracy: 0.8618 - loss: 0.6895 - val_accuracy: 0.8289 - val_loss: 0.7735 - learning_rate: 1.0000e-04
Epoch 17/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 142s 867ms/step - accuracy: 0.8781 - loss: 0.6462 - val_accuracy: 0.8355 - val_loss: 0.7203 - learning_rate: 1.0000e-04
Epoch 18/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 135s 828ms/step - accuracy: 0.8532 - loss: 0.6637 - val_accuracy: 0.8421 - val_loss: 0.7234 - learning_rate: 1.0000e-04
Epoch 19/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 137s 840ms/step - accuracy: 0.8598 - loss: 0.6543 - val_accuracy: 0.8289 - val_loss: 0.7075 - learning_rate: 1.0000e-04
Epoch 20/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 142s 871ms/step - accuracy: 0.8589 - loss: 0.6265 - val_accuracy: 0.8421 - val_loss: 0.7114 - learning_rate: 1.0000e-04
Epoch 21/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 239s 1s/step - accuracy: 0.8563 - loss: 0.6296 - val_accuracy: 0.8355 - val_loss: 0.6842 - learning_rate: 1.0000e-04
Epoch 22/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 168s 1s/step - accuracy: 0.8549 - loss: 0.6199 - val_accuracy: 0.8289 - val_loss: 0.6911 - learning_rate: 1.0000e-04
Epoch 23/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 127s 776ms/step - accuracy: 0.8696 - loss: 0.6038 - val_accuracy: 0.8487 - val_loss: 0.6647 - learning_rate: 1.0000e-04
Epoch 24/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 129s 790ms/step - accuracy: 0.8567 - loss: 0.5905 - val_accuracy: 0.8289 - val_loss: 0.6687 - learning_rate: 1.0000e-04
Epoch 25/25
163/163 ━━━━━━━━━━━━━━━━━━━━ 125s 767ms/step - accuracy: 0.8613 - loss: 0.5928 - val_accuracy: 0.8421 - val_loss: 0.6541 - learning_rate: 1.0000e-04
5/5 ━━━━━━━━━━━━━━━━━━━━ 4s 844ms/step - accuracy: 0.8704 - loss: 0.5832
Test Accuracy: 0.8539
'''