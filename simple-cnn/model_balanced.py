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

"""from keras._tf_keras.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)"""


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
    batch_size=8,
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
    batch_size=8,  
    class_mode='binary',
    shuffle=False  
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='path', 
    y_col='label',  
    target_size=(150, 150),  
    color_mode='rgb',
    batch_size=8,  
    class_mode='binary',
    shuffle=False  
)

class AvgSmoothing(tf.keras.layers.Layer):
    def __init__(self, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        smoothed = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=1, padding='SAME')
        return smoothed

# CNN Model
def build_model():
    model = models.Sequential([
        AvgSmoothing(pool_size=3),
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
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

model.save("model-balanced.h5")

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

with 8 batch on Colab:
652/652 ━━━━━━━━━━━━━━━━━━━━ 0s 142ms/step - accuracy: 0.5409 - loss: 1.24622025-04-14 18:18:17.383734: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[8,32,148,148]{3,2,1,0}, u8[0]{0}) custom-call(f32[8,3,150,150]{3,2,1,0}, f32[32,3,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
2025-04-14 18:18:17.493478: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[8,64,72,72]{3,2,1,0}, u8[0]{0}) custom-call(f32[8,32,74,74]{3,2,1,0}, f32[64,32,3,3]{3,2,1,0}, f32[64]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
2025-04-14 18:18:17.554126: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[8,128,34,34]{3,2,1,0}, u8[0]{0}) custom-call(f32[8,64,36,36]{3,2,1,0}, f32[128,64,3,3]{3,2,1,0}, f32[128]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
652/652 ━━━━━━━━━━━━━━━━━━━━ 104s 147ms/step - accuracy: 0.5410 - loss: 1.2459 - val_accuracy: 0.7368 - val_loss: 0.8218 - learning_rate: 1.0000e-04
Epoch 2/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 111ms/step - accuracy: 0.8093 - loss: 0.6929 - val_accuracy: 0.7829 - val_loss: 0.7204 - learning_rate: 1.0000e-04
Epoch 3/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 73s 111ms/step - accuracy: 0.8408 - loss: 0.5805 - val_accuracy: 0.8158 - val_loss: 0.6114 - learning_rate: 1.0000e-04
Epoch 4/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8521 - loss: 0.5296 - val_accuracy: 0.8026 - val_loss: 0.6504 - learning_rate: 1.0000e-04
Epoch 5/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8779 - loss: 0.4824 - val_accuracy: 0.8618 - val_loss: 0.5902 - learning_rate: 1.0000e-04
Epoch 6/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8758 - loss: 0.4650 - val_accuracy: 0.8355 - val_loss: 0.5494 - learning_rate: 1.0000e-04
Epoch 7/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 111ms/step - accuracy: 0.8817 - loss: 0.4412 - val_accuracy: 0.8355 - val_loss: 0.6067 - learning_rate: 1.0000e-04
Epoch 8/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 111ms/step - accuracy: 0.8890 - loss: 0.4294 - val_accuracy: 0.8882 - val_loss: 0.5422 - learning_rate: 1.0000e-04
Epoch 9/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8830 - loss: 0.4318 - val_accuracy: 0.8684 - val_loss: 0.4910 - learning_rate: 1.0000e-04
Epoch 10/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 71s 109ms/step - accuracy: 0.8904 - loss: 0.4108 - val_accuracy: 0.8750 - val_loss: 0.5086 - learning_rate: 1.0000e-04
Epoch 11/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8945 - loss: 0.3889 - val_accuracy: 0.8684 - val_loss: 0.4695 - learning_rate: 1.0000e-04
Epoch 12/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 73s 112ms/step - accuracy: 0.8910 - loss: 0.3964 - val_accuracy: 0.7829 - val_loss: 0.6683 - learning_rate: 1.0000e-04
Epoch 13/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 71s 110ms/step - accuracy: 0.8920 - loss: 0.3992 - val_accuracy: 0.8684 - val_loss: 0.4700 - learning_rate: 1.0000e-04
Epoch 14/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.8993 - loss: 0.3492 - val_accuracy: 0.8224 - val_loss: 0.4925 - learning_rate: 1.0000e-04
Epoch 15/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 72s 110ms/step - accuracy: 0.9037 - loss: 0.3595 - val_accuracy: 0.8421 - val_loss: 0.5718 - learning_rate: 1.0000e-04
Epoch 16/25
652/652 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - accuracy: 0.8949 - loss: 0.3790
Epoch 16: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-05.
652/652 ━━━━━━━━━━━━━━━━━━━━ 71s 109ms/step - accuracy: 0.8949 - loss: 0.3789 - val_accuracy: 0.7237 - val_loss: 0.8151 - learning_rate: 1.0000e-04
38/39 ━━━━━━━━━━━━━━━━━━━━ 0s 96ms/step - accuracy: 0.8994 - loss: 0.42032025-04-14 18:36:23.392868: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[4,32,148,148]{3,2,1,0}, u8[0]{0}) custom-call(f32[4,3,150,150]{3,2,1,0}, f32[32,3,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
2025-04-14 18:36:23.483550: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[4,64,72,72]{3,2,1,0}, u8[0]{0}) custom-call(f32[4,32,74,74]{3,2,1,0}, f32[64,32,3,3]{3,2,1,0}, f32[64]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
2025-04-14 18:36:23.570811: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[4,128,34,34]{3,2,1,0}, u8[0]{0}) custom-call(f32[4,64,36,36]{3,2,1,0}, f32[128,64,3,3]{3,2,1,0}, f32[128]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kRelu","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
39/39 ━━━━━━━━━━━━━━━━━━━━ 5s 122ms/step - accuracy: 0.8999 - loss: 0.4189
Test Accuracy: 0.9091
'''