import tensorflow as tf
import keras
from keras import layers, models, regularizers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
import numpy as np
import glob
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications import ResNet50
import keras._tf_keras.keras.backend as K

# Clear previous data
K.clear_session()
keras.backend.clear_session()

# Define dataset paths
data_dir = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Dataset downloaded to: {data_dir}")
paths = glob.glob(data_dir+'/*/*/*/*.jpeg')
print(f'found {len(paths)} images in the dataset')
data = pd.DataFrame(paths,columns=['path'])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['label'] = data['path'].apply(lambda x:x.split('/')[12].strip())
print(data.head(50))

"""df_majority = data[data.label == 'PNEUMONIA']
df_minority = data[data.label == 'NORMAL']

majority_count = len(df_majority)
minority_count = len(df_minority)

desired_ratio = 1
new_majority_count = int(minority_count * desired_ratio // 1)

df_majority_downsampled = resample(df_majority,
                                   replace=False,  
                                   n_samples=new_majority_count, 
                                   random_state=42)  
downsampled_data = pd.concat([df_minority, df_majority_downsampled])
print(downsampled_data['label'].value_counts())"""

train, temp = train_test_split(data, test_size=0.2, random_state=37)
test, valid = train_test_split(temp, test_size=0.5, random_state=49)

print(train['path'].isin(valid['path']).sum())  # Should be 0 if no overlap
print("Training class distribution:")
print(train['label'].value_counts())

print("\nValidation class distribution:")
print(valid['label'].value_counts())

print("\nTest class distribution:")
print(test['label'].value_counts())

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='path', 
    y_col='label',   
    target_size=(180, 180), 
    batch_size=32,
    class_mode='binary', 
    color_mode='rgb',
    shuffle=True
)

# Validation data generator
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=valid,
    x_col='path', 
    y_col='label',
    target_size=(180, 180), 
    batch_size=32,
    color_mode='rgb',
    class_mode='binary',
    shuffle=False  
)

# Testing data generator
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='path', 
    y_col='label',  
    target_size=(180, 180),  
    color_mode='rgb',
    batch_size=64,  
    class_mode='binary',
    shuffle=False  
)

print(train_generator.class_indices)  # Check the mapping in flow_from_dataframe
print(val_generator.class_indices)    # Check if labels are consistent in validation

# CNN Model
def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(180,180,3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = build_model()

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

sample_images, sample_labels = next(iter(train_generator))

print("Sample Labels from Batch:", sample_labels[:10])
print("Unique Labels in Batch:", np.unique(sample_labels))

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.4f}')
