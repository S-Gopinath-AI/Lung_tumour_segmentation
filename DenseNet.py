import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
if len(os.listdir('E:/Lung Tumour Segmentation/Dataset/train/Bengin cases'))>400:
    print("Bengin cases exceed 450 images, skipping augmentation.")
    
else:
    bengin_aug=ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,     
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8,1.2],
        rescale=1./255
    )
    aug_gen = bengin_aug.flow_from_directory(
        "E:/Lung Tumour Segmentation/Dataset/train/",
        classes=["Bengin cases"],       # Only this class will be augmented
        target_size=(224,224),
        batch_size=32,
        save_to_dir="E:/Lung Tumour Segmentation/Dataset/train/Bengin cases",    
        save_prefix="aug",
        class_mode=None
    )

    num_augmented_images = 300
    for i in range(num_augmented_images // 32 + 1):
        next(aug_gen)
        

train_aug = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/train/",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/val",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
#Class weights
class_weights = {
    0: 3.04,   
    1: 0.65,   
    2: 0.88    
}

#Defining Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      
    patience=5,              
    restore_best_weights=True  
)


base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False # freeze for faster, stable training

for layer in base.layers[-50:]:   # unfreeze last 50 layers
    layer.trainable = True

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')   # 3 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#model Fitting
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,               # you can increase epochs
    class_weight=class_weights,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

#Generate Predictions
import numpy as np

pred_probs = model.predict(test_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes

#confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(true_classes, pred_classes, target_names=list(test_gen.class_indices.keys())))

print(confusion_matrix(true_classes, pred_classes))
 
#save model
model.save("lung_densenet_model.keras")


