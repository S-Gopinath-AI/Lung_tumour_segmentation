import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# classes (keep names matching your folders)
classes = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Better: apply augmentation to the whole training set, not per-class one-offs
train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    rescale=1.0 / 255
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_aug.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/train/",
    classes=classes,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/val",
    classes=classes,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    "E:/Lung Tumour Segmentation/Dataset/test",
    classes=classes,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
# Debug prints: ensure generators use the same class->index mapping
print('Train class indices:', train_gen.class_indices)
print('Val class indices:  ', val_gen.class_indices)
print('Test class indices: ', test_gen.class_indices)
#Class weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)
print(np.bincount(val_gen.classes))

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Reduce LR when a plateau is reached and save best model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint('lung_tumour_model_best.keras', monitor='val_loss', save_best_only=True)

from tensorflow.keras import layers,models
from tensorflow.keras.layers import Dropout
def lung_tumour_model(input_shape=(224,224,3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = layers.MaxPooling2D(2)(x)

    # Block 4 (Grad-CAM layer)
    x = layers.Conv2D(128, (3,3), padding='same', use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)
model = lung_tumour_model()
#model.summary()

# Build a focal loss that incorporates class weights (alpha)
alpha_for_loss = class_weights_array.astype(np.float32)

def make_focal_loss(alpha=None, gamma=2.0):
    alpha_tf = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # cross-entropy per sample
        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        # probability of the true class
        probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating = tf.pow(1.0 - probs, gamma)
        if alpha_tf is not None:
            alpha_factor = tf.reduce_sum(alpha_tf * y_true, axis=-1)
            loss = alpha_factor * modulating * cross_entropy
        else:
            loss = modulating * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss

focal = make_focal_loss(alpha=alpha_for_loss, gamma=2.0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal,
    metrics=['accuracy', tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec')]
)

#model Fitting
# Callback to inspect validation predictions each epoch (helps debug collapse)
class ValidationInspector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            preds = self.model.predict(val_gen, verbose=0)
            pred_classes = np.argmax(preds, axis=1)
            true = val_gen.classes
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(true, pred_classes)
            counts = np.bincount(pred_classes, minlength=len(classes))
            print(f"\n[Epoch {epoch+1}] Val pred distribution: {counts}")
            print("Confusion matrix:\n", cm)
            # per-class accuracy: diag / true counts (safe)
            true_counts = cm.sum(axis=1).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class_acc = np.where(true_counts > 0, cm.diagonal() / true_counts, 0.0)
            print("Per-class val acc:", np.round(per_class_acc, 3))
        except Exception as e:
            print('ValidationInspector error:', e)

inspector = ValidationInspector()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint, inspector]
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
model.save("lung_tumour_model.keras")
