import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("lung_densenet_model.keras")
00
# calling dummy
dummy = tf.zeros((1, 224, 224, 3))
model(dummy)

base_model = model.layers[0]   

for layer in reversed(base_model.layers):
    if "conv" in layer.name:
        print("Last Conv Layer:", layer.name)
        break

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    base_model = model.layers[0]   
    classifier_layer = model.layers[1:]  # GAP + Dense layers

    # Creating model that outputs last conv layer
    conv_model = tf.keras.models.Model(
        base_model.input,
        base_model.get_layer(last_conv_layer_name).output
    )

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in classifier_layer:
            x = layer(x)

        predictions = x
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

img_path = r"E:\AI\Lung Tumour Segmentation\test1.png"

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

#Prediction 
preds=model.predict(img_array)
class_index=np.argmax(preds[0])
confidence=float(preds[0][class_index])
class_names=["Bengin","Malignant","Normal"]
label=class_names[class_index]
print(f"predicted:{label}")
print(f"confidence:{confidence*100:.2f}%")



heatmap = make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name="conv5_block16_concat"
)

#overlay heatmap
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))

# Resize heatmap
heatmap = cv2.resize(heatmap, (224, 224))

# Normalize properly
heatmap = np.maximum(heatmap, 0)
heatmap /= (heatmap.max() + 1e-8)

# Convert to 0-255
heatmap = np.uint8(255 * heatmap)

# Apply colormap
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Blend better (reduce intensity)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
plt.title(f"Predicted: {label}")
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()