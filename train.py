import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("--- Starting PulmoDetect Training & Visual Saving Pipeline ---")

# 1. Create Model Architecture
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 2. Generate training data
X_train = tf.random.uniform([10, 128, 128, 3])
y_train = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.float32)

print("Training model on dataset...")
model.fit(X_train, y_train, epochs=2, verbose=1)

# 3. Save model
model.save("model.h5")
print("Model saved successfully!")

# 4. Test Prediction on Real Uploaded Image & Save Result Image
print("Running test prediction on real uploaded X-Ray...")
real_img_path = "test_images/x ray.jpg"

if os.path.exists(real_img_path):
    # Load original image for visualization
    original_img = tf.keras.utils.load_img(real_img_path)
    
    # Load and preprocess image for model prediction
    img_for_model = tf.keras.utils.load_img(real_img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img_for_model)
    img_array = tf.expand_dims(img_array, 0)
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    print(f"Prediction Score for real image: {score}")
    
    if score < 0.5:
        result_text = f"Result: NORMAL (Score: {score:.2f})"
        text_color = "green"
    else:
        result_text = f"Result: DISEASE (Score: {score:.2f})"
        text_color = "red"
        
    # Plot and save image with result text on top
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.title(result_text, fontsize=14, color=text_color, fontweight='bold')
    plt.axis('off')
    
    # Save the output image
    output_path = "result_output.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Result image saved successfully as '{output_path}'!")
else:
    print("Warning: Real image not found!")

print("=== Pipeline Executed Successfully with Image Output! ===")
