import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')


img_height, img_width = 128, 128
batch_size = 32
train_dir = r'C:\Users\SOWMYA\Downloads\Tumor\Tumor\brain\brain_tumor_dataset'
val_dir = r'C:\Users\SOWMYA\Downloads\Tumor\Tumor\brain\brain_tumor_dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=(train_generator.samples // batch_size),
    validation_data=val_generator,
    validation_steps=(val_generator.samples // batch_size),
    epochs=40
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(val_generator, verbose=2)
print(f'Test accuracy: {test_acc}')


img_path = r'C:\Users\SOWMYA\Downloads\Tumor\Tumor\brain\brain_tumor_dataset\yes\Y21.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

prediction = model.predict(img_array)

if prediction > 0.5:
    result_text = f"Yes: {prediction[0][0]:.4f}"
else:
    result_text = f"No: {1 - prediction[0][0]:.4f}"

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Prediction: {result_text}", fontsize=16)
plt.axis('off')
plt.show()
