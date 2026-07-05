import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Snake Images')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    print("Saved training history plot as 'training_history.png'")

def main():
    print("Loading datasets...")
    # Load training data
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    # Load test/validation data
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name='data_augmentation')

    print("Building model...")
    # Preprocessing input for MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                             include_top=False,
                             weights='imagenet')

    # Unfreeze the top layers of the base model for fine-tuning
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # increased dropout
    # Binary classification if 2 classes, else multi-class. We assume 2 classes.
    outputs = layers.Dense(1, activation='sigmoid')(x) if len(class_names) == 2 else layers.Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # Compile the model with a lower learning rate for fine-tuning
    loss_fn = tf.keras.losses.BinaryCrossentropy() if len(class_names) == 2 else tf.keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.summary()

    print("Starting training...")
    EPOCHS_FINE_TUNE = 15
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_FINE_TUNE
    )

    print("Training complete. Saving model...")
    model.save('snake_classifier_model.keras')
    print("Model saved to 'snake_classifier_model.keras'")
    
    with open('classes.txt', 'w') as f:
        f.write('\n'.join(class_names))

    plot_history(history)

if __name__ == '__main__':
    main()
