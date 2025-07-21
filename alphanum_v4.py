import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import h5py #We import this as we are dealing with HDF5 files
import tensorflowjs as tfjs

DATA_FRACTION_TO_USE = 0.5  # Use 50% of the dataset to reduce training time

class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def load_and_preprocess_data():
    """
    Loads a fraction of the EMNIST 'byclass' dataset and preprocesses it for training.
    """
    train_split_string = f"train[:{int(DATA_FRACTION_TO_USE * 100)}%]"
    test_split_string = f"test[:{int(DATA_FRACTION_TO_USE * 100)}%]"
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=[train_split_string, test_split_string],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    def preprocess(image, label):
        """
        Normalizes images and one-hot encodes labels.
        IMPORTANT: EMNIST images are rotated and flipped in the raw data.
        We apply rot90(k=3) and flip_left_right to correct orientation.
        This preprocessing MUST be applied to ALL inputs when using the trained model.
        """
        image = tf.image.rot90(image, k=3)  # Rotate 270° clockwise
        image = tf.image.flip_left_right(image)  # Horizontal flip
        image = tf.cast(image, tf.float32) / 255.0

        num_classes = ds_info.features['label'].num_classes
        label = tf.one_hot(label, num_classes)
        return image, label

    batch_size = 64
    
    train_dataset = ds_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    num_train_examples = ds_info.splits[train_split_string].num_examples
    num_test_examples = ds_info.splits[test_split_string].num_examples
    print(f"Using {DATA_FRACTION_TO_USE*100:.0f}% of the dataset.")
    print(f"Number of training examples: {num_train_examples}")
    print(f"Number of testing examples: {num_test_examples}")
    
    return train_dataset, test_dataset, ds_info

def create_model(num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # The filters of this layer have a depth of 32
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    train_ds, test_ds, ds_info = load_and_preprocess_data()
    num_classes = ds_info.features['label'].num_classes
    
    model = create_model(num_classes)
    model.summary()
    print("\nStarting model training...")
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=test_ds
    )
    print("Model training finished.\n")

    file_name = "saved_emnist_model_"+ str(int(DATA_FRACTION_TO_USE*100)) +"_percent_data.keras"

    #Save the *entire* model (architecture + weights + optimizer state)
    # Default format is TensorFlow SavedModel (recommended, non-deprecated).
    model.save(file_name)
    print("Full model saved to file '"+ file_name +"'\n")

    from tensorflow.keras.models import load_model
    loaded_model = load_model(file_name)
    print("Loaded full model from '"+ file_name +"' — here's its summary:")
    loaded_model.summary()

    for images, labels in test_ds.take(1):
        image = images[0]  
        true_label = tf.argmax(labels[0]).numpy()  
        
        prediction = loaded_model.predict(image[tf.newaxis, ...])
        predicted_label = np.argmax(prediction[0])
        
        plt.imshow(image[:, :, 0], cmap='gray')
        plt.title(f"True: {class_names[true_label]}, Predicted: {class_names[predicted_label]}")
        plt.axis('off')
        plt.show()