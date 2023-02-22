import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os, datetime
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, accuracy_score

from keras.utils import image_dataset_from_directory, array_to_img, plot_model
from keras.models import Sequential
from keras.layers import RandomFlip, RandomRotation, Input, GlobalAveragePooling2D, Dense, Dropout
from keras.applications import MobileNetV2, mobilenet_v2
from keras.callbacks import TensorBoard


def dataset_load(DATASET_PATH, BATCH_SIZE, IMG_SIZE):
    """Loading dataset and then splitting it to train, validation, test sets with ratio of 3:1:1.
    Outputs a PrefetchDataset.

    Args:
        DATASET_PATH (str): Path to dataset
        BATCH_SIZE (int): Batch size
        IMG_SIZE (tuple): Image size

    Returns:
        PrefetchDataset: A list of train, validation, test dataset
    """
    dataset = image_dataset_from_directory(DATASET_PATH, batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True)
    dataset_per = tf.data.experimental.cardinality(dataset)
    train_dataset = dataset.skip((dataset_per//5)*2)
    val_test_dataset = dataset.take((dataset_per//5)*2)

    val_test_dataset_per = tf.data.experimental.cardinality(val_test_dataset)
    val_dataset = val_test_dataset.skip(val_test_dataset_per//2)
    test_dataset = val_test_dataset.take(val_test_dataset_per//2)
   
    AUTOTUNE = tf.data.AUTOTUNE
    pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
    pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
    pf_test =test_dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, pf_train, pf_val, pf_test

def img_inspect(dataset, class_names):
    """Inspecting the images in dataset.

    Args:
        dataset (BatchDataset): Dataset
        class_names (list): Label names
    """
    for image, label in dataset.take(1):
        plt.figure(figsize=(10,10))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(array_to_img(image[i]))
            plt.xlabel(class_names[label[i]])
    plt.show()

def augmentation_layer():
    """Image augmentation model as one of the layers in models.

    Returns:
        Sequential: Sequential model
    """
    data_augmentation = Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation

def model_archi(IMG_SIZE, number_class, MODEL_PNG_PATH, data_augmentation=None):
    """Model architecture development.

    Args:
        IMG_SIZE (tuple): Image size
        number_class (int): Number of classes
        MODEL_PNG_PATH (str): Path to save model.png
        data_augmentation (Sequential, optional): Data augmentation model. Defaults to None.

    Returns:
        Functional: Functional model
    """
    IMG_SHAPE = IMG_SIZE + (3,)

    preprocess_input = mobilenet_v2.preprocess_input

    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.trainable = False
    base_model.summary()

    global_avg = GlobalAveragePooling2D()
    output_layer = Dense(number_class, activation='softmax')

    inputs = Input(shape=IMG_SHAPE)

    if data_augmentation:
        x = data_augmentation(inputs)
        x = preprocess_input(x)
    else:
        x = preprocess_input(inputs)

    x = base_model(x, training=False)
    x = global_avg(x)
    outputs = output_layer(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    plot_model(model, to_file=MODEL_PNG_PATH)
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def model_train(model, pf_train, pf_val, EPOCHS):
    """Model training.

    Args:
        model (Functional): Functional model
        pf_train (PrefetchDataset): Train split dataset
        pf_val (PrefetchDataset): Validation split dataset
        EPOCHS (int): Number of epochs for training

    Returns:
        History: Training history
        Functional: Functional model
    """
    log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log_dir=log_dir)

    history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb_callback])

    return history, model

def model_eval(model, pf_test, mode=''):
    """Model evaluation (Accuracy and F1 Score) before and after model training.

    Args:
        model (Functional): Functional model
        pf_test (PrefetchDataset): Test split dataset
        mode (str, optional): To output 'Before' or 'After'. Defaults to ''.
    """
    loss,acc = model.evaluate(pf_test)
    print(f"----------------Evaluation {mode} Training---------------")
    print("Loss = ",loss)
    print("Accuracy = ",acc)

def model_pred(model, pf_test):
    """Model prediction for dataset

    Args:
        model (Functional): Functional model
        pf_test (PrefetchDataset): Test split dataset

    Returns:
        list: All dataset batches of images
        list: All dataset batches of labels
        list: All dataset batches of predictions
    """
    y_pred_batches = []
    label_batches = []
    image_batches = []
    for image, label in pf_test.as_numpy_iterator():
        y_pred = np.argmax(model.predict(image),axis=1)
        image_batches.extend(image)
        y_pred_batches.extend(y_pred)
        label_batches.extend(label)
    
    return image_batches, label_batches, y_pred_batches

def model_report(label_batches, y_pred_batches, class_names):
    """Reports for model performance.

    Args:
        label_batches (list): All dataset batches of labels
        y_pred_batches (list): All dataset batches of predictions
        class_names (list): Label names
    """
    acc_scr = accuracy_score(label_batches, y_pred_batches)
    f1_scr = f1_score(label_batches, y_pred_batches, average='weighted')
    print(f"Accuracy Score: {acc_scr}, F1 Score: {f1_scr}\n\n")
    cm = confusion_matrix(label_batches, y_pred_batches)
    cr = classification_report(label_batches, y_pred_batches)
    print(cr)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def show_predictions(class_names, image_batches, label_batches, y_pred_batches):
    """_summary_

    Args:
        class_names (list): Label names
        image_batches: All dataset batches of images
        label_batches: All dataset batches of labels
        y_pred_batches: All dataset batches of predictions
    """
    plt.figure(figsize=(15,15))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(array_to_img(image_batches[i]))
        plt.xlabel(f"Label: {class_names[label_batches[i]]}, Prediction: {class_names[y_pred_batches[i]]}")
    plt.show()