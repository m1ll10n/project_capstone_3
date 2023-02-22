# %% Import
import os
from module import dataset_load, img_inspect, augmentation_layer, model_archi, model_train, model_eval, model_pred, model_report

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'no_aug_model.h5')
MODEL_PNG_PATH = os.path.join(os.getcwd(), 'saved_models', 'NA_model.png')

MODEL_FOLDER_PATH = os.path.join(os.getcwd(), 'saved_models')
if not os.path.exists(MODEL_FOLDER_PATH):
    os.makedirs(MODEL_FOLDER_PATH)

# %% Data Loading
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
dataset, pf_train, pf_val, pf_test = dataset_load(DATASET_PATH, BATCH_SIZE, IMG_SIZE)

# %% Exploratory Data Analysis(EDA)
class_names = dataset.class_names
number_class = len(class_names)
img_inspect(dataset, class_names)

# %% Model Development
data_augmentation = augmentation_layer()
model = model_archi(IMG_SIZE, number_class, MODEL_PNG_PATH, data_augmentation)
model_eval(model, pf_test, "Before")

# %% Model Train
EPOCHS = 5
history, model = model_train(model, pf_train, pf_val, EPOCHS)
model_eval(model, pf_test, "After")

# %% Model Development
image_batches, label_batches, y_pred_batches = model_pred(model, pf_test)

# %% Model Analysis
model_report(label_batches, y_pred_batches, class_names)

# %% Model Save
model.save(MODEL_PATH)
# %%
