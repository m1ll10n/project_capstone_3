![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# Concrete Crack Classification using Transfer Learning

This GitHub repository contains the code and resources for a machine learning project that aims to detect concrete cracks in building structures. By using transfer learning techniques, an image classification model that can accurately classify concrete images into those with or without cracks is developed. The dataset used for this project is publicly available and is linked in the Credits section. The trained model achieves a validation accuracy of over 90% while avoiding overfitting. The model is also deployed and tested on a small set of test data.

I think that this effort will help to improve building safety and resilience by giving an automated and precise method of detecting possible structural concerns. And helping to showcase the ability to run AI model on Image Classification problems.


## Applications
Below are the steps taken on solving the task.

### 1. Exploratory Data Analysis
Checking a sample of images with two classes (Positive, Negative) from the dataset. Positive class indicating there is cracks and vice versa.

![Data_Inspect](https://user-images.githubusercontent.com/49486823/226516511-4445df08-5e8a-4920-a300-1b5a5426572a.png)

### 2. Data Preprocessing
There are three(3) steps of data preprocessing:\
2.1 Splitting the dataset into train, validation, test sets with the ratio of 3:1:1.\
2.2 Loading the dataset into PrefetchDataset for faster processing time\
2.3 (Optional) Adding data augmentation model as a layer for image classification model. Which includes: Random flip and random rotation of images.

### 4. Model Development
This is the model architecture. Few notable settings not included in the screenshot:\
4.1 MobileNetV2 model is used for image processing and features extraction\
4.2 Data Augmentation are set as optional and thus, having two models as comparison for performance\
4.3 Activation function is Softmax Activation Function\
4.4 Optimizer is Adam Optimization\
4.5 Loss function is Categorical Cross-Entropy Function\
4.6 No early stopping implemented with only 5 epochs of training

![A_model](https://user-images.githubusercontent.com/49486823/226516605-c111d79c-2706-498d-9744-9c05a78c1289.png)\
Augmentation model

![NA_Model](https://user-images.githubusercontent.com/49486823/226516559-0dca6060-0983-4895-8a23-861dd54d1a92.png)\
No Augmentation model
## Results
This section shows all the performance of the model and the reports.
### Training Logs
Before the training begins, we should compare the model's accuracy on the dataset. Augmentation model has a higher model accuracy compared to No Augmentation model.

![A_Before_Training](https://user-images.githubusercontent.com/49486823/226516949-1c27e334-6004-4b86-9b21-83bc9dbed56c.jpg)\
Augmentation model

![NA_Before_Training](https://user-images.githubusercontent.com/49486823/226516992-a113e595-76fd-40ac-83d1-75dc7476a6bc.jpg)\
No Augmentation model


The Augmentation model shows good training accuracy with no signs of underfitting and overfitting.\
![A_Training_Accuracy](https://user-images.githubusercontent.com/49486823/226517208-3df45865-ab80-4845-9d54-ef0e11ede67b.jpg)
![A_Training_Loss](https://user-images.githubusercontent.com/49486823/226517217-ab15b023-ae52-4036-9e10-0bc3778fb393.jpg)

And the No Augmentation model also displayed good training accuracy. Although, the train accuracy is higher than validation accuracy, this does not have or only a small signs of overfitting as the difference between the two accuracies are minimal (< 0.001).\
![NA_Training_Accuracy](https://user-images.githubusercontent.com/49486823/226517257-6ca14b96-8a91-4f60-ae51-88d3bcc813bf.jpg)
![NA_Training_Loss](https://user-images.githubusercontent.com/49486823/226517264-fbc912df-9d26-4fe8-9ff5-f288e9df69be.jpg)


### Accuracy & F1 Score
The Augmentation model recorded an accuracy of 0.9979 and f1 score of 0.9979.\
![A_Accuracy_F1](https://user-images.githubusercontent.com/49486823/226517326-04158a6d-7056-4008-b3a7-13b6f22df7bd.jpg)


The No Augmentation model recorded an accuracy of 0.9986 and f1 score of 0.9986.\
![NA_Accuracy_F1](https://user-images.githubusercontent.com/49486823/226517348-247890c2-c820-4d9e-a42c-23d57b839616.jpg)


### Classification Report
Both models recorded 1.00 scores throughout the report (As both shares the same scores, only a picture is shown).\
![NA_Classification_Report](https://user-images.githubusercontent.com/49486823/226517365-918302a5-26d6-4753-b205-26894dc5f52f.jpg)


### Confusion Matrix
The confusion matrix showed the model are good at classifying both Positive and Negative classes.\
![A_Confusion_Matrix](https://user-images.githubusercontent.com/49486823/226517435-23e816d9-4a3d-4d71-8adf-bf62e106fe18.png)\
Augmentation model

![NA_Confusion_Matrix](https://user-images.githubusercontent.com/49486823/226517443-6b19838b-15f9-445c-83a5-96ffa837923e.png)\
No Augmentation model

### Discussion
![A_Training_Process](https://user-images.githubusercontent.com/49486823/226517531-0687e065-6f57-4410-a47f-93d8c9ef5a39.jpg)\
Augmentation model

![NA_Training_Process](https://user-images.githubusercontent.com/49486823/226517538-762da1a7-4284-482f-829e-3ef529ade239.jpg)\
No Augmentation model

Two models were compared, one with data augmentation (A) and one without (NA), and the training time for the A model was found to be 1.6875 times slower than the NA model. Despite this longer training time, the A model only showed a very slight improvement in training accuracy of less than 0.001 compared to the NA model. Therefore, the NA model was determined to be the better model, as it achieved the same performance on the test set but with less time spent on training.

## Deploy
The NA model is deployed and the picture below are shown with the true label and the model prediction.
![output](https://user-images.githubusercontent.com/49486823/226518061-17a51d29-5ea9-465c-a682-78687cdf106d.png)

## Credits
Data can be obtained from https://data.mendeley.com/datasets/5y9wdsg2zt/2 .
