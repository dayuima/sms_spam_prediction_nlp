# SMS Spam Prediction

## Project Objective
This project was created as an assignment to develop a machine learning model that can detect spam messages in SMS. The main goal of this project is to build a system that is effective in identifying and separating spam messages from legitimate (non-spam) messages in SMS datasets.

## Problem Solved
Spam messages in SMS are becoming an increasingly troubling problem for mobile phone users. In this project, we seek to address the problem by building a machine learning model that can automatically separate spam messages from non-spam messages, so that users can easily identify and avoid these unwanted messages.

## Background of the Problem
Spam messages in SMS have a significant negative impact. Receiving spam messages repeatedly can invade users' privacy, and waste time in checking and deleting unwanted messages. Therefore, it is important to have a tool that can filter and detect spam messages effectively.

## Project Output
The output of this project is a machine learning model that can predict whether an SMS message is spam or not. The model accepts the message text as input and outputs a prediction indicating whether the message falls into the spam or non-spam category. In addition, the evaluation results show that the model has a high accuracy rate, reaching 97%, in classifying SMS messages as spam or non-spam. With such a high accuracy rate, this model can effectively detect and filter spam messages very well.

## Data Used
The data used in this project is taken from Kaggle and can be accessed via the following link: [SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). This dataset contains a number of SMS that are categorized as spam or non-spam. Each entry in the dataset includes the message text and a label indicating whether the message is spam or non-spam.

## Methods Used
The methods to be used in this project include data preprocessing and machine learning stages. 
1. Preprocessing: cleaning the dataset from null values and duplicates to ensure data cleanliness. Next, the text will be converted to lowercase and special characters, punctuation marks, unnecessary characters, and linking words will be removed. In addition, a stemming process will be performed to convert the words into their base form.
2. Machine learning: using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architecture models for classification. These two models are a type of RNN (Recurrent Neural Network) architecture that is well suited to cope with sequential data such as text.

With this approach, this project will produce a model that can accurately classify whether an SMS message is spam or not.

## Stack Used
In this project, we will use several technologies and tools as follows:

- Programming Language: Python
- Library and Framework: Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn, Feature-Engine, TensorFlow, Keras, NLTK.
- Development Environment: Jupyter Notebook

## Directory Structure and Brief Description of Files
This project has the following directory structure:
- `data/sms_spam.csv` contains the SMS Spam dataset used in the project.
- `notebooks/h8dsft_P2M2_dayuima.ipynb` is a Jupyter notebook that contains the complete project code, including data pre-processing, model building, and evaluation.

## Project Advantages and Disadvantages
The advantages of this project are:

- Has the potential to assist users in identifying and avoiding spam messages in SMS.
- Uses an existing dataset, eliminating the need to collect data manually.
- Applies common machine learning techniques and can be applied to similar problems.

The drawbacks of this project are:

- The model may not be completely accurate in classifying messages as spam or non-spam.
- Depends on the quality and representativeness of the dataset used.

## Supporting Links
- [Deployment](https://huggingface.co/spaces/dayuima01/M2P2)
- [Abbreviation ](https://www.kaggle.com/code/life2short/data-processing-replace-abbreviation-of-word/notebook)
- [Chatwords](https://www.kaggle.com/code/niteshk97/nlp-text-preprocessing#Step-5--Chat-word)