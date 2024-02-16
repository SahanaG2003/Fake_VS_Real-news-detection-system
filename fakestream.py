import pandas as pd
import re
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Access and Read the datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Assign the fake and real data to binary values
data_fake['class'] = 0
data_true['class'] = 1

# Merge datasets
data_fake_manual_testing = data_fake.tail(10)
data_true_manual_testing = data_true.tail(10)

data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1

data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)
data['text'] = data['text'].apply(lambda x: x.lower())

# Text Preprocessing function
def wordopt(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Split data
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Model
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Streamlit App
st.title("Fake vs. Real News Classifier")

# Text Input for User
user_input = st.text_area("Enter News Text:", height=200)

# Function to classify news and display the result
def classify_news(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    return pred_LR[0]

# Classify Button
if st.button("Classify"):
    result = classify_news(user_input)
    st.write("Prediction: {}".format("Fake News" if result == 0 else "Real News"))
    