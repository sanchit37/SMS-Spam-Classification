# pip install -U streamlit
# pip install -U plotly

# you can run your app with: streamlit run app.py

import numpy as np
import pandas as pd

df  = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.tsv', sep='\t')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ham = df[df['label']=='ham']
spam = df[df['label']=='spam']
ham = ham.sample(spam.shape[0])
data = pd.concat([ham, spam], axis=0, ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=0, shuffle=True, stratify=data['label'])

# model Building
clf = Pipeline([('tfidf', TfidfVectorizer()),
                ('rfc', RandomForestClassifier(n_estimators=100, n_jobs=-1))])

clf.fit(X_train, y_train)



import streamlit as st
# import pickle

# loading the trained model
# model = pickle.load(open('model.pkl', 'rb'))

# create title
st.title('Welcome to our site.\n enter the message below to ensure if its a legit(ham) or a spam you recieved.')

message = st.text_input('Enter a message')

submit = st.button('Predict')

if submit:
    prediction = clf.predict([message])

    # print(prediction)
    # st.write(prediction)
    
    if prediction[0] == 'spam':
        st.warning('This message is spam')
    else:
        st.success('This message is Legit (HAM)')
        st.balloons()
