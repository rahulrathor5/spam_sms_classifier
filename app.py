import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

import requests

tfid=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title('sms spam classifier')

input_sms=st.text_input('enter the message')

#preprocess

ps=PorterStemmer()

if st.button('predict'):
 def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if  i.isalnum():
            y.append(i)
    #this is used to clone the list you cannot copy the list it is mutable
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    text=y[:]
    y.clear()
    for i in text:
         y.append(ps.stem(i))
    return " ".join(y)

 transformed_sms=transform_text(input_sms)


#vectorize

 vecter_input=tfid.transform([transformed_sms])

#predict

 result=model.predict(vecter_input)[0]

#display

 if result==1:
    st.header('spam')
 else:
    st.header('not spam')