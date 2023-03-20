from fastapi import FastAPI, Depends, File, Form, Query,UploadFile,BackgroundTasks,Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from asyncer import asyncify
#important libraries

import pickle
from operator import index
import pandas as pd
import numpy as np
import os
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download necessary data for natural language processing tasks

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

#init app
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000",
    "157.245.96.101",
]

# Load the pickle file
with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the pickle file
with open('./cv.pkl', 'rb') as f:
    cv = pickle.load(f)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class CommonQueryPostParams:
    def __init__(
        self,
        sms_text:str = Form(default="",description="sms text "),
    ):
        self.sms_text = sms_text


def preprocess_data(df):
    wordnet_lem = WordNetLemmatizer()

    df = pd.DataFrame([{'text': df[0]['text']}])
    reg_vars = ['http\S+', 'www\S+', 'https\S+', '\W\s+', '\d+', '\t+', '\d+', '\-+', '\\+', '\/+', '\"+', '\#+', '\++', '\@+', '\$+',  '\%+', '\^+', '\&+', '\*+', '\(+', '\)+', '\[+', '\]+', '\{+', '\}+', '\|+', '\;+', '\:+', '\<+', '\>+', '\?+', '\,+', '\.+', '\=+',     '\_+', '\~+', '\`+', '\s+']
    df['text'].replace(reg_vars, ' ', regex=True, inplace=True)
    df['text'] = df['text'].astype(str).str.lower()
    df = df[df['text'].map(lambda x: x.isascii())]
    df['text'] = df.apply(lambda column: nltk.word_tokenize(column['text']), axis=1)
    stopwords = nltk.corpus.stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stopwords])
    df['text'] = df['text'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    df['text'] = df['text'].apply(wordnet_lem.lemmatize)

    processed_data = cv.transform(df['text']).toarray()

    return processed_data

def predictSpam(commons: CommonQueryPostParams) -> Response:

    # data = 
    # with open(f"bgremoved/{filename}", "w") as f:
    #     f.write(data)
    data = {"text": commons.sms_text}
    data = [data]
    # Preprocess the data
    processed_data = preprocess_data(data)

    # Use the model to make a prediction
    prediction = model.predict(processed_data)

    # Convert the NumPy array to a Python list
    prediction_list = prediction.tolist()

    # Return "ham" or "spam" depending on the prediction
    if prediction_list[0] == 0:
            return {"spam_score":0,"spam_text":"not spam"}
    else:
            return {"spam_score":1,"spam_text":"spam"}


@app.post("/api/detect")
async def detect_spam(request: Request, commons: CommonQueryPostParams = Depends()):
    response =  await asyncify(predictSpam)(commons)
    return response
    # return {"filename": file_path, "progress": progress}