from operator import index
import pandas as pd
import langdetect as detect

import numpy as np
import pickle
import os

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

df = pd.read_csv('W:/ShunyankTechnologies/Hackathons/CyberDost/ml-models/sms-spam-detector/spam.csv',
                 sep=',', header=0, on_bad_lines='skip', encoding = "ISO-8859-1")

unnamed_cols = df.columns[df.columns.str.contains("Unnamed")]
df.drop(columns=unnamed_cols, inplace=True)


reg_vars = ['http\S+', 'www\S+', 'https\S+', '\W\s+', '\d+', '\t+', '\d+', '\-+', '\\+', '\/+', '\"+', '\#+', '\++', '\@+', '\$+', '\%+', '\^+', '\&+', '\*+', '\(+', '\)+', '\[+', '\]+', '\{+', '\}+', '\|+', '\;+', '\:+', '\<+', '\>+', '\?+', '\,+', '\.+', '\=+', '\_+', '\~+', '\`+', '\s+']

df.replace(reg_vars, ' ', regex=True, inplace=True)

df.drop_duplicates(inplace=True)

df.replace('', np.nan, inplace=True)

df.dropna(inplace=True)
df = df[df['v2'].map(lambda x: x.isascii())]
for i in range(len(df)):
    try:
        ['v2'][i] = detect.detect(df['v2'][i])
        if df['v2'][i] != 'en':
            df.drop(i, inplace=True, index=False)
    except:
        pass

df['v2'] = df['v2'].astype(str).str.lower()    
stopwords = nltk.corpus.stopwords.words("english")
df['TokenSMS'] = df.apply(lambda column: nltk.word_tokenize(column['v2']), axis=1)
df['TokenSMS'].head(2)
df['StopTokenSMS'] = df['TokenSMS'].apply(lambda x: [item for item in x if item not in stopwords])
df['LengthTokenSMS'] = df['StopTokenSMS'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
wordnet_lem = WordNetLemmatizer()
df['LemTokenSMS'] = df['LengthTokenSMS'].apply(wordnet_lem.lemmatize)

reg_vars = ['http\S+', 'www\S+', 'https\S+', '\W\s+', '\d+', '\t+', '\d+', '\-+', '\\+', '\/+', '\"+', '\#+', '\++', '\@+', '\$+', '\%+', '\^+', '\&+', '\*+', '\(+', '\)+', '\[+', '\]+', '\{+', '\}+', '\|+', '\;+', '\:+', '\<+', '\>+', '\?+', '\,+', '\.+', '\=+', '\_+', '\~+', '\`+', '\s+']

df.replace(reg_vars, ' ', regex=True, inplace=True)

df.replace('', np.nan, inplace=True)

df.dropna(inplace=True)
cv = CountVectorizer()
x = cv.fit_transform(df['LemTokenSMS']).toarray()
print(x.shape)
df['v1'] = df['v1'].replace({'spam': 1, 'ham': 0})
y = df['v1'].values

print(y.shape)
y = y.astype('int')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Initialize a MultinomialNB objec

mnb=MultinomialNB()
# Training the classifier and making predictions on the test data

mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred))
with open('model.pkl', 'wb') as file:
    pickle.dump(mnb, file)

# Save the model to a file
with open('cv.pkl', 'wb') as file:
    pickle.dump(cv, file)