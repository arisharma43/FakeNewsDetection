#imports

from lib2to3.pgen2 import token
import nltk
nltk.download('punkt')


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import plotly.express as px
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
#import sklearn for splitting train and test data
from sklearn.model_selection import train_test_split



df_true=pd.read_csv("True.csv")
df_fake=pd.read_csv("Fake.csv")

# print("true dataset size: ")
# print(df_true.size)
# print("fake dataset size: ")
# print(df_fake.size)

# print(df_true.info())
# print(df_fake.info())

df_true['isFake']=0
#print(df_true.head())

df_fake['isFake']=1
#print(df_fake.head())

df=pd.concat([df_true,df_fake]).reset_index(drop=True)
#print(df)

df.drop(columns=['date'],inplace=True)
df['original']=df['title']+' '+df['text']
#print(df['original'][0])

nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])
#print(stop_words)

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    
    return result

df['clean']=df['original'].apply(preprocess)
print(df['original'][2])
print(df['clean'][2])

#find total words
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
# print(list_of_words)

len(list_of_words)

total_words=len(list(set(list_of_words)))
# print(total_words)

#join the words that are seperated by a space
df['clean_joined']=df['clean'].apply(lambda x: " ".join(x))
print(df['clean_joined'][2])

plt.figure(figsize=(8,8))
sns.countplot(y="subject",data=df)
#sns.countplot(y="isFake",data=df)

print(nltk.word_tokenize(df['clean_joined'][2]))

maxlen=-1
for doc in df.clean_joined:
    tokens=nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen=len(tokens)
print("The max number of words in the doc is = ", maxlen)

#fig=px.histogram(x=[len(nltk.word_tokenize(x)) for x in df.clean_joined],nbins=100)

x_train,x_test,y_train,y_test=train_test_split(df.clean_joined,df.isFake,test_size=0.2)

# Create a tokenizer to tokenize the strings into integers
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print(len(train_sequences))
print(len(test_sequences))

#maxlen=4405 works well based off of experimentation. maxlen=40 also works well
padded_train = pad_sequences(train_sequences,maxlen=4405,padding='post',truncating='post')
padded_test = pad_sequences(test_sequences,maxlen=4405,truncating='post')

for x,doc in enumerate(padded_train[:2]):
    print("The padded encoding for docmument",x+1," is : ",doc)