import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

df_true=pd.read_csv("True.csv")
df_fake=pd.read_csv("Fake.csv")

# print("true dataset size: ")
# print(df_true.size)
# print("fake dataset size: ")
# print(df_fake.size)

# print(df_true.info())
# print(df_fake.info())

df_true['isFake']=0
print(df_true.head())

df_fake['isFake']=1
print(df_fake.head())

df=pd.concat([df_true,df_fake]).reset_index(drop=True)
print(df)

df.drop(columns=['date'],inplace=True)
df['original']=df['title']+' '+df['text']
print(df['original'][0])

nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])
print(stop_words)

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    
    return result

df['clean']=df['original'].apply(preprocess)
print(df['original'][0])
print(df['clean'][0])

#find total words
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
print(list_of_words)

len(list_of_words)