#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
train = pd.read_csv('train.tsv', sep ='\t')


# In[ ]:


Rotten tomatoes dataset. I want to figure out the general opinion of the public ce[]


# In[4]:


train.head()


# In[5]:


train.describe()


# In[13]:


train.info()


# In[14]:


data = pd.read_csv('data.csv')


# In[17]:


data.head()


# In[20]:


#Pre-Prcoessing and Bag of Word Vectorization using Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(data['Sentence'])


# In[22]:


#Splitting the data into trainig and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.25, random_state=5)
#Training the model


# In[23]:


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train, Y_train)
#Caluclating the accuracy score of the model
from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, Y_test)
print("Accuracuy Score: ",accuracy_score)


# Using LSTM-Based Models
# Though we were able to obtain a decent accuracy score with the Bag of Words Vectorization method, it might fail to yield the same results when dealing with larger datasets. This gives rise to the need to employ deep learning-based models for the training of the sentiment analysis model.
# 
# For NLP tasks we generally use RNN-based models since they are designed to deal with sequential data. Here, we’ll train an LSTM (Long Short Term Memory) model using TensorFlow with Keras. The steps to perform sentiment analysis using LSTM-based models are as follows:
# 
# Pre-Process the text of training data (Text pre-processing involves Normalization, Tokenization, Stopwords Removal, and Stemming/Lemmatization.)
# Import Tokenizer from Keras.preprocessing.text and create its object. Fit the tokenizer on the entire training text (so that the Tokenizer gets trained on the training data vocabulary). Generated text embeddings using the texts_to_sequence() method of the Tokenizer and store them after padding them to an equal length. (Embeddings are numerical/vectorized representations of text. Since we cannot feed our model with the text data directly, we first need to convert them to embeddings)
# After having generated the embeddings we are ready to build the model. We build the model using TensorFlow — add Input, LSTM, and dense layers to it. Add dropouts and tune the hyperparameters to get a decent accuracy score. Generally, we tend to use ReLU or LeakyReLU activation functions in the inner layers of LSTM models as it avoids the vanishing gradient problem. At the output layer, we use Softmax or Sigmoid activation function.
# Code for Sentiment Analysis using LSTM-based model approach:
# 
# Here, we have used the same dataset as we used in the case of the BOW approach. A training accuracy of 0.90 was obtained.

# In[ ]:




#Importing necessary libraries
import nltk
import pandas as pd
from textblob import Word
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
#Loading the dataset
data = pd.read_csv('Finance_data.csv')
#Pre-Processing the text 
def cleaning(df, stop_words):
    df['sentences'] = df['sentences'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Replacing the digits/numbers
    df['sentences'] = df['sentences'].str.replace('d', '')
    # Removing stop words
    df['sentences'] = df['sentences'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    # Lemmatization
    df['sentences'] = df['sentences'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    return df
stop_words = stopwords.words('english')
data_cleaned = cleaning(data, stop_words)
#Generating Embeddings using tokenizer
tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(data_cleaned['verified_reviews'].values)
X = tokenizer.texts_to_sequences(data_cleaned['verified_reviews'].values)
X = pad_sequences(X)
#Model Building
model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(704, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='LeakyReLU'))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())
#Model Training
model.fit(X_train, y_train, epochs = 20, batch_size=32, verbose =1)
#Model Testing
model.evaluate(X_test,y_test)


# In[25]:


df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df.head()


# In[26]:


import numpy as np
def create_sentiment (rating):
    if rating == 1 or rating == 2:
        return -1
    elif rating == 4 or rating == 5:
        return 1
    else:
        return 0


# What I did here is to create a column for "Sentiment." If the rating is 1 0r 2, then it is negative, if it is 4 or 5, then it is positive, 
# if it is 3, then it is neutral.

# In[27]:


df['Sentiment'] = df['Rating'].apply(create_sentiment)


# In[28]:


df.head()


# To proceed, I have to remove the characters, digits, basically anything that isn't words
