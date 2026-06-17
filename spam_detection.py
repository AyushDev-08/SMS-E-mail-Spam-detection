import os #Suppresses the warning symbols
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tf_keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')
  
np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv('SMSSpamCollection',sep = '\t',header = None , names = ['label','message'])#Giving names to the columns
print(df.head())
print(df.shape)

sns.countplot(x = 'label',data = df)#the graph visualisation of spam and ham 
plt.show()

ham = df[df['label'] == 'ham']
spam = df[df['label'] == 'spam']

ham_balanced = ham.sample(n=len(spam),random_state=42)#downsampling ham to match spam 

#Combine balanced data
balance = pd.concat([ham_balanced,spam]).reset_index(drop = True)

sns.countplot(x =  'label',data = balance)
plt.title("Balanced Distribution of Spam and Ham Emails")
plt.xticks(ticks = [0,1], labels = ['Not Spam(Ham)','Spam'])#the positions of labels    
plt.show()
plt.close()

#Cleaning and preprocessing the text
punctuations_list =  string.punctuation
def remove_punctuations(message):
    temp = str.maketrans('','',punctuations_list)#1 and 2 parameters are for replacement , 3rd one for deleting the string
    return message.translate(temp)#apply the changes made in maketrans

balance['message'] = balance['message'].apply(lambda x:remove_punctuations(x))#calling the remove_punctuations function

#Converting to lower case and remove stopwords
def remove_stopwords(message):
    stop_words = stopwords.words('english')

    imp_words = []

    for i in str(message).split():
        i = i.lower()

        if i not in stop_words:
            imp_words.append(i)

    output =  " ".join(imp_words)
    return output
balance['message'] =  balance['message'].apply(lambda text:remove_stopwords(text))

# visualization of word cloud
def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['message'])
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus)
    plt.figure(figsize=(7, 7))#size of the plot
    plt.imshow(wc, interpolation='bilinear')#image show and bilinear shows smooth,blended edges
    plt.title(f'WordCloud for {typ} Emails', fontsize=15)
    plt.axis('off')
    plt.show()
    plt.close()
plot_word_cloud(balance[balance['label'] == 'ham'], typ='Non-Spam')
plot_word_cloud(balance[balance['label'] == 'spam'], typ='Spam')

#Modifying the data to be used by the ML models (Tokenization and Padding)

train_X,test_X,train_Y,test_Y = train_test_split(balance['message'],balance['label'],test_size = 0.2,random_state=42)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = tokenizer.texts_to_sequences(train_X)#Converts the training data string to tokens
test_sequences = tokenizer.texts_to_sequences(test_X)#same but for testing data to tokens

max_len = 50
#pad_sequences makes sure all the tokens are of same by adding 0's 
#padding adds 0 if too short and truncating remove 0's if tokens length exceededs max_len
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')


train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)

#define the model parameters

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_len),#each word becomes a 64 number vector
    #LSTM Long Short Term Memory remembers context 
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.3), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),#measures how wrong the models is
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),#adjusts weight after each batch to reduce loss
    metrics=['accuracy']
)

es = EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20,
    batch_size=32,
    callbacks=[lr, es]
)

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()