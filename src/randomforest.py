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

from sklearn.model_selection import train_test_split
#Importing modules to use RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix,classification_report


import warnings
warnings.filterwarnings('ignore')
  
np.random.seed(42)
#tf.random.set_seed(42)

df = pd.read_csv('data\SMSSpamCollection',sep = '\t',header = None , names = ['label','message'])#Giving names to the columns
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


train_Y = (train_Y == 'spam').astype(int)
test_Y = (test_Y == 'spam').astype(int)

#define the model parameters

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_X)
X_test_tfidf = tfidf.transform(test_X)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tfidf, train_Y)
rf_acc = accuracy_score(test_Y, rf.predict(X_test_tfidf))
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")#Accuracy prediction


pred_Y = rf.predict(X_test_tfidf)
cm = confusion_matrix(test_Y, pred_Y)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Random Forest - Confusion Matrix')
plt.show()
plt.close()

feature_names = tfidf.get_feature_names_out()#all important words 
importances = rf.feature_importances_#importance score is given to each word
indices = importances.argsort()[-10:][::-1]#sort in ascending order by importance score we take last 5 words and reverse the list

#Visually representing important words
plt.figure(figsize=(10, 5))
plt.bar(range(10), importances[indices])
plt.xticks(range(10), [feature_names[i] for i in indices])
plt.title('Top 10 Important Words - Random Forest')
plt.tight_layout()
plt.show()
plt.close()

#Saving the randomforest model
import joblib

joblib.dump(rf, 'random_forest.pkl')#all the decisions made by randomforest
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')#text to numbers mapping