import twint
import pandas as pd
import re
import csv 
from nltk.tokenize import word_tokenize
import emoji
import string
from nltk.tokenize import TweetTokenizer #tokeniser le texte
import nltk
import matplotlib.pyplot as plt
from spellchecker import SpellChecker  #corriger les fautes d'orthographe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn import model_selection, naive_bayes, svm
import seaborn as sns 



df=pd.read_csv('C:\\Users\\User\\Desktop\\PFE\\model_Data.csv')

#_________________________________________________________ EXPLORER LES DONNEES _____________________________
#print(df.info())
df['clean_text'] = df['clean_text'].astype(str)

#_________________________________________________________ SUPPRIMER LES LIGNES AVEC DES VALEURS NULLES__________________________________________

df = df[df['clean_text'].notna()]
df = df[df['category'].notna()]

#__________________________________________________________ NETTOYER LE TEXTE_______________________________________________________________________ 

def remove_tags(string):       
    string = re.sub('@[^\s]+','',string) #enlever les identifiants
    string = re.sub('http[^\s]+','',string)   #remove URLs
    string = re.sub("[^-9A-Za-z ]", "" , string)    #remove non-alphanumeric characters 
    string = re.sub("\s+"," ", string) #enlever les espaces inutiles 
    string = string.lower()
    return string
df['clean_text']=df['clean_text'].apply(lambda cw : remove_tags(cw)) 

#____________________________________________________________ STOP WORDS ________________________________________________________________________

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#_____________________________________________________________ LEMMATISER LE TWEET ______________________________________________________________


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st
df['clean_text'] = df.clean_text.apply(lemmatize_text)


#sns.countplot(x=df["category"]) 
#import matplotlib.pyplot as plt
#plt.show()

x=df['clean_text'].values
y=df['category'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=50) # je precise le pourcentage 

#print(x_train[0])

vect = TfidfVectorizer()
x_train_vectorized = vect.fit_transform(x_train)
x_test_vectorized = vect.transform(x_test)

#__________________________________________________________________SUPERVISED___K neearest neighbors_____________________________________________________________

"""
from sklearn.neighbors import KNeighborsClassifier

# 2. instantiate the model (with the default parameters)
knn = KNeighborsClassifier()

# 3. fit the model with data (occurs in-place)
knn.fit(x_train_vectorized, y_train)

prediction_knn=knn.predict(x_test_vectorized)
#print("knn Accuracy Score -> ",accuracy_score(prediction_knn,y_test)*100)
"""
#__________________________________________________________________SUPERVISED___SVM_____________________________________________________________
 
"""
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train_vectorized,y_train)
#predictions_SVM = SVM.predict(x_test_vectorized) #predict the labels on validation dataset
"""
#__________________________________________________________________ SUPERVISED ___ NAIVE BAYES _____________________________________________________________

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train_vectorized,y_train)
#prediction_naive_bayes=clf.predict(x_test_vectorized)



#MATRICE DE CONFUSION 


"""
from sklearn.metrics import plot_confusion_matrix

matrix=plot_confusion_matrix(knn,x_test_vectorized,y_test)

matrix.ax_.set_title('Matrice de confusion',color="blue")
plt.xlabel('Predicted label',color="blue")
plt.ylabel('True label',color="blue")
plt.gcf().axes[0].tick_params(colors="blue")
plt.gcf().axes[1].tick_params(colors="blue")
plt.gcf().set_size_inches(10,6)
plt.show()
"""


#33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
data=pd.read_json('file.json',lines=True)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#___________________________________________________________________ PARTIE PRETRAITEMENT _________________________________________________________________________________
'''lien de pretraitement de texte en itilisant la bib NLTK : https://www.analyticsvidhya.com/blog/2020/11/text-cleaning-nltk-library/'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#____________________________________________________________________ FEATURE SELECTION _____________________________________________________________________________________________________________________

data=data[data["language"] == "en"]
new_data=data.drop(['name','likes_count','geo','source',
'thumbnail','hashtags','cashtags','retweet','quote_url',
'id','link','user_rt_id','user_rt','retweet_id','reply_to',
'conversation_id','created_at','time','timezone','user_id',
'username','place','urls','photos','video','language',
'mentions','translate','replies_count','retweets_count',
'retweet_date','trans_src','trans_dest'], axis=1)
#print(new_data.head())
#print(new_data.info())



#____________________________________________________________________ Enlever les noms d'utilisateurs, caracteres speciaux et URLs, espaces inutils _______________________________________
def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet) #enlever les identifiants
    tweet = re.sub('http[^\s]+','',tweet) #enlever les URL
    tweet = re.sub("[^-9A-Za-z ]", "" , tweet)
    tweet = "".join([i.lower() for i in tweet if i not in string.punctuation]) #convertir en minuscule car python est sensible a la case 
    tweet = re.sub("\s+"," ", tweet)#enlever les espaces inutiles 
    return tweet

new_data['tweet'] = new_data['tweet'].apply(remove_usernames_links)


#____________________________________________________________________  EMOJI -->WORDS ___________________________________________________________________________________ 

def emoji_transform(tweet):
  tweet=emoji.demojize(tweet, delimiters=("", ""))
  tweet = re.sub('_|\-',' ',tweet)
  return tweet
new_data['tweet'] = new_data['tweet'].apply(emoji_transform)

#____________________________________________________________________  TOKENISATION ___________________________________________________________________________________ 
def tokeniseur(text):
  tweet = TweetTokenizer()
  text=tweet.tokenize(text)
  return text
new_data['tweet'] = new_data['tweet'].apply(tokeniseur) 


#___________________________________________________________________Corriger les fautes d'orthographe___________________________________________________________________
spell = SpellChecker(distance=1)
def Correct(tweet):
  for x in tweet:
    x=spell.correction(x)
  return tweet
new_data['tweet'] = new_data['tweet'].apply(Correct)
#___________________________________________________________________ Enlever les STOP WORDS ( mots vides )___________________________________________________
def stopword(tweet):
  stopwords = nltk.corpus.stopwords.words('english')
  tweet = [i for i in tweet if i not in stopwords]
  return tweet
new_data['tweet'] = new_data['tweet'].apply(stopword)




#___________________________________________________________________Lemmatization et Stemming____________________________________________________

 

def stem_lema(tweet):
  ss = nltk.SnowballStemmer(language = 'english')#stemming ,enleve les prefixes et suffixes
  tweet = [ss.stem(word) for word in tweet]

  wn = nltk.WordNetLemmatizer()
  w = [wn.lemmatize(word) for word in tweet]#Lemmatization:racine du mot,  forme canonique du mot , ex L'infinitif est la forme canonique d'un verbe français.
  return tweet


new_data['tweet'] = new_data['tweet'].apply(stem_lema)


print(new_data.info())


def join(tweet):
  tweet=' '.join(tweet)
  return tweet

new_data['tweet'] = new_data['tweet'].apply(join)

f=new_data['tweet'].values

f = vect.transform(f)

 
yfit =clf.predict(f)

#print("pourcentage de tweets positifs :")
#print((np.count_nonzero(yfit == 1)*100)/yfit.size)
#print("pourcentage de tweets neutres :")
#print((np.count_nonzero(yfit == 0)*100)/yfit.size) 
#print("pourcentage de tweets negatifs :")
#print((np.count_nonzero(yfit == -1)*100)/yfit.size)
#
##print(yfit)
sns.countplot(yfit) 
import matplotlib.pyplot as plt
plt.show()
#print("Naive Bayes score :")
#print(classification_report(y_test,prediction_naive_bayes))