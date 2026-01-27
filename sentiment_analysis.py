import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from textblob import TextBlob
t = "i have finished my work so i am so happy"
sentiment = TextBlob(t).sentiment
print(sentiment)

import pandas as pd

df = pd.read_csv('IMDB Dataset.csv', nrows=5000)
df.head(10)

df

df.tail()

df.describe()

df.info()

df.shape

df.fillna(0, inplace=True)

df.info()

df['review'][0]

df['sentiment'].replace({'positive':1, 'negative':0}, inplace=True)

df.head()

import re
print(df.iloc[0].review)

clean=re.compile('<.*?>')
re.sub(clean,'',df.iloc[0].review)

def clean_html(text):
    clean=re.compile('<.*?>')
    return re.sub(clean,'',text)

df['review']=df['review'].apply(clean_html)

df.head()

def convert_lower(text):
    return text.lower()

df['review']=df['review'].apply(convert_lower)

df.head()

def remove_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+' '
    return x

remove_special('Th%e @ classic use of the word. it is called oz as that is the nickname given to the oswald maximum security state')

df['review']=df['review'].apply(remove_special)

df.head()

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')

def remove_stopwords(text):
    x=[]
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y

df['review']=df['review'].apply(remove_stopwords)

df.head()

y=[]
from nltk.stem import PorterStemmer
ps = PorterStemmer()
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z

stem_words(['I','loved','loving','went', 'mentioned'])

df['review']=df['review'].apply(stem_words)

df.head()

def join_back(list_input):
    return " ".join(list_input)

df['review']=df['review'].apply(join_back)

df.head()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=150)

X=cv.fit_transform(df['review']).toarray()

X[1:5,:]

X.shape

X[0].mean()

y=df.iloc[:,-1].values

y

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2)

y_train.shape

X_train.shape

X_test.shape

from sklearn.naive_bayes import GaussianNB

clf1=GaussianNB()

clf1.fit(X_train,y_train)

y_pred1=clf1.predict(X_test)

from sklearn.metrics import accuracy_score

print("Gaussian",accuracy_score(y_test,y_pred1))