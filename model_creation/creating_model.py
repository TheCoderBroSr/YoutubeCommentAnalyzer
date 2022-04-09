def rpint(msg):print(f"\n{msg}\n")

import pandas as pd
import pickle

data = pd.read_csv("data/train.csv")

rpint(data.info())
rpint(data.head())
rpint(data['polarity'].value_counts())

import re
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()

def text_transformation(data):
    corpus = []
    for sentence in data:
        new_item = re.sub('[^a-zA-Z]', ' ', str(sentence))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus

print("Here")
corpus = text_transformation(data["text"][700000:900000])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (1,2))

print("HERERERERE")

x = cv.fit_transform(corpus)
y = data['polarity'][700000:900000]

print(y.value_counts())

parameters = {
    "max_features": 'auto',
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True
}

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    rpint("Fitting Data into RFC")
    rfc = RandomForestClassifier(max_features = parameters['max_features'], n_estimators=parameters['n_estimators'], min_samples_split=parameters['min_samples_split'], min_samples_leaf = parameters['min_samples_leaf'], bootstrap = parameters['bootstrap'])
    rfc.fit(x,y)
    rpint("Model done")

    #Saving Model
    filename = 'trial_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))
    rpint("Saved Model")