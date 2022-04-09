import pickle
import pandas as pd

import re
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("data/train.csv")

lm = WordNetLemmatizer()
def text_transformation(data):
    c = []
    for sentence in data:
        new_item = re.sub('[^a-zA-Z]', ' ', str(sentence))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        c.append(' '.join(str(x) for x in new_item))
    return c

corpus = text_transformation(data["text"][700000:900000])
cv = CountVectorizer(ngram_range = (1,2))
cv.fit_transform(corpus)

filename = "countvectorizer.sav"
pickle.dump(cv, open(filename, 'wb'))
print("Saved Model")