from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import validators, re, pickle

API_KEY = "YOUR KEY"
CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def load_file(filename):
    return pickle.load(open(filename, 'rb'))

def is_valid_url(url_string):
    url_string = url_string.strip() #To get rid of leading white space
    result = validators.url(url_string)

    if isinstance(result, validators.ValidationFailure):
        return False

    if "youtube" not in url_string:
        return False

    return result

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANER, '', raw_html)
    return cleantext

def build_yt_api():
    return build('youtube', 'v3', developerKey=API_KEY)

def get_video_id(url):
    video_prop = url.split("?")[1].split("&")
    video_id = video_prop[0][2:]

    return video_id

def get_comments(yt, v_id, amt=90):
    try:
        request = yt.commentThreads().list(
            part='snippet',
            videoId=v_id
            ).execute()
    except HttpError:
        return False
    
    comments = []

    #Getting the comments
    while len(comments) < amt:
        for item in request['items']:
            comment = item['snippet']['topLevelComment']
            text = comment['snippet']['textDisplay']
            text = cleanhtml(text) #Getting rid of html stuff

            #Making sure to not let spam comments in our list
            if text not in comments:
                comments.append(text)

        if ('nextPageToken' in request) and len(comments)<amt: 
            request = yt.commentThreads().list(
                part='snippet',
                videoId=v_id,
                textFormat="plainText",
                pageToken=request['nextPageToken']
                ).execute()
        # if no request is received, break
        else:
            break
    
    return comments

import re
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def sentiment_predictor(inputStr, loaded_model, cv):
    inputStr = text_transformation(inputStr)
    transformed_text = cv.transform(inputStr)
    prediction = loaded_model.predict(transformed_text)
    return prediction

def get_emotions(cv, loaded_model, comments):
    emotion = ''
    c_class = ""
    pos, neg = 0, 0

    comments_with_emotion = []

    for c_id, comment in enumerate(comments):
        emotion_of_comment = sentiment_predictor([comment], loaded_model, cv)
        if emotion_of_comment  == 4:
            emotion = 'p'
            pos += 1
            c_class = "success"

        if emotion_of_comment == 0:
            emotion = 'n'
            neg += 1
            c_class = "error"
        
        comments_with_emotion.append([c_class, f"[Comment#{c_id+1}, emotion:{emotion}]: {comment}"])
            
    return comments_with_emotion, pos, neg
