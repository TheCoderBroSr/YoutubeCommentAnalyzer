from flask import Flask, render_template, request, url_for, redirect
from func import *

note = 0
if API_KEY[0] == "Y":
    note = 1

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html', note=note)

@app.route("/processing", methods=["POST"])
def processing():
    url = request.form['url']
    video_response = -1
    positivity_perc = -1
    comments = []
    err = -1

    #Validating Url
    if not is_valid_url(url):
        err = 1
    else:
        err = 0
    
    if err == 0:
        #Loading model, cv
        model = load_file("data/trial_model.sav")
        cv = load_file("data/countvectorizer.sav")

        video_id = get_video_id(url)
        youtube = build_yt_api()
        comments = get_comments(youtube, video_id, 175)

        try:
            comments, pos_emotions, negative_emotions = get_emotions(cv, model, comments)
            print(pos_emotions, negative_emotions)

            if pos_emotions >= negative_emotions:
                video_response = 1
            else:
                video_response = 0

            positivity_perc = (pos_emotions / (pos_emotions+negative_emotions)) * 100
            positivity_perc = round(positivity_perc, 1)
        except TypeError:
            err = 2

    return render_template('index.html', emotion_perc=positivity_perc, error=err, video_response=video_response, comments=comments, note=note)