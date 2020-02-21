from flask import Flask, jsonify, abort, make_response, request, url_for
import numpy as np
import pandas as pd
import os
import csv
import sys
import re
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import pickle as p


ratingsPath = "/Users/kuanysh/Desktop/methodPro/project/ratings.csv"
moviesPath = "/Users/kuanysh/Downloads/movies.csv"
movieID_to_name = {}

def loadMovieLensLatestSmall():

        # Look for files relative to the directory we are running from
        # os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        name_to_movieID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(ratingsPath, reader=reader)

        with open(moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    movieName = row[1]
                    movieID_to_name[movieID] = movieName

        return ratingsDataset

def getMovieName(movieID):
    if movieID in movieID_to_name:
        return movieID_to_name[movieID]
    else:
        return ""

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []

    u = trainset.to_inner_uid(str(testSubject))

    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset

app = Flask(__name__)

movieids = []

@app.route('/model/', methods=['POST'])
def makecalc():


    movieIds = request.get_json()
    for id in movieIds:
        movieids.append(id['id'])

    df = pd.read_csv("/Users/kuanysh/Downloads/ratings.csv")

    uid = df['userId'].iloc[-1]+1
    for i in range (0,len(movieids)):
        df2=pd.Series([uid, movieids[i], 5.0,df.timestamp.mean()], index=df.columns )
        df=df.append(df2,ignore_index=True)

    df.to_csv("ratings.csv", index=False)
    data = loadMovieLensLatestSmall()
    trainSet = data.build_full_trainset()
    testSet = BuildAntiTestSetForUser(uid, trainSet)

    # print(testSet)
    predictions = model.test(testSet)

    recommendations = []

    print ("\nWe recommend:")
    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        intMovieID = int(movieID)
        recommendations.append((intMovieID, estimatedRating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    ans = []
    for ratings in recommendations[:10]:
        print(getMovieName(ratings[0]))
        ans.append(getMovieName(ratings[0]))

    return jsonify(ans)

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    modelfile = 'final_prediction.pickle'
    model = p.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')




