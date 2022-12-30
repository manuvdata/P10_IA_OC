import pandas as pd
#import pickle
#import urllib
import surprise
from math import *
from heapq import nlargest
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def model_recommand(model , articles, clicks, name ):
    index = list(articles.index)
    arts = clicks['click_article_id'].loc[int(name)]
    for art in arts:
        if art in index:
           index.remove(art)
    results = dict()
    for i in index:
        pred = model.predict(int(name), i)
        results[pred.iid] = pred.est
    
    return nlargest(5, results, key = results.get)


def contentBasedRecommendArticle(articles, users, name):

    articles_read = users['click_article_id'].loc[int(name)]

    if len(articles_read) == 0:
        return "L'utilisateur n'a lu aucun article"

    articles_read_embedding = articles.loc[articles_read]
# on limites à 20000 pour rester à l'offre gratuite azure
    articles = articles.drop(articles_read).head(20000)
    #articles = articles.drop(articles_read)


    matrix = cosine_similarity(articles_read_embedding, articles)

    rec = []

    for i in range(5):
        coord_x = floor(np.argmax(matrix)/matrix.shape[1])
        coord_y = np.argmax(matrix)%matrix.shape[1]

        rec.append(int(coord_y))

        matrix[coord_x][coord_y] = 0

    return rec



