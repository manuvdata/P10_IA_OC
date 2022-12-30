import logging
import os, uuid
import pandas as pd
import pickle
import urllib
#import surprise
from math import *
from heapq import nlargest
import function_extern as fc
#import numpy as np


import azure.functions as func
logging.info('Python HTTP trigger function processed a request.')

url = 'https://functionp9.blob.core.windows.net/p9oc/articles.pickle'
f = urllib.request.urlopen(url)
article = pickle.load(f)

url = 'https://functionp9.blob.core.windows.net/p9oc/click.pickle'
f = urllib.request.urlopen(url)
click = pickle.load(f)

url = 'https://functionp9.blob.core.windows.net/p9oc/model_SVD.pickle'
f = urllib.request.urlopen(url)
model = pickle.load(f)





def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')


    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        id = req_body.get('id')
        type = req_body.get('type')

    if isinstance(id, int) and isinstance(type, str):
        pred_recom = fc.model_recommand(model , article, click, id) if type == "cb" else fc.contentBasedRecommendArticle(article, click, id)


    #name = req.params.get('name')
    #if not name:
        #try:
            #req_body = req.get_json()
        #except ValueError:
            #pass
        #else:
            #name = req_body.get('name')

    #if name:
        #pred = fc.model_recommand(model , article, click, name)
        #pred2 = fc.contentBasedRecommendArticle(article, click, name)
      
        return func.HttpResponse(str(pred_recom),status_code=200)
    else:
        return func.HttpResponse(
             "Requete invalide .",
             status_code=400
        )