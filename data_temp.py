import os
import json
# import pandas as pd
import numpy as np
import re

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity


root="/home/adarsh/nlp_sales/lead_gen_text_assets/"

path = "/home/adarsh/nlp_sales/w2v_google/GoogleNews-vectors-negative300.bin.gz"

def get_model(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
    return model
w2v_model = get_model(path)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))
def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lda_2_ranking(all_files , services, search_phrases):
    vectorServices = []
    for keyword in services:
        vectorServices.append(w2v_model[keyword])
    
    for company_name in all_files:
        filename = root + company_name + ".txt"
        file = open(filename,'r')
        data = file.read()
        # get words from text
        content = data.strip()
        content = content.replace("\ufeff", '')
        x = content.split('\n')
        data = [str(sent) for sent in x]
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
        data = [re.sub('\s+', ' ', sent) for sent in data]
        data = [re.sub("\'"," ", sent) for sent in data]
        while '' in data:
            data.remove('')
        
        data_words = list(sent_to_words(data))
        for i in range(len(data_words)):
            for word in data_words[i]:
                if len(word) < 3:
                    data_words[i].remove(word)
        data_words_nostops = remove_stopwords(data_words)
        for words in data_words_nostops:
            if word in w2v_model.key_to_index.keys():
                vectorWord = w2v_model[word]
                similarity = cosine_similarity(keyword.reshape(1,-1), vectorWord.reshape(1,-1))
                if similarity > 0.5:
                    print(word)
   
        

all_files = {
    "hdfc":"india",
    "auctus_ycp":"india_japan",
    "technopro_robosoft":"india_japan",
    "twitter":"us",
    "google":"us"
}

services = ["merger"]
# ["india", "market", "entry"]
search_phrases = "test"
lda_2_ranking(all_files,services,search_phrases)