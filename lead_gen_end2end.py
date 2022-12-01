
import os
import numpy as np
import re
import json

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="password",
  database="lda_keywords"
)
mycursor = mydb.cursor()


root="/home/adarsh/nlp_sales/final_text_db/texts/"
links_file = "/home/adarsh/nlp_sales/final_text_db/links.json"

f = open(links_file)
links = json.load(f)

path = "/home/adarsh/nlp_sales/w2v_google/GoogleNews-vectors-negative300.bin.gz"

def get_model(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
    return model
w2v_model = get_model(path)

def get_topic_words_query(filename):
    mycursor.execute(f"SELECT * FROM KEYWORDS WHERE textfile={filename}")
    all_topic_words = []
    myresults = mycursor.fetchall()
    for each in myresults:
        per_topic_words = each[1:]
        all_topic_words.append(per_topic_words)
    return all_topic_words


def csort(d):
    return d["priority_ranking"]

def lda_2_ranking(all_files , services, search_phrases):
    vectorServices = []
    for keyword in services:
        vectorServices.append(w2v_model[keyword])
    ranking_dict = {}
    i = 0
    per_company_similarity = []
    per_company_relevant_words = []
    all_company_rankings = []
    company_similarity = []
    similar_textcount_percompany = []
    for company_name in all_files:
        if company_name not in os.listdir(root):
            continue
        relevant_links = []
        print(company_name)
        textroot = root + company_name + "/"
        all_text_topics = []
        relevant_keywords_per_textfile = []
        summed_similarity_per_text = 0
        similar_textcount_company = 0
        textfile_list = os.listdir(textroot)
        for textfile in textfile_list:
            count = 0
            filename = textroot + textfile
            all_topic_words = get_topic_words_query(textfile)
            all_text_best_topic_similarity = []
            for each_topic in all_topic_words:
                topic_similarity = []
                summed_similarity = 0
                for word in each_topic:
                    if word in w2v_model.key_to_index.keys():
                        vectorword = w2v_model[word]
                        similarity = 0
                        for keyphrase in vectorServices:
                            similarity += cosine_similarity(keyphrase.reshape(1,-1), vectorword.reshape(1,-1))
                        if similarity >= 0.5:
                            print(word)
                            count += 1
                # print(count)
                if count == 0:
                    continue
            # print(count)
            similar_textcount_company += count
            # print(max_similairy)
            if(count > 2):
                # print(similar_textcount)
                textfile_name = textfile.split(".")[0]
                # print(textfile)
                if textfile_name in links:
                    relevant_links.append(links[textfile_name])
        print(similar_textcount_percompany)
        similar_textcount_percompany.append(similar_textcount_company)
        sp = company_name.split('_')
        res = []
        for each in sp:
            each = each[0].upper() + each[1:]
            res.append(each)
        result = ' '.join(res)
        company_dictionary = {}
        company_dictionary["organization_name"] = result
        company_dictionary["priority_ranking"] = ""
        company_dictionary["keywords"] = relevant_links
        # company_dictionary["similarity_score"] = summed_similarity_per_text_mean
        # company_dictionary["links"] = relevant_links
        all_company_rankings.append(company_dictionary)
    # print(f'total similar texts list {similar_textcount_percompany}')
    # sorted_companies = sorted(company_similarity, reverse=True)
    # j = 0
    # for each in all_company_rankings:
    #     each["priority_ranking"] = sorted_companies.index(company_similarity[j]) + 1
    #     j += 1
    sorted_companies = sorted(similar_textcount_percompany, reverse=True)
    print(sorted_companies)
    j = 0
    prev = -1
    temp_rank = []
    for each in all_company_rankings:
        rank = sorted_companies.index(similar_textcount_percompany[j]) + 1
        if rank in temp_rank:
            rank += 1
        print(rank)
        each["priority_ranking"] = rank
        j += 1
    all_company_rankings = sorted(all_company_rankings,key=csort)
    
    return all_company_rankings

# "hdfc":"india",
#     "chrys_capital":"india",
#     "twitter":"us"
# "hdfc":"india",
#     "auctus_ycp":"india_japan",
#     "technopro_robosoft":"india_japan",
#     "twitter":"us",

# ,
    


# status - got relevant keywords for each text file
# todo - 1. get similarity score for averaged for all texts in each company and rank them
#        2  get links
        #3  json file

# "datapipe":"us",
#     "park_place_technologies":"a",
#     "eagleview":"b",
#     "liquidhub" : "c",
#     "mobilite" : "d",
#     "r1_rcm" : "e",
#     "red_river" : "g",
#     "virtusa" : "s"
