

def lead_gen(company_list=list_of_companies_for_ai_model, search_phrases=words_to_search, revenue_filter=range_of_revenue, region_filter=region):
    # code for ai model
    # company_list = list_of_companies_for_ai_model
    # search_phrases = words_to_search
    # revenue filter = eg - 100M - 500M
    # region filter = eg - Japan
    prioritized_companies = {
        
        "google" : [18 , ["investment", "merger", "acquisition", "amount"]],
        "amazon" : [1, ["products", "sell", "budget"]],
        "chrysCaptival" : [5, ["investment", "merger"]],
        "facebook" : [9 , ["advertisements" , "friends", "poke"]]
    }
    
    
    # this part is just for explaintation
    # explaination starts here
    '''
    prioritized_company_list is a python dictionary where 
    key = company name -> string
    value = [[ranking -> integer], [list of relevant phrases -> list of strings]]
    '''
    # explaination ends here

    return prioritized_companies


# changes

'''
instead of matching 1 search phrase per service , match many
'''

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

# 

from sklearn.metrics.pairwise import cosine_similarity

all_files = os.listdir('/home/adarsh/nlp_sales/lead_gen_text_assets/')

root="/home/adarsh/nlp_sales/lead_gen_text_assets/"

def generate_LDA(file_name):
    
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))

    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    def compute_coherence_values(dictionary, corpus, texts, limit, start, step=1):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics, 
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=16,
                                                   passes=200,
                                                   alpha='auto',
                                                   per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            print("Iteration Completed -- ", num_topics)
            print('Coherence value: ', coherencemodel.get_coherence())
        return model_list, coherence_values
    
    file = open(file_name,'r')
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
                
    bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_trigrams(data_words_nostops)
    
    wordnet_lemmatizer = WordNetLemmatizer()
    final = []
    for i in range(len(data_words_bigrams)):
        x = []
        for j in range(len(data_words_bigrams[i])):
            k = wordnet_lemmatizer.lemmatize(data_words_bigrams[i][j])
            x.append(k)
        final.append(x)
    data_lemmatized = final
    
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    print('Pre-processing Completed!')
    
    print('\nFinding optimal number for the topics...')    
    start = 2
    limit = 10
   
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, 
                                                            start=start, limit=limit)
    idx = coherence_values.index(max(coherence_values))
    lda_model = model_list[idx]
    print('Optimal Topics: ', (idx+start))
    print('Coherence Score: ', max(coherence_values))
    print('Perplexity: ', lda_model.log_perplexity(corpus))
        
#     vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
#     pyLDAvis.save_html(vis, './assets/google_sleep_mosquito_jio_no_merger.html')
#     pyLDAvis.save_html(vis, './assets/google_file_1_2_combined_29_sept.html')
#     pyLDAvis.save_html(vis, './assets/lda/'+date+'/google_file_1_2_combined_29_sept_lda.html')
    print('LDA-html Created!')
    return lda_model

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/adarsh/nlp_sales/w2v_google/GoogleNews-vectors-negative300.bin.gz', binary=True)  

def get_topic_words(lda_model):
    all_topic_words = []
    for idx, topic in lda_model.show_topics(formatted=False, num_words=30):
        topic_words = [w[0] for w in topic]
        all_topic_words.append(topic_words)
    return all_topic_words

def lda_2_ranking(all_files , services):
    
    
    vectorServices = []
    ranking_dict = {}
    for keyword in services:
        vectorServices.append(w2v_model[keyword])
    i = 0
    per_company_similarity = []
    per_company_relevant_words = []
    for file in all_files:
        filename = root + file + ".txt"
        print(f"company nunber {i}")
        lda_model = generate_LDA(filename)
        all_topic_words = get_topic_words(lda_model) #words per topic
        all_topic_similarities_mean = [] # has the similarity of each topic to the serivec
        for each in all_topic_words:
            topic_similarity = [] # all words similarity per topic
            for word in each:
                if word in w2v_model.key_to_index.keys():
                    vectorWord = w2v_model[word]
                    similarity = 0
                    for keyword in vectorServices:
                        similarity += cosine_similarity(keyword.reshape(1,-1), vectorWord.reshape(1,-1))
                    average_similarity_per_keyword = similarity / len(vectorServices) 
                    topic_similarity.append(average_similarity_per_keyword)
            all_topic_similarities_mean.append(sum(topic_similarity) / len(topic_similarity))
        max_similairy = max(all_topic_similarities_mean)
        print(f"For company {i} the similarity score is {max_similairy[0]}")
        
        max_index = all_topic_similarities_mean.index(max_similairy)
            #per_company_relevant_words.append(all_topic_words[max_index])
        print(all_topic_words[max_index])
        #adding companies to the dictionaries to be returned but the rank isn't modified yet
        ranking_dict[file] = [0, all_topic_words[max_index]]
        i+=1
        per_company_similarity.append(max_similairy)

    sorted_companies = per_company_similarity.sort() 
    j = 0      
    for key,value in ranking_dict.items():
        value[0] = sorted_companies.index(per_company_similarity[j]) + 1
        j += 1

    return sorted_companies

per_company_similariy = lda_2_ranking(all_files,["merger","acquisition"])

for each in per_company_similariy:
    print(each[0])

def lead_gen_e2e(filenames=filenames, services=services, region=region, revenue=revenue):
    per_company_similariy = lda_2_ranking()

# [sorted(l).index(x) for x in l]