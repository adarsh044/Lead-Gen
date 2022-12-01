# import os
# import json
# # import pandas as pd
# import numpy as np
# import re

# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# from functools import lru_cache
# from sklearn.metrics.pairwise import cosine_similarity

# root="/home/adarsh/nlp_sales/texts/"

# def generate_LDA(file_name):
    
#     def sent_to_words(sentences):
#         for sentence in sentences:
#             yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))

#     def remove_stopwords(texts):
#         stop_words = stopwords.words('english')
#         return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#     def make_bigrams(texts):
#         return [bigram_mod[doc] for doc in texts]

#     def make_trigrams(texts):
#         return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
#     def compute_coherence_values(dictionary, corpus, texts, limit, start, step=1):
#         coherence_values = []
#         model_list = []
#         for num_topics in range(start, limit, step):
#             model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                                    id2word=id2word,
#                                                    num_topics=num_topics, 
#                                                    random_state=100,
#                                                    update_every=1,
#                                                    chunksize=16,
#                                                    passes=200,
#                                                    alpha='auto',
#                                                    per_word_topics=True)
#             model_list.append(model)
#             coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#             coherence_values.append(coherencemodel.get_coherence())
#         return model_list, coherence_values
    
#     file = open(file_name,'r')
#     data = file.read()
#     # get words from text
#     content = data.strip()
#     content = content.replace("\ufeff", '')
#     x = content.split('\n')
#     data = [str(sent) for sent in x]
#     data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
#     data = [re.sub('\s+', ' ', sent) for sent in data]
#     data = [re.sub("\'"," ", sent) for sent in data]
#     while '' in data:
#         data.remove('')
    
#     data_words = list(sent_to_words(data))
#     for i in range(len(data_words)):
#         for word in data_words[i]:
#             if len(word) < 3:
#                 data_words[i].remove(word)
                
#     bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100)
#     trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
#     bigram_mod = gensim.models.phrases.Phraser(bigram)
#     trigram_mod = gensim.models.phrases.Phraser(trigram)
    
#     data_words_nostops = remove_stopwords(data_words)
#     data_words_bigrams = make_trigrams(data_words_nostops)
    
#     wordnet_lemmatizer = WordNetLemmatizer()
#     final = []
#     for i in range(len(data_words_bigrams)):
#         x = []
#         for j in range(len(data_words_bigrams[i])):
#             k = wordnet_lemmatizer.lemmatize(data_words_bigrams[i][j])
#             x.append(k)
#         final.append(x)
#     data_lemmatized = final
    
#     id2word = corpora.Dictionary(data_lemmatized)
#     texts = data_lemmatized
#     corpus = [id2word.doc2bow(text) for text in texts]
#     start = 2
#     limit = 10
   
#     model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, 
#                                                             start=start, limit=limit)
#     idx = coherence_values.index(max(coherence_values))
#     lda_model = model_list[idx]
#     print('LDA-html Created!')
#     return lda_model
# path = "/home/adarsh/nlp_sales/w2v_google/GoogleNews-vectors-negative300.bin.gz"

# @lru_cache(maxsize=32)
# def get_model(path):
#     model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
#     return model
# w2v_model = get_model(path)

# def get_topic_words(lda_model):
#     all_topic_words = []
#     for idx, topic in lda_model.show_topics(formatted=False, num_words=30):
#         topic_words = [w[0] for w in topic]
#         all_topic_words.append(topic_words)
#     return all_topic_words

# def lda_2_ranking(all_files , services, search_phrases):
#     vectorServices = []
#     ranking_dict = {}
#     for keyword in services:
#         vectorServices.append(w2v_model[keyword])
#     i = 0
#     per_company_similarity = []
#     per_company_relevant_words = []
#     all_company_rankings = []
#     # get company names
#     for company_name in all_files:
#         folder = root + company_name + "/"
#         per_company_per_text_similarity = []
#         # get all texts from each company
#         for each in os.listdir(folder): 
#             filename = folder + each 
#             print(filename)
        
#             # LDA
#             lda_model = generate_LDA(filename)
#             all_topic_words = get_topic_words(lda_model) #words per topic
#             # simlarity score
#             all_topic_similarities_mean = [] # has the similarity of each topic to the serivec
#             for elememt in all_topic_words:
#                 topic_similarity = [] # all words similarity per topic
#                 per_company_per_word_similarity  = []
#                 for word in elememt:
#                     if word in w2v_model.key_to_index.keys():
#                         vectorWord = w2v_model[word]
#                         summed_similarity = 0
#                         for keyword in vectorServices:
#                             # print(keyword)
#                             similarity = cosine_similarity(keyword.reshape(1,-1), vectorWord.reshape(1,-1))
#                             summed_similarity += similarity
#                             # per_company_per_word_similarity.append(similarity)
#                         average_similarity_per_keyword = summed_similarity / len(vectorServices) 
#                         topic_similarity.append(average_similarity_per_keyword)
#                         # print(max(per_company_per_word_similarity))
#             all_topic_similarities_mean.append(sum(topic_similarity) / len(topic_similarity))
#             max_similairy = max(all_topic_similarities_mean)
#         per_company_per_text_similarity.append(max_similairy)
#         # print(f"For company {i} the similarity score is {max_similairy[0]}")
#     print(per_company_per_text_similarity)
#     # break
#     '''
#         max_index = all_topic_similarities_mean.index(max_similairy)
#         # print(all_topic_words[max_index])
#         #adding companies to the dictionaries to be returned but the rank isn't modified yet
#         company_dictionary = {}
#         company_dictionary["organization_name"] = company_name 
#         company_dictionary["priority_ranking"] = ""
#         company_dictionary["keywords"] = all_topic_words[max_index]
#         company_dictionary["similariy_score"] = max_similairy
#         # ranking_dict[company_name] = [0, all_topic_words[max_index]]
#         all_company_rankings.append(company_dictionary)
#         per_company_similarity.append(max_similairy)
#         print("next company")
    
#     sorted_companies = sorted(per_company_similarity, reverse=True)
#     j = 0     
#     for company in all_company_rankings:
#         company["priority_ranking"] = sorted_companies.index(per_company_similarity[j]) + 1
#         j += 1
# '''
#     return 0

# all_files = {
#     "cryusone":"us",
#     "liquidhub":"us"
# }

# # all_files = {
# #     "hdfc":"india",
# #     "auctus_ycp":"india_japan",
# #     "technopro_robosoft":"india_japan",
# #     "twitter":"us",
# #     "google":"us"
# # }

# services = ["merger"]
# # ["india", "market", "entry"]
# search_phrases = "test"

# # all_company_rankings = lda_2_ranking(all_files,services,search_phrases)
# lda_2_ranking(all_files,services,search_phrases)

# # print(all_company_rankings)



# # "hdfc":"india",
# #     "chrys_capital":"india",
# #     "twitter":"us"
# # "hdfc":"india",
# #     "auctus_ycp":"india_japan",
# #     "technopro_robosoft":"india_japan",
# #     "twitter":"us",



# # '''
# # key variables :


# # '''

# # per_company_similariy = [] #would contain simlarity score of each text in company

# # for company_name in all_files:
# #     per_company_similariy = []
# #     for text in company_name:
# #         lda()
# #         got get_topics
# #         per_topic_similarity = []
# #         for each topic :
# #             per word similarity = []
# #              for each word :
# #                 get similarity
# #                 per word similarity.append(get similarity)

texts = [['dataart', 'deg', 'private', 'solution', 'list', 'growing', 'global', 'fastest', 'enterprise', 'europe', 'company', 'finance', 'provided', 'key', 'german', 'healthcare', 'develops', 'mobile', 'iot', 'including', 'britain', 'emerging', 'revenue', 'three', 'industry', 'investment', 'america', 'appeared', 'consultancy', 'market'], ['mobile', 'meetup', 'nasdaq', 'ocado', 'professional', 'throughout', 'employ', 'latin', 'use', 'medium', 'life', 'internet', 'including', 'implemented', 'hospitality', 'healthcare', 'financial', 'expertise', 'excellence', 'entertainment', 'enterprise', 'development', 'deep', 'customized', 'center', 'blockchain', 'ataart', 'project', 'major', 'location'], ['annual', 'nasdaq', 'travelport', 'eugene', 'ocado', 'growth', 'led', 'million', 'president', 'rate', 'record', 'fund', 'goland', 'increase', 'major', 'include', 'client', 'brand', 'revenue', 'company', 'dataart', 'use', 'market', 'intends', 'capacity', 'emerging', 'engineering', 'meetup', 'compound', 'global'], ['company', 'operation', 'russian', 'federation', 'dataart', 'june', 'stop', 'war', 'contract', 'called', 'almost', 'aggressor', 'beginning', 'everything', 'exit', 'deal', 'due', 'term', 'law', 'change', 'stopped', 'became', 'completely', 'complete', 'condemned', 'country', 'decision', 'employ', 'business', 'end'], ['metro', 'dataart', 'spain', 'utm_medium', 'marketplace', 'company', 'partner', 'thepaypers', 'com', 'allowing', 'backend', 'ensures', 'european', 'expand', 'flexibility', 'integration', 'twitter', 'launching', 'necessary', 'plan', 'platform', 'representative', 'scalability', 'september', 'stated', 'successful', 'support', 'system', 'growth', 'germany']]


flat_list = [item for sublist in texts for item in sublist]
# for sublist in texts:
#     print(sublist)
#     break

print(flat_list)
