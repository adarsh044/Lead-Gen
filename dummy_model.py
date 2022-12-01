import random
def dummy_ai(all_files , services, search_phrases):
    k = 0
    word = "hello"
    relevant_words = []
    for i in range(10):
        relevant_words.append(word)
    all_company_rankings = []
    for company_name in all_files:
        if k >= 10:
            break
        k += 1
        # print(all_topic_words[max_index])
        #adding companies to the dictionaries to be returned but the rank isn't modified yet
        company_dictionary = {}
        company_dictionary["organization_name"] = company_name
        company_dictionary["priority_ranking"] = str(random.randrange(20))
        company_dictionary["keywords"] = relevant_words
        # ranking_dict[company_name] = [0, all_topic_words[max_index]]
        all_company_rankings.append(company_dictionary)

    return all_company_rankings