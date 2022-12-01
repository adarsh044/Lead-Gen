import os
import json


def prerocess_input_json(file):
    data = file
    resultant_dict = {}
    for key,value in data.items():
        if(key == 'productOrService'):
            break
        new_value = value.split(",")
        region_revenue = new_value[2].replace('\xa0','').split("-")
        region = region_revenue[0][:-1]
        revenue = region_revenue[1]
        company_name = value.split("-")[0].lower()
        name = company_name.split(" ")
        name.remove('')
        name = '_'.join(name)
        resultant_dict[name] = [region,revenue]
    services = data["productOrService"]
    search_phrases = data["searchPhrase"]
    return resultant_dict , services, search_phrases

