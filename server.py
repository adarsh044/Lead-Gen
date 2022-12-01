import os
from flask import Flask, request, jsonify
from preprocess_json import prerocess_input_json
from lead_gen_end2end import lda_2_ranking
# from dummy_model import dummy_ai
import sys

app = Flask(__name__)

@app.route("/priority", methods=["POST"])

def priority_list():
    
    json_file = request.json
    #preprocess the json file
    company_dictionary, services, search_phrases = prerocess_input_json(json_file)
    print(company_dictionary, services, search_phrases)
    prioritized_list = lda_2_ranking(company_dictionary, services, search_phrases)
    # prioritized_list = dummy_ai(company_dictionary, services, search_phrases)
    result = {"keyword":prioritized_list}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
