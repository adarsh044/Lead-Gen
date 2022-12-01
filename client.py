import requests

URL = "http://127.0.0.1:5000/priority" #server url

# filepath = "./sample.json"

json_format = {
    "0": "Google - Milton, Ontario, Canada - $500M to $1B",
    "1": "Chrys Capital - Bolton, Ontario, Canada - $100M to $500M",
    "2": "Facebook , Alberta, Canada - $100M to $500M",
    "3": "HDFC - Toronto, Ontario, Canada - $10B+",
    "4": "Microsoft - St. John's, Newfoundland, Canada - $500M to $1B",
    "5": "Twitter - Quebec, Quebec, Canada - $100M to $500M",
    "productOrService": "M&A duedeligence",
    "searchPhrase": "test"
}


if __name__ == "__main__":
    # file = open(filepath,"rb")

    #packaging to perform post request
    # values = {"file": (filepath, file, "application/json")}
    response = requests.post(url=URL, json=json_format)
    data = response.json()
    print(data["keyword"])