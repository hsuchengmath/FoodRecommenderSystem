


import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Pool


from NewAccount import InsertNewAccount



app = Flask(__name__)
CORS(app)










@app.route('/RecommendService', methods=['POST'])
def RecommendService():
    insertValues = request.get_json()
    # check input data 
    try:
        account = insertValues['account']
        password = insertValues['password']
        latitude = insertValues['latitude']
        longitude = insertValues['longitude']
        if 'food_tags' is in insertValues and isinstance(insertValues['food_tags'], list) is True:
            food_tags = insertValues['food_tags']
        else:
            food_tags = None
    except:
        account = None
        password = None 
        latitude = None
        longitude = None

    if account is not None and password is not None and latitude is not None and longitude is not None:
        # recommendation
        a = 0
        MesssageCode = 200
    else:
        MesssageCode = -888
    json_format = {'return' : MesssageCode}
    return jsonify(json_format)





@app.route('/AccountService', methods=['POST'])
def AccountService():
    insertValues = request.get_json()
    # check input data 
    try:
        account, password = str(insertValues['account']), str(insertValues['password'])
    except:
        account, password = None, None
    
    # build account
    if account is not None and password is not None:
        # build new account
        InsertNewAccount(account=account, password=password)
        MesssageCode = 200
    else:
        MesssageCode = -888
    # output
    json_format = {'return' : MesssageCode}
    return jsonify(json_format)





@app.route('/TriggerUpdateMenu', methods=['POST'])
def TriggerUpdateMenu():
    insertValues = request.get_json()
    # check input data 
    try:
        SpecialCase = insertValues['SpecialCase']
        MesssageCode = 200
    except:
        MesssageCode = -888
    
    # check task is special type or overall type
    if SpecialCase is not None:
        # only update single case
        latitude = insertValues['latitude']
        longitude = insertValues['longitude']
    else:
        # update all history menu
        a = 0

    # output 
    json_format = {'return' : MesssageCode}
    return jsonify(json_format)










if __name__ == '__main__': 
    app.run(threaded=True, port=9020, host="0.0.0.0")





