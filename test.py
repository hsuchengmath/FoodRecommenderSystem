


import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Pool


123
app = Flask(__name__)
CORS(app)


@app.route('/test', methods=['POST'])
def ServicePost():
    insertValues = request.get_json() 
    latitude = insertValues['latitude']#22.6394088
    longitude = insertValues['longitude']#120.3025474
    url = \
    '''
        https://disco.deliveryhero.io/listing/api/v1/pandora/vendors?
        latitude={}&longitude={}&language_id={}&include={}&configuration={}
        &country={}&sort={}&vertical={}&customer_type={}
    '''
    url = url.format()
    obj = requests.get(url)
    json_format = {'return' : 200}
    return jsonify(json_format)

https://disco.deliveryhero.io/listing/api/v1/pandora/vendors?latitude=22.6394088&longitude=120.3025474&configuration=Original

if __name__ == '__main__': 
    app.run(threaded=True, port=9020, host="0.0.0.0")
