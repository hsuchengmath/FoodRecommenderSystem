




import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from multiprocessing import Pool



app = Flask(__name__)
CORS(app)



@app.route('/test', methods=['POST'])
def ServicePost():
    json_format = {'return' : 200}
    return jsonify(json_format)