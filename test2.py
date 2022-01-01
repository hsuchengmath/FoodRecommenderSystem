


import requests





# # First Stage
json_format = {'StartDate': 1640359800, "Class":[{"BrandID":1,"Type":1}]}#1640007000
obj = requests.post('http://140.116.52.225:9020/test',json=json_format)
#obj = requests.post('http://192.168.36.48:5050/dcgs20/scheduling',json=json_format)
print('-----First Stage-----')
print(obj.text) 