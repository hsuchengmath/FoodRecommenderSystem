


url = 'https://disco.deliveryhero.io/listing/api/v1/pandora/vendors'



import requests

 
headers = {'x-disco-client-id' : 'web'}
params = {'longitude' : 120.3025185, 
          'latitude' : 22.639473,
          'language_id' : 6,
          'customer_type' : 'regular',
          'vertical' : 'restaurants',
          'configuration': 'Original',
          'country' : 'tw',
          'include' : 'characteristics',
          'offset' : 0,
          'sort' : 'distance_asc',
          'limit':1}
'''
料理種類	cuisine:
中港	166
健康餐	225
小吃	214
披薩	165
日韓	164
東南亞	252
歐式	175
漢堡	177
甜點	176
異國	183
素食	186
美式	179
飲料	181
麵食料理	201
----
budgets
1,2,3
'''
obj = requests.get(url, headers=headers, params=params)
#print(dict(obj.json))

import json
data = json.loads(obj.text)
print(data['data']['items'][0].keys())
print('storeID : ',data['data']['items'][0]['id'])
print('address : ', data['data']['items'][0]['address'])
print('budget : ', data['data']['items'][0]['budget'])
print('name : ', data['data']['items'][0]['name'])
print('rating : ', data['data']['items'][0]['rating'])
print('review_number : ', data['data']['items'][0]['review_number'])
print('cuisines : ', data['data']['items'][0]['cuisines'])